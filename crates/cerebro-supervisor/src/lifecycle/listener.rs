//! Which process owns the TCP listener on the port a supervised child was launched on.
//!
//! Preflight binds and releases the port before the child binds it (see
//! [`crate::preflight::PortRequest`]), so any local process (a second Cerebro, any `llama-server`
//! already on this machine, or another user's process, since loopback ports are machine-wide) can
//! win the port while our child is still loading. The readiness probe sends the bearer key and
//! publishes [`super::LifecycleState::Ready`] on a 200, so it must not talk to a listener it cannot
//! attribute to its own child: [`listener_owner`] answers that from the OS's own TCP table, and
//! `super::readiness::wait_ready` sends nothing until the answer is [`ListenerOwner::Ours`].

/// Who owns the TCP listener(s) accepting `127.0.0.1:port`, as far as the OS's TCP table shows.
///
/// Four states, so an answer that could not be established is never folded into "ours".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListenerOwner {
    /// Every listener accepting the port belongs to the child's own process.
    Ours,
    /// At least one listener accepting the port belongs to another process. A second socket
    /// sharing the port (`SO_REUSEADDR`, or a wildcard bind beside our specific one) counts as
    /// foreign even when our child also listens there: a connection could reach either socket.
    Foreign,
    /// Nothing listens on the port yet (the child is still loading and has not bound it).
    NoneYet,
    /// The TCP table could not be read. Treated like [`ListenerOwner::NoneYet`] by the caller: keep
    /// waiting, never assume ownership. Always the answer off Windows, where there is no table to
    /// read; that is acceptable only because spawning a supervised child is Windows-only.
    Unknown,
}

/// Who owns the listener(s) accepting `127.0.0.1:port`, judged against `pid` (the child's).
///
/// The caller must hold a handle to `pid`'s process while asking, so the PID cannot have been
/// reused by an unrelated process in the meantime.
pub(crate) fn listener_owner(pid: u32, port: u16) -> ListenerOwner {
    verdict(pid, listener_pids(port))
}

/// The pure half of [`listener_owner`]: the verdict for `pid` given the port's owner PIDs, or
/// `None` when the table could not be read.
fn verdict(pid: u32, owners: Option<Vec<u32>>) -> ListenerOwner {
    match owners {
        None => ListenerOwner::Unknown,
        Some(owners) if owners.is_empty() => ListenerOwner::NoneYet,
        Some(owners) if owners.iter().all(|&o| o == pid) => ListenerOwner::Ours,
        Some(_) => ListenerOwner::Foreign,
    }
}

/// The PIDs owning a TCP listener that accepts `127.0.0.1:port`: IPv4 listeners on `127.x.x.x` or
/// `0.0.0.0`, and IPv6 listeners on `::` or an IPv4-mapped loopback address (a dual-stack wildcard
/// socket accepts IPv4 connections too). `None` when a table could not be read.
#[cfg(windows)]
fn listener_pids(port: u16) -> Option<Vec<u32>> {
    use std::mem::{offset_of, size_of};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        MIB_TCP6ROW_OWNER_PID, MIB_TCP6TABLE_OWNER_PID, MIB_TCPROW_OWNER_PID,
        MIB_TCPTABLE_OWNER_PID,
    };

    let mut pids = Vec::new();

    let v4 = tcp_listener_table(AF_INET)?;
    let rows = TableRows {
        bytes: &v4,
        first_row: offset_of!(MIB_TCPTABLE_OWNER_PID, table),
        row_size: size_of::<MIB_TCPROW_OWNER_PID>(),
    };
    for row in rows.iter()? {
        let addr: [u8; 4] = bytes_at(row, offset_of!(MIB_TCPROW_OWNER_PID, dwLocalAddr))?;
        let accepts_loopback = addr[0] == 127 || addr == [0; 4];
        if accepts_loopback
            && local_port(row, offset_of!(MIB_TCPROW_OWNER_PID, dwLocalPort))? == port
        {
            pids.push(read_u32(
                row,
                offset_of!(MIB_TCPROW_OWNER_PID, dwOwningPid),
            )?);
        }
    }

    let v6 = tcp_listener_table(AF_INET6)?;
    let rows = TableRows {
        bytes: &v6,
        first_row: offset_of!(MIB_TCP6TABLE_OWNER_PID, table),
        row_size: size_of::<MIB_TCP6ROW_OWNER_PID>(),
    };
    for row in rows.iter()? {
        let addr: [u8; 16] = bytes_at(row, offset_of!(MIB_TCP6ROW_OWNER_PID, ucLocalAddr))?;
        let ip = std::net::Ipv6Addr::from(addr);
        let unspecified = ip.is_unspecified();
        let mapped_loopback = ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback());
        if (unspecified || mapped_loopback)
            && local_port(row, offset_of!(MIB_TCP6ROW_OWNER_PID, dwLocalPort))? == port
        {
            pids.push(read_u32(
                row,
                offset_of!(MIB_TCP6ROW_OWNER_PID, dwOwningPid),
            )?);
        }
    }

    Some(pids)
}

/// Off Windows there is no table to read; [`listener_owner`] then answers [`ListenerOwner::Unknown`].
#[cfg(not(windows))]
fn listener_pids(_port: u16) -> Option<Vec<u32>> {
    None
}

/// `AF_INET` / `AF_INET6` as `GetExtendedTcpTable` takes them (`ULONG`); the Winsock values, which
/// `windows-sys` exposes only behind a much larger feature.
#[cfg(windows)]
const AF_INET: u32 = 2;
#[cfg(windows)]
const AF_INET6: u32 = 23;

/// One `GetExtendedTcpTable(TCP_TABLE_OWNER_PID_LISTENER)` snapshot for `family`, as raw bytes.
///
/// The buffer is `u32`-backed so the table the API writes is 4-byte aligned, as its `DWORD` fields
/// require; it is then copied out as bytes and parsed by offset, so no pointer into it is ever cast
/// to a struct. The size is re-asked when the table grows between the two calls.
#[cfg(windows)]
fn tcp_listener_table(family: u32) -> Option<Vec<u8>> {
    use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, NO_ERROR};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, TCP_TABLE_OWNER_PID_LISTENER,
    };

    let mut size: u32 = 0;
    // Bounded: a table that keeps growing between calls is a lookup failure, not a reason to spin.
    for _ in 0..5 {
        // Headroom for listeners opened between the size query and the read.
        let words = (size as usize).div_ceil(4) + 64;
        let mut buf = vec![0u32; words];
        let mut cap = u32::try_from(words * 4).ok()?;
        // SAFETY: `buf` owns `cap` writable, 4-byte-aligned bytes for the whole call, and `cap` is
        // passed by pointer as the API documents; the API writes at most `cap` bytes. No pointer
        // into `buf` outlives the call.
        let rc = unsafe {
            GetExtendedTcpTable(
                buf.as_mut_ptr().cast(),
                &mut cap,
                0,
                family,
                TCP_TABLE_OWNER_PID_LISTENER,
                0,
            )
        };
        if rc == NO_ERROR {
            return Some(buf.iter().flat_map(|w| w.to_ne_bytes()).collect());
        }
        if rc != ERROR_INSUFFICIENT_BUFFER {
            return None;
        }
        size = cap;
    }
    None
}

/// The rows of one owner-PID TCP table, bounds-checked against the bytes actually returned.
#[cfg(windows)]
struct TableRows<'a> {
    bytes: &'a [u8],
    first_row: usize,
    row_size: usize,
}

#[cfg(windows)]
impl<'a> TableRows<'a> {
    /// Every row the table's `dwNumEntries` announces, or `None` if that count does not fit in the
    /// returned bytes (a malformed table is a lookup failure, never a partial answer).
    fn iter(&self) -> Option<impl Iterator<Item = &'a [u8]> + 'a> {
        let count = usize::try_from(read_u32(self.bytes, 0)?).ok()?;
        let end = count
            .checked_mul(self.row_size)?
            .checked_add(self.first_row)?;
        let rows = self.bytes.get(self.first_row..end)?;
        Some(rows.chunks_exact(self.row_size))
    }
}

/// `N` bytes of `bytes` starting at `offset`, or `None` if they are not all there.
#[cfg(windows)]
fn bytes_at<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    bytes.get(offset..offset.checked_add(N)?)?.try_into().ok()
}

#[cfg(windows)]
fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes_at(bytes, offset).map(u32::from_ne_bytes)
}

/// The local port of a row: the low 16 bits of `dwLocalPort`, stored in network byte order.
#[cfg(windows)]
fn local_port(row: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be(read_u32(row, offset)? as u16))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn a_loopback_listener_is_ours_foreign_to_any_other_pid_and_gone_once_dropped() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port");
        let port = listener.local_addr().expect("read back the port").port();
        let pids = listener_pids(port).expect("the TCP table must be readable");
        assert!(pids.contains(&std::process::id()), "port {port}: {pids:?}");
        assert_eq!(
            listener_owner(std::process::id(), port),
            ListenerOwner::Ours
        );
        // PID 4 (System) never owns a test's ephemeral listener.
        assert_eq!(listener_owner(4, port), ListenerOwner::Foreign);
        drop(listener);
        // Released: nothing listens now, which is "not yet", never "ours".
        assert_eq!(
            listener_owner(std::process::id(), port),
            ListenerOwner::NoneYet
        );
    }

    #[test]
    fn a_second_owner_on_the_port_makes_it_foreign_even_beside_ours() {
        // A socket shared through SO_REUSEADDR (or a wildcard bind beside our specific one) shows
        // up as two owner PIDs on one port: a connection may not reach our child, so never Ours.
        let me = std::process::id();
        assert_eq!(verdict(me, Some(vec![me, 4])), ListenerOwner::Foreign);
        assert_eq!(verdict(me, Some(vec![me, me])), ListenerOwner::Ours);
        assert_eq!(verdict(me, Some(Vec::new())), ListenerOwner::NoneYet);
        assert_eq!(verdict(me, None), ListenerOwner::Unknown);
    }
}
