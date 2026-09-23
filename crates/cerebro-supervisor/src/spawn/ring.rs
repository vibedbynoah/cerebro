//! Bounded in-memory tail of a child's recent output, and the bounded reader that fills it.
//!
//! Two independent bounds are needed, not one: the number of lines retained (so a long-running
//! server does not grow memory forever) and the size of any single line (so one pathological line —
//! a server that logs a multi-gigabyte JSON blob with no newline, say — cannot grow memory without
//! bound before the per-line cap even has a chance to apply). [`BoundedLineReader`] enforces the
//! second bound while reading, byte by byte, *before* a line is ever assembled into a `String`;
//! [`LineRing`] enforces the first, afterward.

use std::collections::VecDeque;
use std::io::Read;

/// Marker appended to a line that was cut short at [`BoundedLineReader`]'s byte cap.
const TRUNCATED_SUFFIX: &str = "…[truncated]";

/// A fixed-capacity tail of the most recent output lines, for crash diagnostics.
///
/// Not a general-purpose ring buffer: it exists to answer exactly one question — "what did this
/// child print right before it died?" — cheaply and with a memory footprint that does not grow with
/// how long the child has been running.
#[derive(Debug)]
pub(super) struct LineRing {
    lines: VecDeque<String>,
    capacity: usize,
}

impl LineRing {
    /// A ring retaining at most `capacity` lines. `capacity` of `0` is treated as `1`: a ring that
    /// could hold no diagnostic output at all would defeat its own purpose.
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Append one already-decoded, already-length-bounded line, evicting the oldest line first if
    /// the ring is full.
    pub(super) fn push(&mut self, line: String) {
        if self.lines.len() >= self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// A snapshot of every retained line, oldest first.
    pub(super) fn snapshot(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

/// Read `\n`-delimited lines from `reader`, decoding each lossily as UTF-8 and handing it to
/// `on_line`, without ever holding more than `max_line_bytes` bytes of any single line in memory.
///
/// # Why this cannot be `BufRead::read_line` plus a length check afterward
///
/// `read_line` (or `read_until`) grows its buffer to hold the *entire* line before returning it —
/// for a line with no newline in it at all, that means the whole remaining output, however large,
/// landing in one `String` before this function ever gets a chance to truncate it. The bound has to
/// be enforced while the line is being *accumulated*, not after: once `current` reaches
/// `max_line_bytes`, every further byte up to the next `\n` is counted (so the truncation marker
/// added by `finish_line` reports honestly) but never pushed into the buffer, so the buffer itself
/// never exceeds `max_line_bytes`.
///
/// Reads in fixed-size chunks off the underlying `Read` rather than one byte at a time, since a
/// child's stdout/stderr pipe is typically not itself a `BufRead`.
///
/// Runs until `reader` reaches EOF (an error is also treated as "no more data", since a background
/// log reader has nowhere to propagate it once its child has already exited) or until `on_line`
/// asks to stop by continuing to be called — it is not given a way to abort early, because every
/// caller in this crate wants every line for the log file even once the ring above has all it needs.
pub(super) fn read_bounded_lines<R: Read>(
    mut reader: R,
    max_line_bytes: usize,
    mut on_line: impl FnMut(String),
) {
    let max_line_bytes = max_line_bytes.max(1);
    let mut chunk = [0u8; 8192];
    let mut current: Vec<u8> = Vec::new();
    let mut current_overflow: usize = 0;

    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        for &byte in &chunk[..n] {
            if byte == b'\n' {
                on_line(finish_line(&mut current, &mut current_overflow));
            } else if current.len() < max_line_bytes {
                current.push(byte);
            } else {
                current_overflow += 1;
            }
        }
    }
    // A final line with no trailing newline (the common case for a process that was just killed
    // mid-write) is still worth keeping — a stack trace's last frame matters exactly as much as an
    // earlier one.
    if !current.is_empty() || current_overflow > 0 {
        on_line(finish_line(&mut current, &mut current_overflow));
    }
}

/// Decode the accumulated bytes of one line, append a truncation marker naming exactly how many
/// bytes were dropped when the line ran over the cap, and reset both accumulators for the next
/// line.
fn finish_line(current: &mut Vec<u8>, overflow: &mut usize) -> String {
    // llama.cpp is a C program writing through the CRT in text mode, so every line reaching this
    // function arrives as "...\r\n" — measured against the b10751 fixture's `--version` output,
    // which is CRLF throughout. Splitting on `\n` alone (as the caller does) would leave that `\r`
    // inside every retained line, where it would reach `recent_lines()` and the crash log. Only
    // meaningful when the line was *not* truncated: a truncated line's trailing bytes, `\r`
    // included, were counted into `overflow` above and never made it into `current` at all.
    if *overflow == 0 && current.last() == Some(&b'\r') {
        current.pop();
    }
    let mut line = String::from_utf8_lossy(current).into_owned();
    if *overflow > 0 {
        line.push_str(TRUNCATED_SUFFIX);
        line.push_str(&format!(" ({overflow} more bytes)"));
    }
    current.clear();
    *overflow = 0;
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn ring_evicts_oldest_once_over_capacity() {
        let mut ring = LineRing::new(3);
        for i in 0..5 {
            ring.push(format!("line {i}"));
        }
        assert_eq!(
            ring.snapshot(),
            vec!["line 2".to_string(), "line 3".to_string(), "line 4".to_string()]
        );
    }

    #[test]
    fn ring_of_capacity_zero_still_retains_the_most_recent_line() {
        // A ring that retained nothing at all would silently defeat its own purpose.
        let mut ring = LineRing::new(0);
        ring.push("only".to_string());
        ring.push("newest".to_string());
        assert_eq!(ring.snapshot(), vec!["newest".to_string()]);
    }

    #[test]
    fn ordinary_lines_are_reproduced_exactly() {
        let input = Cursor::new(b"first\nsecond\nthird\n".to_vec());
        let mut got = Vec::new();
        read_bounded_lines(input, 1024, |line| got.push(line));
        assert_eq!(got, vec!["first", "second", "third"]);
    }

    #[test]
    fn a_final_line_without_a_trailing_newline_is_still_captured() {
        let input = Cursor::new(b"first\nsecond".to_vec());
        let mut got = Vec::new();
        read_bounded_lines(input, 1024, |line| got.push(line));
        assert_eq!(got, vec!["first", "second"]);
    }

    #[test]
    fn a_pathological_single_line_is_bounded_in_memory_and_marked_truncated() {
        // The scenario the whole function exists for: one line, no newline until the very end,
        // vastly larger than the cap. If the accumulator were not itself bounded, this test's own
        // input would already have defeated the point by allocating the full 5 MiB up front — the
        // assertion below is on the *output*, but the fix under test is that `current` inside
        // `read_bounded_lines` never grows past `max_line_bytes`.
        let huge = vec![b'x'; 5 * 1024 * 1024];
        let mut input = huge.clone();
        input.push(b'\n');
        let mut got = Vec::new();
        read_bounded_lines(Cursor::new(input), 16, |line| got.push(line));
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with(&"x".repeat(16)), "{}", &got[0][..32.min(got[0].len())]);
        assert!(got[0].len() < 100, "line was not bounded: {} bytes", got[0].len());
        assert!(got[0].contains("truncated"), "{}", got[0]);
    }

    #[test]
    fn crlf_line_endings_do_not_leave_a_carriage_return() {
        // The regression case: a test suite that only ever fed LF-only fixtures would pass even
        // with the CR left in, which is exactly what happened before this test existed.
        let mut got = Vec::new();
        read_bounded_lines(Cursor::new(b"first\r\nsecond\r\n".to_vec()), 1024, |l| got.push(l));
        assert_eq!(got, vec!["first", "second"]);
    }

    #[test]
    fn a_truncated_crlf_line_counts_the_carriage_return_in_its_overflow() {
        // "abcd\r\n" against a 4-byte cap: "abcd" fills the buffer exactly, so the trailing `\r` is
        // the one byte that overflows — it must be counted, not silently absorbed by the
        // now-truncated-line guard in `finish_line`.
        let input = Cursor::new(b"abcd\r\n".to_vec());
        let mut got = Vec::new();
        read_bounded_lines(input, 4, |line| got.push(line));
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with("abcd"), "{}", got[0]);
        assert!(got[0].contains("(1 more bytes)"), "{}", got[0]);
    }

    #[test]
    fn overflow_count_is_exact() {
        let mut input = vec![b'a'; 10];
        input.push(b'\n');
        let mut got = Vec::new();
        read_bounded_lines(Cursor::new(input), 4, |line| got.push(line));
        // 10 bytes in, 4 kept, 6 dropped.
        assert!(got[0].contains("(6 more bytes)"), "{}", got[0]);
    }
}
