//! Low-latency worker pool for the per-layer matrix work. Decoding issues hundreds of small parallel
//! jobs per token (a few hundred microseconds each), separated by waits on the GPU; a general pool
//! that puts idle threads to sleep spends much of each job waking them up. Here workers spin
//! between jobs while generation is running and only start sleeping after a period with no work.
//!
//! A job completes when all its items are done, not when every worker has checked in: a worker the
//! OS has descheduled (common on hybrid P/E-core CPUs) simply misses the job instead of stalling it.
//!
//! `run(n, f)` calls `f(i)` for every i in 0..n across the workers and the calling thread, and
//! returns when all calls are done.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    gen: usize,
}

struct Shared {
    gen: AtomicUsize,
    /// The current job (on the caller's stack); null when none. Lock-free: a mutex here put
    /// contending workers to sleep and cost several times the job itself on 12+ threads.
    job: AtomicPtr<Job>,
    next: AtomicUsize,
    completed: AtomicUsize,
    /// Workers that took the current job's pointer and may still call it.
    active: AtomicUsize,
    stop: AtomicBool,
}

// SAFETY: the job pointer is only used by workers counted in `active`, and `run` waits for `active`
// to reach zero after clearing the job, so the closure outlives every use; the closure is Sync.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: usize,
    handles: Vec<JoinHandle<()>>,
    /// Serialises callers (the pool runs one job at a time).
    lock: Mutex<()>,
}

/// Idle workers spin briefly and then yield their time slice between checks, so a thread with real
/// work (the caller, the GPU driver, the UI) always gets the core. Pure spinning measured 1.9 tok/s
/// against 11.4 for yielding on Qwen3-Coder-30B-A3B with 16 threads (i9-12900H): the spinners took
/// the cores and power budget from the serial parts of each layer. After a longer idle period workers
/// sleep instead, so the pool costs nothing between requests.
const SLEEP_AFTER: Duration = Duration::from_millis(200);

fn drain(sh: &Shared, f: &(dyn Fn(usize) + Sync), n: usize) {
    loop {
        let i = sh.next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
        sh.completed.fetch_add(1, Ordering::Release);
    }
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let workers = threads.saturating_sub(1);
        let shared = Arc::new(Shared {
            gen: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        });
        let handles = (0..workers)
            .map(|_| {
                let sh = shared.clone();
                std::thread::spawn(move || worker(&sh))
            })
            .collect();
        Self { shared, workers, handles, lock: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if self.workers == 0 || n == 1 {
            (0..n).for_each(f);
            return;
        }
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let sh = &*self.shared;
        // SAFETY: lifetime erased; see the Send/Sync note on Shared.
        let fp: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync)) };
        let gen = sh.gen.load(Ordering::Relaxed) + 1;
        let mut job = Job { f: fp, n, gen };
        sh.next.store(0, Ordering::Relaxed);
        sh.completed.store(0, Ordering::Relaxed);
        sh.job.store(&mut job, Ordering::SeqCst);
        sh.gen.store(gen, Ordering::SeqCst);
        drain(sh, f, n);
        while sh.completed.load(Ordering::Acquire) < n {
            std::hint::spin_loop();
        }
        // A worker either registered in `active` before this store (and is waited for) or will
        // read null afterwards (SeqCst on both sides).
        sh.job.store(std::ptr::null_mut(), Ordering::SeqCst);
        while sh.active.load(Ordering::SeqCst) > 0 {
            std::hint::spin_loop();
        }
        drop(job);
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0;
    let mut idle_since = Instant::now();
    loop {
        let mut spins = 0u32;
        let g = loop {
            if sh.stop.load(Ordering::Relaxed) {
                return;
            }
            let g = sh.gen.load(Ordering::Acquire);
            if g != seen {
                break g;
            }
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else if spins % 1024 == 0 && idle_since.elapsed() > SLEEP_AFTER {
                std::thread::sleep(Duration::from_millis(1));
            } else {
                std::thread::yield_now();
            }
        };
        seen = g;
        sh.active.fetch_add(1, Ordering::SeqCst);
        let jp = sh.job.load(Ordering::SeqCst);
        // SAFETY: registered in `active` before reading the pointer, so the job is alive if non-null.
        if let Some(j) = unsafe { jp.as_ref() } {
            if j.gen == g {
                drain(sh, unsafe { &*j.f }, j.n);
            }
        }
        sh.active.fetch_sub(1, Ordering::SeqCst);
        idle_since = Instant::now();
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// The process-wide pool: `CEREBRO_THREADS` threads, or the logical CPU count minus four, leaving
/// room for the OS, the GPU driver and the UI.
pub fn global() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| {
        let n = crate::switches::get(crate::switches::CPU_THREADS).max(1) as usize;
        Pool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_once_across_many_jobs() {
        let p = Pool::new(4);
        for round in 0..200 {
            let n = [1usize, 3, 17, 1000][round % 4];
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            p.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }
}
