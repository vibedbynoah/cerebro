//! Host-to-device bandwidth: pinned vs pageable, 3 MB (one expert) and 64 MB transfers.
use std::time::Instant;
fn main() {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    for &mb in &[3usize, 64] {
        let n = mb << 20;
        let mut pinned = unsafe { ctx.alloc_pinned::<u8>(n) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(1);
        let pageable = vec![1u8; n];
        let mut dev = s.alloc_zeros::<u8>(n).unwrap();
        let reps = (512 / mb).max(4);
        for (name, pin) in [("pinned", true), ("pageable", false)] {
            s.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                if pin { s.memcpy_htod(&pinned, &mut dev).unwrap() } else { s.memcpy_htod(&pageable, &mut dev).unwrap() }
            }
            s.synchronize().unwrap();
            let secs = t.elapsed().as_secs_f64();
            println!("{mb} MB {name}: {:.1} GB/s, {:.0} us per copy", (n * reps) as f64 / secs / 1e9, secs / reps as f64 * 1e6);
        }
    }
}
