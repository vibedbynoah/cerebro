//! Pool overhead: time per job for empty and small items.
use std::time::Instant;
fn main() {
    let p = cerebro_engine::pool::global();
    for &(items, work) in &[(64usize, 0usize), (64, 2000), (16, 0)] {
        let n = 20000;
        let t = Instant::now();
        for _ in 0..n {
            p.run(items, &|i| {
                let mut x = i as u64;
                for _ in 0..work { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); }
                std::hint::black_box(x);
            });
        }
        println!("{items} items x {work} work: {:.1} us per job ({} threads)", t.elapsed().as_secs_f64() / n as f64 * 1e6, p.threads());
    }
}
