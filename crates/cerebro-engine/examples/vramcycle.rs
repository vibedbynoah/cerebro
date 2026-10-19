//! Builds and drops GPU sessions of one model in a loop, printing free VRAM and the planner's
//! layer count each time: a dropped session must give its memory back.
//!
//! Usage: vramcycle <model.gguf> [rounds]

use cerebro_engine::model::{Model, Session};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("model");
    let rounds: usize = a.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let m = Model::load(std::path::Path::new(&path)).expect("load");
    let free = || cerebro_engine::hw::gpus().first().map_or(0.0, |g| g.vram_free as f64 / (1u64 << 30) as f64);
    println!("before: {:.2} GiB free", free());
    for r in 0..rounds {
        let s = Session::new_gpu_with(&m, 5120, None).expect("gpu session");
        let layers = s.gpu.as_ref().map_or(0, |g| g.gpu_layers());
        println!("round {r}: {layers} layers on the GPU, {:.2} GiB free while loaded", free());
        drop(s);
        println!("round {r}: {:.2} GiB free after drop", free());
    }
}
