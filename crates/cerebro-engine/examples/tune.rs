//! Tunes the processor settings for this machine on a model and prints every trial.
//! cargo run --release --example tune -- MODEL.gguf [cpu|gpu|split:N]
use std::sync::atomic::AtomicBool;

use cerebro_engine::model::{Model, Session};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Model::load(std::path::Path::new(&a[1])).expect("model");
    let mut s = match a.get(2).map_or("cpu", String::as_str) {
        "cpu" => Session::new(&m, 2048),
        "gpu" => Session::new_gpu(&m, 2048).expect("gpu"),
        x => Session::new_gpu_with(&m, 2048, Some(x.trim_start_matches("split:").parse().expect("split:N"))).expect("split"),
    };
    println!("machine: {}", cerebro_engine::hw::machine_key());
    println!("cores: {:?}", cerebro_engine::hw::cores());
    let o = cerebro_engine::tune::tune(&m, &mut s, &AtomicBool::new(false), &mut |t| {
        println!("  {:9} {:28} {:6.1} tok/s  {:?}", t.phase, t.label, t.tps, t.reps.iter().map(|r| (r * 10.0).round() / 10.0).collect::<Vec<_>>());
    })
    .expect("tune");
    println!("start {:.1} -> tuned {:.1} tok/s ({:+.1}%), kept {}: {}", o.start_tps, o.tuned_tps, o.gain * 100.0, o.kept, o.note);
    println!("settings: {:?}  ({:.0} s)", o.settings.iter().map(|s| (s.id, s.value)).collect::<Vec<_>>(), o.seconds);
}
