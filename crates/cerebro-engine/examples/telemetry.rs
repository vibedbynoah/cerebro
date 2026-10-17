//! Prints two telemetry samples one second apart (check against nvidia-smi / Task Manager).
fn main() {
    let mut s = cerebro_engine::telemetry::Sampler::new();
    s.sample();
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{}", serde_json::to_string_pretty(&s.sample()).unwrap());
}
