//! Self-test of the GPU expert kernels against the CPU ones on a real MoE model.
fn main() {
    std::env::set_var("CEREBRO_EXPERT_VRAM_MB", "64");
    let p = std::env::args().nth(1).expect("model");
    let m = cerebro_engine::model::Model::load(std::path::Path::new(&p)).expect("load");
    let mut g = cerebro_engine::gpu::GpuState::load(&m, 256).expect("gpu");
    println!("{:?}", g.check_expert_kernels(&m).expect("check"));
}
