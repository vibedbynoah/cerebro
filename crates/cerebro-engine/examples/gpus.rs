//! Lists the GPUs the portable backend sees (CEREBRO_PORTABLE_BACKEND=vulkan|dx12 narrows it).
fn main() {
    for g in cerebro_engine::portable::adapters() {
        println!("{g:?}");
    }
}
