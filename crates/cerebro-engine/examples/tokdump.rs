//! Token ids of a text file, space-separated (compare with llama-tokenize --ids).
//! cargo run --release --example tokdump -- MODEL.gguf TEXT.txt
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = cerebro_engine::gguf::Gguf::open(std::path::Path::new(&a[1])).expect("open");
    let tok = cerebro_engine::tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
    let ids = tok.encode(&std::fs::read_to_string(&a[2]).expect("text"));
    println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
}
