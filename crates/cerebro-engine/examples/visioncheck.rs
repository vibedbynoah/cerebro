//! Checks the vision encoder against the reference implementation's tensor dumps
//! (`llama-mtmd-debug -p encode -n <side> --image <kind>`, saved as `encode-<kind>-<side>.txt`):
//! the printed corner values and the sums of the position-embedded input, the first and last
//! layer outputs and the final embedding.
//!
//! Usage: visioncheck <mmproj.gguf> <dumps folder>

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use cerebro_engine::vision::VisionModel;

/// One dumped tensor: its printed rows (first and last few, corner values each) and its sum.
struct Dump {
    shape: Vec<usize>,
    rows: Vec<Vec<f32>>,
    sum: f64,
}

fn parse(text: &str) -> (HashMap<String, Dump>, String) {
    let mut out = HashMap::new();
    let mut last = String::new();
    let mut cur: Option<(String, Dump)> = None;
    for line in text.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("common_debug_cb_eval:") {
            let name = rest.trim().split(" = ").next().unwrap_or("").trim().to_owned();
            let shape: Vec<usize> = rest.rsplit_once("= {").map(|(_, s)| s.trim_end_matches('}').split(',').filter_map(|v| v.trim().parse().ok()).collect()).unwrap_or_default();
            cur = Some((name, Dump { shape, rows: Vec::new(), sum: 0.0 }));
        } else if let Some((_, d)) = cur.as_mut() {
            let l = l.trim_end_matches(',');
            if l.starts_with('[') && l.ends_with(']') && l.len() > 2 {
                let vals: Vec<f32> = l.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|v| v.trim().parse().ok()).collect();
                if !vals.is_empty() {
                    d.rows.push(vals);
                }
            } else if let Some(s) = l.strip_prefix("sum = ") {
                d.sum = s.trim().parse().unwrap_or(0.0);
                let (name, d) = cur.take().expect("current");
                last = name.clone();
                out.insert(name, d);
            }
        }
    }
    (out, last)
}

/// Compares our tensor (tokens x width) with a dump; returns (max error of the printed values,
/// each relative to its own size plus the tensor's rms, relative sum error).
fn compare(ours: &[f32], width: usize, d: &Dump) -> (f64, f64) {
    let n = ours.len() / width;
    let picks_r: Vec<usize> = if n <= 6 { (0..n).collect() } else { vec![0, 1, 2, n - 3, n - 2, n - 1] };
    let picks_c: Vec<usize> = if width <= 6 { (0..width).collect() } else { vec![0, 1, 2, width - 3, width - 2, width - 1] };
    let scale = (ours.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / ours.len() as f64).sqrt().max(1e-3);
    let mut worst = 0f64;
    for (ri, r) in picks_r.iter().enumerate() {
        let Some(row) = d.rows.get(ri) else { break };
        for (ci, c) in picks_c.iter().enumerate() {
            if let Some(&want) = row.get(ci) {
                worst = worst.max((ours[r * width + c] as f64 - want as f64).abs() / (scale + (want as f64).abs()));
            }
        }
    }
    let sum: f64 = ours.iter().map(|v| *v as f64).sum();
    let abs: f64 = ours.iter().map(|v| (*v as f64).abs()).sum();
    (worst, (sum - d.sum).abs() / abs.max(1e-6))
}

fn main() {
    let mut a = std::env::args().skip(1);
    let proj = a.next().expect("mmproj path");
    let dumps = a.next().expect("dumps folder");
    let t = Instant::now();
    let vm = VisionModel::load(Path::new(&proj)).expect("load projector");
    println!("loaded in {:.2} s", t.elapsed().as_secs_f64());
    let mut failed = false;
    let mut files: Vec<_> = std::fs::read_dir(&dumps).expect("dumps").flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    for f in files {
        let stem = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let parts: Vec<&str> = stem.split('-').collect();
        if parts.len() != 3 || parts[0] != "encode" {
            continue;
        }
        let (kind, side): (&str, usize) = (parts[1], parts[2].parse().expect("side"));
        let (dump, last) = parse(&std::fs::read_to_string(&f).expect("read dump"));
        let mut traced: HashMap<String, Vec<f32>> = HashMap::new();
        let input = cerebro_engine::vision::debug_image(kind, side).expect("test image");
        let t = Instant::now();
        let e = vm
            .encode_traced(&input, &mut |name, v| {
                traced.insert(name.to_owned(), v.to_vec());
            })
            .expect("encode");
        println!("{kind} {side}x{side}: {} x {} tokens in {:.2} s", e.cols, e.rows, t.elapsed().as_secs_f64());
        let n_layers = traced.keys().filter(|k| k.starts_with("layer_out.")).count();
        let checks = [
            ("inp_pos_emb".to_owned(), "inp_pos_emb".to_owned()),
            ("layer_out.0".to_owned(), "layer_out-0".to_owned()),
            (format!("layer_out.{}", n_layers - 1), format!("layer_out-{}", n_layers - 1)),
            ("out".to_owned(), last.clone()),
        ];
        for (ours, theirs) in checks {
            let (Some(v), Some(d)) = (traced.get(&ours), dump.get(&theirs)) else {
                println!("  {ours}: missing ({theirs})");
                failed = true;
                continue;
            };
            let width = d.shape.first().copied().unwrap_or(1);
            let (val, sum) = compare(v, width, d);
            if std::env::var("VERBOSE").is_ok() {
                let n = v.len() / width;
                for (ri, r) in (if n <= 6 { (0..n).collect::<Vec<_>>() } else { vec![0, 1, 2, n - 3, n - 2, n - 1] }).iter().enumerate() {
                    let cols = [0, 1, 2, width - 3, width - 2, width - 1];
                    println!("    ours   {:?}", cols.iter().map(|c| (v[r * width + c] * 1e4).round() / 1e4).collect::<Vec<_>>());
                    println!("    theirs {:?}", d.rows.get(ri));
                }
            }
            let ok = val < 0.08 && sum < 0.02;
            failed |= !ok;
            println!("  {:<14} max value error {:.4}, sum error {:.4}  {}", ours, val, sum, if ok { "ok" } else { "FAIL" });
        }
    }
    // Speed on a full-size image.
    let side = 768;
    let t = Instant::now();
    let e = vm.encode_debug("rainbow", side).expect("encode");
    println!("{side}x{side}: {} tokens in {:.2} s", e.cols * e.rows, t.elapsed().as_secs_f64());
    if failed {
        std::process::exit(1);
    }
}
