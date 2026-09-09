//! Astra Memory CLI (binary) — flat vector store cross-check against Python.
//!
//! Subcommands:
//!   `astramem stats --store <store.json>` — print record count, dims, kinds
//!   `astramem query --store <store.json> --query-emb <b64> --k <N>` — top-k cosine search
//!   `astramem query --store <store.json> --text <query> --embedder-hash <seed>` — hash embedder demo
//!
//! Output is JSON lines for easy parsing by the cross-check script.

use astra_rt::memory::{decode_b64_f32, flat_search, load_store_json};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(2);
    }
    match args[1].as_str() {
        "stats" => cmd_stats(&args[2..]),
        "query" => cmd_query(&args[2..]),
        "--help" | "-h" => print_usage(),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_usage();
            process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!(
        "astramem — Astra memory vector CLI\n\
usage:\n  astramem stats --store <store.json>\n  astramem query --store <store.json> --query-emb <b64> --k <N>\n  astramem query --store <store.json> --text <query> --embedder-hash <seed>"
    );
}

fn parse_args(args: &[String], keys: &[&str]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            let key = &args[i][2..];
            if keys.contains(&key) {
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    out.insert(key.to_string(), args[i + 1].clone());
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

use std::collections::HashMap;

fn cmd_stats(args: &[String]) {
    let opts = parse_args(args, &["store"]);
    let Some(store) = opts.get("store") else {
        eprintln!("error: --store required");
        process::exit(2);
    };
    let recs = load_store_json(store).unwrap_or_else(|e| {
        eprintln!("error loading store: {e}");
        process::exit(1);
    });
    let mut kinds = HashMap::new();
    for r in &recs {
        *kinds.entry(r.kind.clone()).or_insert(0) += 1;
    }
    let dim = recs.first().map(|r| r.embedding.len()).unwrap_or(0);
    println!(
        "{{\"records\":{}, \"dim\":{}, \"kinds\":{}}}",
        recs.len(),
        dim,
        serde_json::to_string(&kinds).unwrap()
    );
}

fn cmd_query(args: &[String]) {
    let opts = parse_args(args, &["store", "query-emb", "k", "text", "embedder-hash"]);
    let Some(store) = opts.get("store") else {
        eprintln!("error: --store required");
        process::exit(2);
    };
    let recs = load_store_json(store).unwrap_or_else(|e| {
        eprintln!("error loading store: {e}");
        process::exit(1);
    });
    let k = opts.get("k").and_then(|s| s.parse().ok()).unwrap_or(8);
    let q: Vec<f32> = if let Some(emb) = opts.get("query-emb") {
        decode_b64_f32(emb).unwrap_or_else(|e| {
            eprintln!("error decoding query-emb: {e}");
            process::exit(1);
        })
    } else if let Some(text) = opts.get("text") {
        // Simple deterministic hash embedding (seed = embedder-hash or 1)
        let seed = opts.get("embedder-hash").and_then(|s| s.parse().ok()).unwrap_or(1);
        hash_embed(text, seed, recs.first().map(|r| r.embedding.len()).unwrap_or(64))
    } else {
        eprintln!("error: either --query-emb or --text required");
        process::exit(2);
    };
    let hits = flat_search(&recs, &q, k);
    for (idx, score) in hits {
        let r = &recs[idx];
        println!(
            "{{\"id\":\"{}\",\"kind\":\"{}\",\"score\":{:.6},\"content\":{}}}",
            r.id, r.kind, score, serde_json::to_string(&r.content).unwrap()
        );
    }
}

/// Very simple deterministic hash embedding (for demo/query without Python).
/// Not cryptographic; matches HashEmbedder spirit: 64-bit hash → dim.
fn hash_embed(text: &str, seed: u64, dim: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    seed.hash(&mut h);
    let mut out = Vec::with_capacity(dim);
    let mut x = h.finish();
    for _ in 0..dim {
        // xorshift64*
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        out.push((x as f32) / (u64::MAX as f32) * 2.0 - 1.0);
    }
    // L2 normalise
    let n = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        for v in &mut out {
            *v /= n;
        }
    }
    out
}