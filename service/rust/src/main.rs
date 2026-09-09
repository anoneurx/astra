//! Astra Rust runtime — checkpoint fingerprint verification.
//!
//! Phase 3 skeleton: the Rust *engine* (forward/backward math) is a later
//! milestone. This crate is the service skeleton: it reads a checkpoint file,
//! computes its SHA-256, and verifies it against an expected fingerprint so
//! that model registry audits (docs/VERSIONING.md § 5) can be reproduced in
//! the Rust toolchain without pulling the Python reference.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -- --checkpoint <path.npz> [--expected <sha256>]
//! cargo test
//! ```

use astra_rt::sha256::{digest, hex};
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint: Option<String> = None;
    let mut expected: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                i += 1;
                checkpoint = args.get(i).cloned();
            }
            "--expected" => {
                i += 1;
                expected = args.get(i).cloned();
            }
            "--help" | "-h" => {
                println!(
                    "astra-rt service skeleton\n\
usage: astra-rt --checkpoint <path.npz> [--expected <sha256>]\n\
verifies the checkpoint's SHA-256; exits 1 on mismatch when --expected is given."
                );
                return;
            }
            other => {
                eprintln!("unknown flag: {other}");
                process::exit(2);
            }
        }
        i += 1;
    }
    let Some(ckpt) = checkpoint else {
        eprintln!("error: --checkpoint is required");
        process::exit(2);
    };

    match verify(&ckpt, expected.as_deref()) {
        Ok(true) => println!("MATCH sha256 {}", hex_digest(&ckpt)),
        Ok(false) => {
            eprintln!("MISMATCH sha256 {}", hex_digest(&ckpt));
            process::exit(1);
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn hex_digest(path: &str) -> String {
    let raw = fs::read(path).expect("read checkpoint");
    hex(&digest(&raw))
}

fn verify(path: &str, expected: Option<&str>) -> Result<bool, String> {
    let raw = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let digest = digest(&raw);
    let hex = hex(&digest);
    match expected {
        Some(exp) => Ok(hex.eq_ignore_ascii_case(exp)),
        None => {
            let n = raw.len();
            println!("sha256 {hex}  {path} ({n} bytes)");
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_empty_string() {
        let out = hex(&digest(b""));
        assert_eq!(
            out,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_classic_vectors() {
        let out = hex(&digest(b"abc"));
        assert_eq!(
            out,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let out = hex(&digest(b"The quick brown fox jumps over the lazy dog"));
        assert_eq!(
            out,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }
}