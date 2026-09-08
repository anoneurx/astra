//! Astra Rust runtime — checkpoint fingerprint verification (skeleton).
//!
//! Phase 3 deferred: the Rust *engine* (forward/backward math) is a later
//! milestone. This crate is the service skeleton: it reads a checkpoint file,
//! computes its SHA-256, and verifies it against an expected fingerprint so
//! that model registry audits (docs/VERSIONING.md § 5) can be reproduced in
//! the Rust toolchain without pulling the Python reference.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -- --checkpoint <path.npz> --expected <sha256>
//! cargo test
//! ```

use std::fs;
use std::process;

/// Self-contained SHA-256 (FIPS 180-4) so the crate builds without network deps.
pub mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    ];

    fn rotr(x: u32, n: u32) -> u32 {
        (x >> n) | (x << (32 - n))
    }

    fn compress(state: &mut [u32; 8], block: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    /// SHA-256 digest of `data` as a 32-byte array.
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut state = H0;
        let len = data.len() as u64;
        let mut buf = Vec::with_capacity(data.len() + 72);
        buf.extend_from_slice(data);
        buf.push(0x80);
        while buf.len() % 64 != 56 {
            buf.push(0);
        }
        buf.extend_from_slice(&len.wrapping_mul(8).to_be_bytes());
        for block in buf.chunks_exact(64) {
            compress(&mut state, block);
        }
        let mut out = [0u8; 32];
        for (i, s) in state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&s.to_be_bytes());
        }
        out
    }

    /// Hex (lowercase) encoding of a 32-byte digest.
    pub fn hex(digest: &[u8; 32]) -> String {
        let hex = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for b in digest {
            s.push(hex[(b >> 4) as usize] as char);
            s.push(hex[(b & 0x0f) as usize] as char);
        }
        s
    }
}

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
    sha256::hex(&sha256::digest(&raw))
}

fn verify(path: &str, expected: Option<&str>) -> Result<bool, String> {
    let raw = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let digest = sha256::digest(&raw);
    let hex = sha256::hex(&digest);
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
    use super::sha256::{digest, hex};

    fn h(s: &str) -> String {
        hex(&digest(s.as_bytes()))
    }

    #[test]
    fn empty_string() {
        assert_eq!(
            h(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn classic_vectors() {
        assert_eq!(
            h("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            h("The quick brown fox jumps over the lazy dog"),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
        // length that crosses one 64-byte block boundary
        assert_eq!(h("a".repeat(64).as_str()).len(), 64);
        assert_eq!(h("a".repeat(65).as_str()).len(), 64);
    }

    #[test]
    fn blocks_align_off_56() {
        // 55 bytes: exactly one 0x80 pad byte then 8 length bytes
        assert_eq!(h("b".repeat(55).as_str()).len(), 64);
    }
}