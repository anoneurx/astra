//! Flat exact vector store for Astra memory (Phase 4, GAP-4).
//!
//! Provides:
//! - `MemRecord`: record with id, kind, content, tags, embedding (Vec<f32>)
//! - `decode_b64_f32`: base64 → little-endian f32 vector (matches Python's base64 float32)
//! - `cosine`: cosine similarity of two L2-normalised vectors
//! - `flat_search`: exact top-k cosine search over a slice of records
//! - `load_store_json`: parse the Python `memory/store/<name>.json` format
//!
//! Cross-checked against Python's `retrieve(..., weights={cosine:1, others:0})`
//! on identical embeddings. Hybrid terms (recency/confidence/kind) remain
//! Python-side in v1; Rust implements the vector core only.

/// A memory record with a pre-computed embedding.
#[derive(Debug, Clone)]
pub struct MemRecord {
    pub id: String,
    pub kind: String,
    pub content: String,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
}

/// Base64 decode a Python-style embedding blob: little-endian float32 array.
/// Python uses `base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes())`.
pub fn decode_b64_f32(s: &str) -> Result<Vec<f32>, String> {
    let raw = base64::decode(s).map_err(|e| format!("base64 decode: {e}"))?;
    if raw.len() % 4 != 0 {
        return Err(format!("embedding length {} not a multiple of 4", raw.len()));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().expect("4 bytes");
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

/// Cosine similarity of two vectors (assumed L2-normalised, or not).
/// Returns value in [-1, 1].
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Normalise a vector in-place to L2 unit length.
pub fn l2_normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Exact top-k cosine search over a flat slice of records.
/// Returns indices into `records` + cosine scores, sorted descending.
pub fn flat_search(records: &[MemRecord], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    if records.is_empty() || k == 0 {
        return vec![];
    }
    let mut scored: Vec<(usize, f32)> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            if r.embedding.is_empty() {
                None
            } else {
                Some((i, cosine(&r.embedding, query)))
            }
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Python store JSON entry (record + state).
#[derive(Debug, serde::Deserialize)]
struct StoreEntry {
    record: JsonRecord,
    state: String,
}

#[derive(Debug, serde::Deserialize)]
struct JsonRecord {
    id: String,
    kind: String,
    content: String,
    embedding: Option<String>,
    tags: Vec<String>,
}

/// Load a Python memory store JSON file and extract ACTIVE records with embeddings.
pub fn load_store_json(path: &str) -> Result<Vec<MemRecord>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("json parse: {e}"))?;
    let entries = v.get("entries")
        .and_then(|e| e.as_object())
        .ok_or_else(|| "no entries object".to_string())?;
    let mut out = Vec::new();
    for (_rid, entry) in entries {
        let e: StoreEntry = serde_json::from_value(entry.clone())
            .map_err(|e| format!("entry parse: {e}"))?;
        if e.state != "active" {
            continue;
        }
        if let Some(emb_b64) = e.record.embedding {
            let emb = decode_b64_f32(&emb_b64).map_err(|e| format!("embedding {e}"))?;
            out.push(MemRecord {
                id: e.record.id,
                kind: e.record.kind,
                content: e.record.content,
                tags: e.record.tags,
                embedding: emb,
            });
        }
    }
    Ok(out)
}

/// Small base64 implementation (no-std compatible subset).
mod base64 {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const PAD: u8 = b'=';

    #[inline]
    fn decode_table() -> [u8; 256] {
        let mut table = [0xFFu8; 256];
        for (i, &b) in ALPHABET.iter().enumerate() {
            table[b as usize] = i as u8;
        }
        table
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, &'static str> {
        let table = decode_table();
        let bytes = s.as_bytes();
        let mut out = Vec::new();
        let mut buf = 0u32;
        let mut bits = 0;
        let mut saw_pad = false;
        for &b in bytes {
            if b == PAD {
                saw_pad = true;
                // Flush any remaining full bytes from buffer.
                while bits >= 8 {
                    bits -= 8;
                    out.push((buf >> bits) as u8);
                }
                break;
            }
            let v = table[b as usize];
            if v == 0xFF {
                return Err("invalid base64 char");
            }
            buf = (buf << 6) | v as u32;
            bits += 6;
            while bits >= 8 {
                bits -= 8;
                out.push((buf >> bits) as u8);
            }
        }
        // Valid base64 with padding may leave 4 zero bits (== padding).
        // Only error if we have leftover bits AND didn't see padding.
        if bits >= 4 && !saw_pad {
            return Err("incomplete base64");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_b64_f32() {
        // Python: base64.b64encode(np.array([1.0, -2.0, 3.0], dtype=np.float32).tobytes())
        let b64 = "AACAPwAAAMAAAEBA";
        let v = decode_b64_f32(b64).expect("decode");
        assert_eq!(v.len(), 3);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - -2.0).abs() < 1e-6);
        assert!((v[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - -1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flat_search() {
        let recs = vec![
            MemRecord { id: "a".into(), kind: "fact".into(), content: "".into(), tags: vec![], embedding: vec![1.0, 0.0, 0.0] },
            MemRecord { id: "b".into(), kind: "fact".into(), content: "".into(), tags: vec![], embedding: vec![0.0, 1.0, 0.0] },
            MemRecord { id: "c".into(), kind: "fact".into(), content: "".into(), tags: vec![], embedding: vec![0.0, 0.0, 1.0] },
        ];
        let q = vec![1.0, 0.0, 0.0];
        let hits = flat_search(&recs, &q, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0); // a
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].0, 1); // b or c (0.0)
    }

    #[test]
    fn test_l2_normalise() {
        let mut v = vec![3.0, 4.0];
        l2_normalise(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let mut v = vec![0.0, 0.0];
        l2_normalise(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}