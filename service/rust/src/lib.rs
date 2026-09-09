//! `astra-rt` library crate — self-contained runtime pieces for Astra.
//!
//! Modules:
//! - `sha256`: FIPS 180-4 digest (checkpoint fingerprint verification).
//! - `memory`: flat exact vector store (cosine) cross-checked against the
//!   Python memory engine (Phase 4, GAP-4).

pub mod memory;
pub mod sha256;