//! Fixed-shape predicate factory for native composition. No generic JSON parser
//! or caller-supplied expression enters this path; the enclosing plan reserves
//! descriptor/compiler scratch before calling it. This is an arithmetic-domain
//! predicate, not private-input membership or checkpoint provenance validation.
use super::*;
use std::fmt::Write;

struct Canonical {
    bytes: [u8; 2048],
    len: usize,
}
impl Write for Canonical {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.len.checked_add(text.len()).ok_or(std::fmt::Error)?;
        if end > self.bytes.len() {
            return Err(std::fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}
fn exact<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    if values.capacity() != count {
        return Err(Error::Allocation);
    }
    Ok(values)
}

impl Descriptor {
    pub(crate) fn simple_comparison(bits: u8, signed: bool, threshold: i64) -> Result<Self> {
        if !(1..=50).contains(&bits) {
            return Err(Error::Range);
        }
        let n = 1u64 << bits;
        let half = (n / 2) as i64;
        let expression = if signed {
            if !(-half..=half).contains(&threshold) {
                return Err(Error::Range);
            }
            Expr::SignedLt(threshold)
        } else {
            if threshold < 0 || threshold as u64 > n {
                return Err(Error::Range);
            }
            Expr::Lt(threshold as u64)
        };
        let tag = if signed { "slt" } else { "lt" };
        let mut encoded = Canonical {
            bytes: [0; 2048],
            len: 0,
        };
        write!(&mut encoded,
            "{{\"boundaries\":[0,{n}],\"coefficient_fractional\":[0,0],\"coefficient_order\":[\"constant\",\"linear\"],\"degree\":1,\"external_encoding\":\"nearest-ties-even-input-and-weight\",\"gate\":\"native-typed-comparison.v1\",\"input\":{{\"bits\":{bits},\"domain\":\"A\",\"fractional\":0,\"signed\":true}},\"internal_rounding\":\"signed-floor-only\",\"output\":{{\"arithmetic\":\"same-as-input\",\"boolean_arity\":1}},\"partition_order\":\"unsigned-canonical-half-open\",\"phi\":\"identity\",\"pieces\":[{{\"booleans\":[[\"{tag}\",{threshold}]],\"coefficients\":[0,0]}}],\"profile\":\"{SYNTHETIC_PROFILE}\",\"schema\":\"pllm.fusefss.scalar-descriptor.v1\",\"selector_semantics\":\"descriptor-defined\",\"source_lock_sha256\":\"{SOURCE_LOCK_SHA256}\",\"source_schedule\":{{}}}}")
            .map_err(|_| Error::DescriptorBytes)?;
        let mut canonical = exact(encoded.len)?;
        canonical.extend_from_slice(&encoded.bytes[..encoded.len]);
        let digest = Sha256::new()
            .chain_update(b"FuseFSS-reference-descriptor-v1\0")
            .chain_update(&canonical)
            .finalize()
            .into();
        let mut boundaries = exact(2)?;
        boundaries.extend_from_slice(&[0, n]);
        let mut pieces = exact(1)?;
        pieces.push(Piece {
            coefficients: [0, 0],
            roots: [0; MAX_BOOLEAN_OUTPUTS],
        });
        let mut expressions = exact(1)?;
        expressions.push(expression);
        let mut descriptor = Self {
            canonical,
            digest,
            word: ArithmeticType {
                bits,
                fractional: 0,
                signed: true,
            },
            boundaries,
            pieces,
            expressions,
            shape: Shape {
                source_intervals: 0,
                padded_intervals: 0,
                boolean_outputs: 1,
                queries: 0,
                vm_ops: 0,
                boolean_ands: 0,
                boolean_xors: 0,
                reference_bytes: 0,
                program_lengths: [0; MAX_BOOLEAN_OUTPUTS],
            },
        };
        descriptor.shape = compile::admission(&descriptor)?;
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_comparison_matches_bounded_parser_identity_and_shape() {
        for bits in [1, 8, 38, 50] {
            let n = 1i64 << bits;
            for signed in [false, true] {
                let thresholds = if signed {
                    [-(n / 2), -1, 0, n / 2]
                } else {
                    [0, 1, n / 2, n]
                };
                for threshold in thresholds {
                    let typed = Descriptor::simple_comparison(bits, signed, threshold).unwrap();
                    assert!(typed.canonical_bytes().len() < 2048);
                    let parsed = Descriptor::from_canonical_bytes(typed.canonical_bytes()).unwrap();
                    assert_eq!(typed.digest(), parsed.digest());
                    assert_eq!(typed.shape(), parsed.shape());
                    assert_eq!(typed.word(), parsed.word());
                    assert_eq!(typed.canonical.capacity(), typed.canonical.len());
                    assert_eq!(typed.pieces.capacity(), 1);
                    assert_eq!(typed.expressions.capacity(), 1);
                    assert_eq!(typed.boundaries.capacity(), 2);
                }
            }
        }
        for (bits, signed, threshold) in [
            (0, false, 0),
            (51, false, 0),
            (8, false, -1),
            (8, false, 257),
            (8, true, -129),
            (8, true, 129),
        ] {
            assert!(Descriptor::simple_comparison(bits, signed, threshold).is_err());
        }
    }
}
