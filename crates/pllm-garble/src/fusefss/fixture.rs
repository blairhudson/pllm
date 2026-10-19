//! Public frozen data copied from the independently generated local scalar
//! contract, not upstream executable code. Tests check table and descriptor hashes.
use super::*;

pub(super) const GELU_DIGEST: [u8; 32] = [
    0x5f, 0x1e, 0x88, 0x8e, 0x7b, 0xda, 0xbe, 0x3c, 0xba, 0xd4, 0xb6, 0xbb, 0xf1, 0xab, 0xd6, 0x74,
    0x9c, 0x8d, 0xff, 0x09, 0x18, 0xc6, 0xf2, 0x43, 0x91, 0x6d, 0xb0, 0x71, 0x39, 0x8e, 0x79, 0xd3,
];
pub(super) const TABLE: [u16; 256] = [
    0, 31, 62, 92, 121, 150, 177, 204, 230, 255, 280, 303, 326, 349, 370, 391, 410, 430, 448, 466,
    482, 499, 514, 529, 543, 556, 569, 581, 592, 603, 613, 623, 631, 640, 647, 654, 660, 666, 672,
    676, 680, 684, 687, 690, 692, 694, 695, 695, 696, 696, 695, 694, 693, 691, 689, 686, 683, 680,
    677, 673, 669, 664, 660, 655, 649, 644, 638, 632, 626, 620, 613, 607, 600, 593, 586, 578, 571,
    564, 556, 548, 540, 533, 525, 517, 508, 500, 492, 484, 476, 468, 459, 451, 443, 435, 426, 418,
    410, 402, 394, 386, 378, 370, 362, 354, 346, 338, 331, 323, 316, 308, 301, 294, 287, 280, 273,
    266, 259, 252, 246, 239, 233, 227, 221, 214, 209, 203, 197, 191, 186, 180, 175, 170, 165, 160,
    155, 150, 146, 141, 137, 132, 128, 124, 120, 116, 112, 108, 105, 101, 98, 94, 91, 88, 85, 82,
    79, 76, 73, 71, 68, 66, 63, 61, 58, 56, 54, 52, 50, 48, 46, 44, 42, 41, 39, 38, 36, 34, 33, 32,
    30, 29, 28, 27, 25, 24, 23, 22, 21, 20, 19, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 11, 11, 10,
    10, 9, 9, 8, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

struct Writer {
    bytes: Vec<u8>,
}
impl Writer {
    fn new() -> Result<Self> {
        let mut bytes = Vec::new();
        reserve(&mut bytes, MAX_DESCRIPTOR_BYTES)?;
        Ok(Self { bytes })
    }
    fn text(&mut self, text: &str) -> Result<()> {
        if self
            .bytes
            .len()
            .checked_add(text.len())
            .ok_or(Error::DescriptorBytes)?
            > MAX_DESCRIPTOR_BYTES
        {
            return Err(Error::DescriptorBytes);
        }
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(())
    }
    fn number(&mut self, mut n: u64) -> Result<()> {
        let mut digits = [0u8; 20];
        let mut i = digits.len();
        loop {
            i -= 1;
            digits[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        self.text(std::str::from_utf8(&digits[i..]).map_err(|_| Error::Encoding)?)
    }
}

pub(super) fn gelu() -> Result<Descriptor> {
    let n = 1u64 << 50;
    let mut bounds = Vec::new();
    reserve(&mut bounds, 512)?;
    bounds.push(0);
    for j in 1..256 {
        bounds.push(j * 64);
    }
    bounds.push(n / 2);
    for j in (1..255).rev() {
        bounds.push(n - j * 64);
    }
    bounds.push(n);
    let mut w = Writer::new()?;
    w.text("{\"boundaries\":[")?;
    for (i, &b) in bounds.iter().enumerate() {
        if i != 0 {
            w.text(",")?;
        }
        w.number(b)?;
    }
    w.text("],\"coefficient_fractional\":[12,0],\"coefficient_order\":[\"constant\",\"linear\"],\"degree\":1,\"external_encoding\":\"nearest-ties-even-input-and-weight\",\"gate\":\"sigma-cpu-gelu.affine-q0-q12.identity.v1\",\"input\":{\"bits\":50,\"domain\":\"A\",\"fractional\":12,\"signed\":true},\"internal_rounding\":\"signed-floor-only\",\"output\":{\"arithmetic\":\"same-as-input\",\"boolean_arity\":0},\"partition_order\":\"unsigned-canonical-half-open\",\"phi\":\"identity\",\"pieces\":[")?;
    for (i, &lower) in bounds[..bounds.len() - 1].iter().enumerate() {
        if i != 0 {
            w.text(",")?;
        }
        let x = if lower < n / 2 {
            lower as i64
        } else {
            lower as i64 - n as i64
        };
        let y = x.div_euclid(64);
        let index = y.unsigned_abs().min(255) as usize;
        let c = (n - u64::from(TABLE[index])) & (n - 1);
        w.text("{\"booleans\":[],\"coefficients\":[")?;
        w.number(c)?;
        w.text(",")?;
        w.number(u64::from(y >= 0))?;
        w.text("]}")?;
    }
    w.text("],\"profile\":\"fusefss.source-derived-ring50-q12.v1\",\"schema\":\"pllm.fusefss.scalar-descriptor.v1\",\"selector_semantics\":\"floor-x-div-64-before-abs-clip255\",\"source_lock_sha256\":\"")?;
    w.text(SOURCE_LOCK_SHA256)?;
    w.text("\",\"source_schedule\":{\"clipped_index\":[8,6],\"early_tr_output\":[32,6],\"effective_input\":[38,12],\"lut_and_output\":[50,12],\"range_status\":\"requires-sign-extension-provenance-not-a-model-proof\",\"required_signed_input_range\":[-137438953472,137438953472]}}")?;
    Descriptor::from_frozen_gelu_bytes(&w.bytes)
}
