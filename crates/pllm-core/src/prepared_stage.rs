//! A bounded one-use client stage owner. Python keeps authenticated transport and
//! compiler/checkpoint binding; this owner keeps masks, row scales and output slots.
//! The existing residue frame is checked in place, without a decoded body copy.
use crate::{mask_stream::MaskRows, offset_transport};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

const MAX_VALUES: usize = 4_000_000;
const MAX_FRAME: usize = 16 * 1024 * 1024 + 8192;

pub struct StageLayout {
    stage: String,
    input_width: usize,
    widths: Vec<u8>,
    width_digest: [u8; 32],
    wire_bits: u8,
    input_bits: u8,
    activation_bits: u8,
    maximum_rows: usize,
    weight_scales: Vec<u8>,
    bias: Option<Vec<u8>>,
}

impl StageLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage: String,
        input_width: usize,
        widths: Vec<u8>,
        wire_bits: u8,
        input_bits: u8,
        activation_bits: u8,
        maximum_rows: usize,
        weight_scales: Vec<u8>,
        bias: Option<Vec<u8>>,
    ) -> Result<Self, String> {
        if stage.is_empty()
            || stage.len() > 512
            || !(1..=4096).contains(&maximum_rows)
            || input_width == 0
            || input_width > MAX_VALUES / maximum_rows
            || widths.is_empty()
            || widths.len() > MAX_VALUES / maximum_rows
            || !matches!(wire_bits, 16 | 24 | 32)
            || !(2..=8).contains(&activation_bits)
            || !matches!(input_bits, 1..=32)
            || input_bits > wire_bits
            || widths.iter().any(|&w| w == 0 || w > input_bits)
            || weight_scales.len() != widths.len() * 4
            || bias.as_ref().is_some_and(|v| v.len() != widths.len() * 4)
            || weight_scales.chunks_exact(4).any(|v| {
                let x = f32::from_le_bytes(v.try_into().unwrap());
                !x.is_finite() || x <= 0.
            })
            || bias.as_ref().is_some_and(|v| {
                v.chunks_exact(4)
                    .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
            })
        {
            return Err("invalid bounded prepared client stage layout".into());
        }
        Ok(Self {
            stage,
            input_width,
            width_digest: Sha256::digest(&widths).into(),
            widths,
            wire_bits,
            input_bits,
            activation_bits,
            maximum_rows,
            weight_scales,
            bias,
        })
    }

    /// Claim only from the existing one-use stream. Failed construction burns the
    /// stream; input masks are retired before this call returns to Python.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        self: &Arc<Self>,
        masks: &mut MaskRows,
        start: usize,
        rows: usize,
        chunk_rows: usize,
        tickets: Vec<String>,
        input: &[u8],
    ) -> Result<(StageQuery, Vec<Vec<u8>>), String> {
        let result = self.prepare_inner(masks, start, rows, chunk_rows, tickets, input);
        if result.is_err() {
            masks.cancel();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        self: &Arc<Self>,
        masks: &mut MaskRows,
        start: usize,
        rows: usize,
        chunk_rows: usize,
        tickets: Vec<String>,
        input: &[u8],
    ) -> Result<(StageQuery, Vec<Vec<u8>>), String> {
        if rows == 0
            || rows > self.maximum_rows
            || chunk_rows == 0
            || chunk_rows > rows
            || rows.div_ceil(chunk_rows) > 128
            || tickets.len() != rows.div_ceil(chunk_rows)
            || input.len() != rows * self.input_width * 4
            || !masks.matches_layout(self.input_width, self.widths.len(), self.wire_bits)
            || tickets
                .iter()
                .any(|t| t.len() != 32 || !t.bytes().all(|c| c.is_ascii_hexdigit()))
            || tickets
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != tickets.len()
        {
            return Err("prepared client stage exceeds its row/frame contract".into());
        }
        let (r, s) = masks.take(start, rows)?;
        let (r, s) = (Zeroizing::new(r), Zeroizing::new(s));
        if r.len() != input.len() || s.len() != rows * self.widths.len() * 4 {
            return Err("prepared mask stream differs from stage geometry".into());
        }
        let mut scales = Zeroizing::new(Vec::with_capacity(rows * 4));
        let mut requests = Vec::with_capacity(tickets.len());
        for offset in (0..rows).step_by(chunk_rows) {
            let end = (offset + chunk_rows).min(rows);
            let range = offset * self.input_width * 4..end * self.input_width * 4;
            let (packed, row_scales) = offset_transport::quantize_mask_pack(
                &input[range.clone()],
                &r[range],
                end - offset,
                self.input_width,
                self.activation_bits,
                self.input_bits,
            )?;
            scales.extend_from_slice(&Zeroizing::new(row_scales));
            requests.push(packed);
        }
        Ok((
            StageQuery {
                layout: Arc::clone(self),
                rows,
                chunk_rows,
                tickets,
                state: Some(QueryState {
                    masks: s,
                    scales,
                    output: Zeroizing::new(vec![0; rows * self.widths.len() * 4]),
                    next: 0,
                    server_ns: 0,
                }),
            },
            requests,
        ))
    }
}

struct QueryState {
    masks: Zeroizing<Vec<u8>>,
    scales: Zeroizing<Vec<u8>>,
    output: Zeroizing<Vec<u8>>,
    next: usize,
    server_ns: u64,
}

/// Not cloneable or serializable. Every error retires all remaining private state.
pub struct StageQuery {
    layout: Arc<StageLayout>,
    rows: usize,
    chunk_rows: usize,
    tickets: Vec<String>,
    state: Option<QueryState>,
}

impl StageQuery {
    pub fn receive(&mut self, payload: &[u8]) -> Result<(), String> {
        let mut state = self
            .state
            .take()
            .ok_or("prepared client stage is retired")?;
        if payload.len() > MAX_FRAME || !payload.starts_with(b"PLLMPR01") {
            return Err("prepared client response namespace or size differs".into());
        }
        let ticket = self
            .tickets
            .get(state.next)
            .ok_or("prepared client stage received excess frames")?;
        let start = state.next * self.chunk_rows;
        let end = (start + self.chunk_rows).min(self.rows);
        let mut frame = Frame(&payload[8..]);
        if frame.byte()? != 0x97
            || frame.string()? != ticket
            || frame.string()? != self.layout.stage
            || frame.unsigned()? != (end - start) as u64
            || frame.unsigned()? != u64::from(self.layout.wire_bits)
            || frame.binary()? != self.layout.width_digest
        {
            return Err("prepared client response differs from its ordered stage contract".into());
        }
        let server_ns = frame.unsigned()?;
        if server_ns >= 1 << 63 {
            return Err("invalid prepared server duration".into());
        }
        let packed = frame.binary()?;
        if !frame.0.is_empty() {
            return Err("trailing prepared response bytes".into());
        }
        let range = start * self.layout.widths.len() * 4..end * self.layout.widths.len() * 4;
        offset_transport::unmask_dequantize_into(
            packed,
            &state.masks[range.clone()],
            &self.layout.widths,
            end - start,
            &state.scales[start * 4..end * 4],
            &self.layout.weight_scales,
            self.layout.bias.as_deref(),
            &mut state.output[range.clone()],
        )?;
        state.masks[range].zeroize();
        state.scales[start * 4..end * 4].zeroize();
        state.server_ns = state
            .server_ns
            .checked_add(server_ns)
            .ok_or("prepared duration overflow")?;
        state.next += 1;
        self.state = Some(state);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(Zeroizing<Vec<u8>>, u64), String> {
        let state = self
            .state
            .take()
            .ok_or("prepared client stage is retired")?;
        if state.next != self.tickets.len() {
            return Err("prepared client stage is incomplete".into());
        }
        Ok((state.output, state.server_ns))
    }

    pub fn cancel(&mut self) {
        self.state = None;
    }

    pub fn retained_bytes(&self) -> usize {
        self.state.as_ref().map_or(0, |s| {
            s.masks.capacity() + s.scales.capacity() + s.output.capacity()
        })
    }
}

/// Bounded borrowed MessagePack fields. No allocation, recursion, signed/bool
/// aliases, extensions, coercions or trailing objects are accepted.
struct Frame<'a>(&'a [u8]);
impl<'a> Frame<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        if length > self.0.len() {
            return Err("truncated prepared response".into());
        }
        let (head, tail) = self.0.split_at(length);
        self.0 = tail;
        Ok(head)
    }
    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn number(&mut self, bytes: usize) -> Result<u64, String> {
        Ok(self
            .take(bytes)?
            .iter()
            .fold(0, |n, &v| (n << 8) | u64::from(v)))
    }
    fn unsigned(&mut self) -> Result<u64, String> {
        match self.byte()? {
            n @ 0..=127 => Ok(u64::from(n)),
            0xcc => self.number(1),
            0xcd => self.number(2),
            0xce => self.number(4),
            0xcf => self.number(8),
            _ => Err("invalid prepared integer field".into()),
        }
    }
    fn string(&mut self) -> Result<&'a str, String> {
        let size = match self.byte()? {
            n @ 0xa0..=0xbf => u64::from(n - 0xa0),
            0xd9 => self.number(1)?,
            0xda => self.number(2)?,
            0xdb => self.number(4)?,
            _ => return Err("invalid prepared string field".into()),
        };
        if size > 512 {
            return Err("overlarge prepared string field".into());
        }
        std::str::from_utf8(self.take(size as usize)?).map_err(|_| "invalid prepared UTF-8".into())
    }
    fn binary(&mut self) -> Result<&'a [u8], String> {
        let size = match self.byte()? {
            0xc4 => self.number(1)?,
            0xc5 => self.number(2)?,
            0xc6 => self.number(4)?,
            _ => return Err("invalid prepared binary field".into()),
        };
        if size > MAX_FRAME as u64 {
            return Err("overlarge prepared binary field".into());
        }
        self.take(size as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Arc<StageLayout> {
        Arc::new(
            StageLayout::new(
                "stage".into(),
                2,
                vec![8],
                16,
                8,
                8,
                1,
                1_f32.to_le_bytes().to_vec(),
                None,
            )
            .unwrap(),
        )
    }

    fn masks() -> MaskRows {
        MaskRows::new(&[7; 32], b"input", b"output", 1, 2, 1, 16).unwrap()
    }

    fn input() -> Vec<u8> {
        [1_f32, 2.].into_iter().flat_map(f32::to_le_bytes).collect()
    }

    // Independent fixed MessagePack fixture: signed output 3, one row, one column.
    fn reply() -> Vec<u8> {
        let (_, s) = masks().take(0, 1).unwrap();
        let residue = 3_u8.wrapping_sub(s[0]);
        let mut frame = b"PLLMPR01".to_vec();
        frame.extend([0x97, 0xd9, 32]);
        frame.extend([b'a'; 32]);
        frame.extend(b"\xa5stage\x01\x10\xc4\x20");
        frame.extend(Sha256::digest([8]));
        frame.extend([1, 0xc4, 1, residue]);
        frame
    }

    #[test]
    fn owned_output_matches_float_order_and_consumes_masks() {
        let mut stream = masks();
        let (mut query, _) = layout()
            .prepare(&mut stream, 0, 1, 1, vec!["a".repeat(32)], &input())
            .unwrap();
        query.receive(&reply()).unwrap();
        let (output, duration) = query.finish().unwrap();
        assert_eq!(&*output, &(3_f32 * (2_f32 / 127_f32)).to_le_bytes());
        assert_eq!(duration, 1);
        assert_eq!(query.retained_bytes(), 0);
        assert!(query.finish().is_err());
        assert!(stream.take(0, 1).is_err());
    }

    #[test]
    fn every_truncated_response_and_failed_input_burns() {
        let response = reply();
        for length in 0..response.len() {
            let (mut query, _) = layout()
                .prepare(&mut masks(), 0, 1, 1, vec!["a".repeat(32)], &input())
                .unwrap();
            assert!(query.receive(&response[..length]).is_err());
            assert_eq!(query.retained_bytes(), 0);
            assert!(query.receive(&response).is_err());
        }
        let mut stream = masks();
        let bad = [f32::NAN, 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(layout()
            .prepare(&mut stream, 0, 1, 1, vec!["a".repeat(32)], &bad)
            .is_err());
        assert!(stream.take(0, 1).is_err());
    }
}
