//! Bounded one-use masked-wire SIGMA-style SiLU × up reference.
//!
//! SIGMA Figure 2, §4.2.2, Figure 3/7 and Appendix F. BGI/Grotto
//! Boolean DPFs replace generic arithmetic DCFs. Masks bind adjacent gates;
//! evaluators reconstruct only freshly masked intermediate values. The local
//! dealer/client is research-only; no malicious security or role transport.
//! Lookup uses the Pika Boolean selection/sign-correction algebra, with its
//! sign shares derived from each local full-domain vector. The optional fused
//! layout applies FuseFSS §4.3 / Appendix E's public-view rewrite to the input
//! gate shift, residual-index shift and predicates. All later post-processing
//! remains charged. No GPU implementation or complete transformer reproduction.

use crate::compact_dpf::{Bits, Key};
use pllm_core::residual_silu_reference::{
    Profile, EFFECTIVE_GATE_BITS, GATE_BOUND, RING_BITS, UP_BOUND,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

pub const MAX_LANES: usize = 256;
pub const MAX_ALLOCATION: usize = 64 << 20;
pub const HEADER: usize = 78;
pub const ROUNDS: usize = 11;
const MAGIC: &[u8; 8] = b"PLLMMW01";
const N: u8 = RING_BITS;
const M: u8 = EFFECTIVE_GATE_BITS;
const Y: u8 = M - 6;
const INPUT_GATE_BITS: u8 = M + 12;
const WIDTHS: [&[u8]; ROUNDS] = [
    &[1, 1],
    &[N, N],
    &[1],
    &[Y],
    &[1, 1],
    &[10],
    &[N],
    &[N],
    &[N],
    &[1],
    &[N],
];
const FUSED_WIDTHS: [&[u8]; 8] = [
    &[1, 1, 1, 1, 1],
    &[N, N, Y],
    &[10],
    &[N],
    &[N],
    &[N],
    &[1],
    &[N],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Unfused,
    Fused,
}

impl Layout {
    pub fn rounds(self) -> usize {
        self.widths().len()
    }
    fn widths(self) -> &'static [&'static [u8]] {
        match self {
            Self::Unfused => &WIDTHS,
            Self::Fused => &FUSED_WIDTHS,
        }
    }
    fn sign_bits(self) -> u8 {
        match self {
            Self::Unfused => Y,
            Self::Fused => INPUT_GATE_BITS,
        }
    }
    fn reduce_bits(self) -> (u8, u8) {
        match self {
            Self::Unfused => (M, 6),
            Self::Fused => (INPUT_GATE_BITS, 18),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Domain,
    Resource,
    Randomness,
    Consumed,
    Context,
}

fn mask(n: u8) -> u64 {
    u64::MAX >> (64 - n)
}
fn random(n: u8) -> Result<u64, Error> {
    let mut b = Zeroizing::new([0; 8]);
    getrandom::fill(&mut *b).map_err(|_| Error::Randomness)?;
    Ok(u64::from_le_bytes(*b) & mask(n))
}
fn share(x: u64, n: u8) -> Result<[u64; 2], Error> {
    let a = random(n)?;
    Ok([a, x.wrapping_sub(a) & mask(n)])
}
fn signed(x: u64, n: u8) -> i64 {
    ((x << (64 - n)) as i64) >> (64 - n)
}

struct Shift {
    dpf: Key,
    n: u8,
    shift: u8,
    gap: bool,
    constants: Zeroizing<[u64; 3]>,
}
impl Shift {
    fn issue(n: u8, shift: u8, gap: bool, rin: u64, rout: u64) -> Result<[Self; 2], Error> {
        assert!(shift > 0 && shift < n - 1);
        let keys = Key::issue(shift, rin & mask(shift)).map_err(|_| Error::Randomness)?;
        let out_bits = if gap { n } else { n - shift };
        let c = share(random(1)?, out_bits)?;
        let wrap = if gap {
            share((rin >> (n - 1)) << (n - shift), n)?
        } else {
            [0; 2]
        };
        // GapARS uses the public offset x=v+2^(n-2). The -2^(n-f-2)
        // offset is folded into output-mask shares, not discarded.
        let r = share(
            rout.wrapping_sub(rin >> shift).wrapping_sub(if gap {
                1 << (n - shift - 2)
            } else {
                0
            }),
            out_bits,
        )?;
        let [k0, k1] = keys;
        Ok([k0, k1]
            .into_iter()
            .enumerate()
            .map(|(b, dpf)| Self {
                dpf,
                n,
                shift,
                gap,
                constants: Zeroizing::new([c[b], wrap[b], r[b]]),
            })
            .collect::<Vec<_>>()
            .try_into()
            .ok()
            .unwrap())
    }
    fn adjusted(&self, x: u64) -> u64 {
        x.wrapping_add(if self.gap { 1 << (self.n - 2) } else { 0 }) & mask(self.n)
    }
    fn start(&self, b: u8, x: u64) -> u64 {
        u64::from(self.dpf.less(b, self.adjusted(x) & mask(self.shift))) ^ (self.constants[0] & 1)
    }
    fn finish(&self, b: u8, x: u64, opened_bit: u64) -> u64 {
        let x = self.adjusted(x);
        let carry = u64::from(b)
            .wrapping_mul(opened_bit)
            .wrapping_add(self.constants[0])
            .wrapping_sub(
                2u64.wrapping_mul(opened_bit)
                    .wrapping_mul(self.constants[0]),
            );
        let wrap = self.constants[1].wrapping_mul(1 - (x >> (self.n - 1)));
        (u64::from(b)
            .wrapping_mul(x >> self.shift)
            .wrapping_add(self.constants[2])
            .wrapping_add(wrap)
            .wrapping_sub(carry))
            & mask(if self.gap {
                self.n
            } else {
                self.n - self.shift
            })
    }
    fn bits(n: u8, shift: u8, gap: bool) -> usize {
        Key::encoded_bits(shift)
            + if gap {
                3 * n as usize
            } else {
                2 * (n - shift) as usize
            }
    }
    fn encode(&self, out: &mut Bits) {
        self.dpf.encode(out);
        let w = if self.gap {
            self.n
        } else {
            self.n - self.shift
        };
        out.put(self.constants[0].into(), w as usize);
        if self.gap {
            out.put(self.constants[1].into(), w as usize);
        }
        out.put(self.constants[2].into(), w as usize);
    }
}

struct Signs {
    key: Key,
    bits: u8,
    scale: u8,
    constants: Zeroizing<[u64; 2]>,
}
impl Signs {
    fn issue(rin: u64, dmask: u64, imask: u64) -> Result<[Self; 2], Error> {
        Self::issue_scaled(Y, 0, rin, dmask, imask)
    }
    fn issue_scaled(
        bits: u8,
        scale: u8,
        rin: u64,
        dmask: u64,
        imask: u64,
    ) -> Result<[Self; 2], Error> {
        let neg = 0u64.wrapping_sub(rin) & mask(bits);
        let keys = Key::issue(bits - 1, neg & mask(bits - 1)).map_err(|_| Error::Randomness)?;
        let d = share(dmask ^ (neg >> (bits - 1)) ^ 1, 1)?;
        let i = share(imask, 1)?;
        let [k0, k1] = keys;
        Ok([
            Self {
                key: k0,
                bits,
                scale,
                constants: Zeroizing::new([d[0], i[0]]),
            },
            Self {
                key: k1,
                bits,
                scale,
                constants: Zeroizing::new([d[1], i[1]]),
            },
        ])
    }
    fn raw(&self, b: u8, x: u64) -> u64 {
        let x = x & mask(self.bits);
        (u64::from(b) * (x >> (self.bits - 1)))
            ^ u64::from(
                self.key
                    .less(b, mask(self.bits - 1) - (x & mask(self.bits - 1))),
            )
    }
    fn eval(&self, b: u8, x: u64) -> [u64; 2] {
        // Only the combined membership share is exposed. Distinct output masks
        // prevent the shared DPF's two affine comparison constants cancelling
        // into an unmasked, reconstructed interval-membership bit.
        [
            self.raw(b, x) ^ self.constants[0],
            self.raw(b, x.wrapping_add(1023 << self.scale))
                ^ self.raw(b, x.wrapping_sub(1024 << self.scale))
                ^ self.constants[1],
        ]
    }
    fn encode(&self, out: &mut Bits) {
        self.key.encode(out);
        for x in *self.constants {
            out.put(x.into(), 1);
        }
    }
}

struct Select {
    bits: u8,
    coefficients: Zeroizing<Vec<[u64; 2]>>,
}
impl Select {
    fn issue(
        bits: u8,
        selector_mask: usize,
        rin: u64,
        rout: u64,
        affine: &[[i64; 2]],
    ) -> Result<[Self; 2], Error> {
        let mut rows = [Zeroizing::new(Vec::new()), Zeroizing::new(Vec::new())];
        for s in 0..affine.len() {
            let [a, c] = affine[s ^ selector_mask];
            let left = share(a as u64, bits)?;
            let right = share(
                (c as u64)
                    .wrapping_add(rout)
                    .wrapping_sub((a as u64).wrapping_mul(rin)),
                bits,
            )?;
            for b in 0..2 {
                rows[b].push([left[b], right[b]]);
            }
        }
        Ok(rows.map(|coefficients| Self { bits, coefficients }))
    }
    fn eval(&self, selector: usize, x: u64) -> u64 {
        let [a, b] = self.coefficients[selector];
        a.wrapping_mul(x).wrapping_add(b) & mask(self.bits)
    }
    fn encode(&self, out: &mut Bits) {
        for row in self.coefficients.iter() {
            for x in row {
                out.put((*x).into(), self.bits as usize);
            }
        }
    }
}

struct Lookup {
    dpf: Key,
    constants: Zeroizing<[u64; 2]>,
}
impl Lookup {
    fn issue(rin: u64, rout: u64) -> Result<[Self; 2], Error> {
        let point = 0u64.wrapping_sub(rin) & 1023;
        let keys = Key::issue(10, point).map_err(|_| Error::Randomness)?;
        let sign = if keys[0].point(0, point) == 1 {
            1u64
        } else {
            u64::MAX
        };
        let r = random(N)?;
        let rs = share(r, N)?;
        let correction = share(rout.wrapping_sub(sign.wrapping_mul(r)), N)?;
        let [left, right] = keys;
        Ok([
            Self {
                dpf: left,
                constants: Zeroizing::new([rs[0], correction[0]]),
            },
            Self {
                dpf: right,
                constants: Zeroizing::new([rs[1], correction[1]]),
            },
        ])
    }
    fn start(&self, b: u8, x: u64, table: &[i64; 1024]) -> (u64, u64) {
        let values = self.dpf.all(b);
        let mut inner = 0u64;
        let mut sign_share = 0u64;
        for (i, &bit) in values.iter().enumerate() {
            let value = u64::from(bit);
            inner = inner.wrapping_add(value.wrapping_mul(table[(i + x as usize) & 1023] as u64));
            sign_share = sign_share.wrapping_add(value);
        }
        if b == 1 {
            inner = inner.wrapping_neg();
            sign_share = sign_share.wrapping_neg();
        }
        (
            inner.wrapping_add(self.constants[0]) & mask(N),
            sign_share & mask(N),
        )
    }
    fn finish(&self, opened: u64, sign_share: u64) -> u64 {
        opened
            .wrapping_mul(sign_share)
            .wrapping_add(self.constants[1])
            & mask(N)
    }
    fn encode(&self, out: &mut Bits) {
        self.dpf.encode(out);
        for x in *self.constants {
            out.put(x.into(), N as usize);
        }
    }
}

struct Multiply {
    coefficients: Zeroizing<[u64; 3]>,
}
impl Multiply {
    fn issue(rx: u64, ry: u64, rout: u64) -> Result<[Self; 2], Error> {
        let a = share(rx.wrapping_neg(), N)?;
        let b = share(ry.wrapping_neg(), N)?;
        let c = share(rx.wrapping_mul(ry).wrapping_add(rout), N)?;
        Ok([0, 1].map(|i| Self {
            coefficients: Zeroizing::new([a[i], b[i], c[i]]),
        }))
    }
    fn eval(&self, party: u8, x: u64, y: u64) -> u64 {
        u64::from(party)
            .wrapping_mul(x)
            .wrapping_mul(y)
            .wrapping_add(self.coefficients[0].wrapping_mul(y))
            .wrapping_add(self.coefficients[1].wrapping_mul(x))
            .wrapping_add(self.coefficients[2])
            & mask(N)
    }
    fn encode(&self, out: &mut Bits) {
        for x in *self.coefficients {
            out.put(x.into(), N as usize);
        }
    }
}

struct Lane {
    gate: Shift,
    up: Shift,
    reduce: Shift,
    signs: Signs,
    clip: Select,
    relu: Select,
    lookup: Lookup,
    product: Multiply,
    output: Shift,
}
impl Lane {
    fn encode(&self, out: &mut Bits) {
        self.gate.encode(out);
        self.up.encode(out);
        self.reduce.encode(out);
        self.signs.encode(out);
        self.clip.encode(out);
        self.relu.encode(out);
        self.lookup.encode(out);
        self.product.encode(out);
        self.output.encode(out);
    }
}

#[derive(Clone, Copy)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub first_tensor_index: u64,
}

#[derive(Debug)]
pub struct Resources {
    pub lanes: usize,
    pub party_key_bits: usize,
    pub party_key_bytes: usize,
    pub combined_allocation_estimate_bytes: usize,
    pub peer_frame_bytes: usize,
    pub peer_payload_bits: usize,
    pub public_table_bytes: usize,
}

pub fn resources(lanes: usize) -> Result<Resources, Error> {
    resources_with_layout(lanes, Layout::Unfused)
}

pub fn resources_with_layout(lanes: usize, layout: Layout) -> Result<Resources, Error> {
    if !(1..=MAX_LANES).contains(&lanes) {
        return Err(Error::Resource);
    }
    let (reduce_bits, reduce_shift) = layout.reduce_bits();
    let per_lane = 3 * Shift::bits(N, 12, true)
        + Shift::bits(reduce_bits, reduce_shift, false)
        + Key::encoded_bits(layout.sign_bits() - 1)
        + 2
        + 8 * 10
        + 4 * N as usize
        + Key::encoded_bits(10)
        + 2 * N as usize
        + 3 * N as usize;
    let key_bits = per_lane * lanes;
    let key_bytes = key_bits.div_ceil(8);
    let allocation = 2 * (64 * key_bytes + 4096 * lanes) + (1 << 16);
    if allocation > MAX_ALLOCATION {
        return Err(Error::Resource);
    }
    let payload_bits = 2
        * lanes
        * layout
            .widths()
            .iter()
            .flat_map(|v| v.iter())
            .map(|&w| w as usize)
            .sum::<usize>();
    let peer_bytes = 2 * layout
        .widths()
        .iter()
        .map(|widths| {
            HEADER + (lanes * widths.iter().map(|&w| w as usize).sum::<usize>()).div_ceil(8)
        })
        .sum::<usize>();
    Ok(Resources {
        lanes,
        party_key_bits: key_bits,
        party_key_bytes: key_bytes,
        combined_allocation_estimate_bytes: allocation,
        peer_frame_bytes: peer_bytes,
        peer_payload_bits: payload_bits,
        public_table_bytes: 1024 * 8,
    })
}

pub struct Client {
    binding: [u8; 32],
    nonce: [u8; 32],
    inputs: Option<Zeroizing<Vec<[u64; 2]>>>,
    outputs: Option<Zeroizing<Vec<u64>>>,
}
pub struct MaskedInputs {
    binding: [u8; 32],
    nonce: [u8; 32],
    values: Zeroizing<Vec<[u64; 2]>>,
}
pub struct MaskedOutput {
    binding: [u8; 32],
    nonce: [u8; 32],
    values: Zeroizing<Vec<u64>>,
}

impl Client {
    pub fn encode(&mut self, gate_q24: &[i64], up_q24: &[i64]) -> Result<MaskedInputs, Error> {
        let masks = self.inputs.take().ok_or(Error::Consumed)?;
        if gate_q24.len() != masks.len()
            || up_q24.len() != masks.len()
            || gate_q24
                .iter()
                .any(|x| x.unsigned_abs() > (GATE_BOUND as u64) << 24)
            || up_q24
                .iter()
                .any(|x| x.unsigned_abs() > (UP_BOUND as u64) << 24)
        {
            self.outputs = None;
            return Err(Error::Domain);
        }
        let values = Zeroizing::new(
            masks
                .iter()
                .zip(gate_q24.iter().zip(up_q24))
                .map(|(m, (&g, &u))| {
                    [
                        (g as u64).wrapping_add(m[0]) & mask(N),
                        (u as u64).wrapping_add(m[1]) & mask(N),
                    ]
                })
                .collect(),
        );
        Ok(MaskedInputs {
            binding: self.binding,
            nonce: self.nonce,
            values,
        })
    }
    pub fn decode(&mut self, output: &MaskedOutput) -> Result<Vec<i64>, Error> {
        let masks = self.outputs.take().ok_or(Error::Consumed)?;
        self.inputs = None;
        if output.binding != self.binding
            || output.nonce != self.nonce
            || output.values.len() != masks.len()
        {
            return Err(Error::Context);
        }
        Ok(output
            .values
            .iter()
            .zip(masks.iter())
            .map(|(&x, &r)| signed(x.wrapping_sub(r) & mask(N), N))
            .collect())
    }
}

pub fn issue_reference(context: Context, lanes: usize) -> Result<(Client, [Party; 2]), Error> {
    issue_reference_with_layout(context, lanes, Layout::Unfused)
}

pub fn issue_reference_with_layout(
    context: Context,
    lanes: usize,
    layout: Layout,
) -> Result<(Client, [Party; 2]), Error> {
    let resource = resources_with_layout(lanes, layout)?;
    context
        .first_tensor_index
        .checked_add(lanes as u64)
        .ok_or(Error::Domain)?;
    let profile = Profile::new();
    let mut h = Sha256::new();
    h.update(b"pllm.masked_wire_reference.v1\0");
    h.update(profile.digest());
    h.update(context.plan);
    h.update(context.session);
    h.update(context.operation);
    h.update(context.first_tensor_index.to_le_bytes());
    h.update((lanes as u64).to_le_bytes());
    if layout == Layout::Fused {
        h.update(b"masked-input-fusion.v1\0");
    }
    let binding = h.finalize().into();
    let mut nonce = [0; 32];
    getrandom::fill(&mut nonce).map_err(|_| Error::Randomness)?;
    let mut inputs = Zeroizing::new(Vec::with_capacity(lanes));
    let mut outputs = Zeroizing::new(Vec::with_capacity(lanes));
    let mut parties = [Vec::with_capacity(lanes), Vec::with_capacity(lanes)];
    for _ in 0..lanes {
        let r = Zeroizing::new([
            random(N)?,
            random(N)?,
            random(N)?,
            random(N)?,
            random(Y)?,
            random(1)?,
            random(1)?,
            random(10)?,
            random(N)?,
            random(N)?,
            random(N)?,
            random(N)?,
        ]);
        inputs.push([r[0], r[1]]);
        outputs.push(r[11]);
        let gate = Shift::issue(N, 12, true, r[0], r[2])?;
        let up = Shift::issue(N, 12, true, r[1], r[3])?;
        // floor(floor(g / 2^12) / 2^6) == floor(g / 2^18).
        // The same public gate bound proves a signed 31-bit input view and
        // 13-bit output. Predicate thresholds are lifted by 18 bits, so the
        // sign and residual interval use the original masked wire as well.
        let (reduce_bits, reduce_shift) = layout.reduce_bits();
        let reduce_input_mask = if layout == Layout::Fused { r[0] } else { r[2] };
        let reduce = Shift::issue(
            reduce_bits,
            reduce_shift,
            false,
            reduce_input_mask & mask(reduce_bits),
            r[4],
        )?;
        let signs = if layout == Layout::Fused {
            Signs::issue_scaled(
                INPUT_GATE_BITS,
                18,
                r[0] & mask(INPUT_GATE_BITS),
                r[5],
                r[6],
            )?
        } else {
            Signs::issue(r[4], r[5], r[6])?
        };
        let clip = Select::issue(
            10,
            ((r[6] << 1) | r[5]) as usize,
            r[4] & 1023,
            r[7],
            &[[0, 1023], [0, 1023], [-1, 0], [1, 0]],
        )?;
        let relu = Select::issue(N, r[5] as usize, r[2], r[8], &[[0, 0], [1, 0]])?;
        let lookup = Lookup::issue(r[7], r[9])?;
        let product = Multiply::issue(r[8].wrapping_sub(r[9]) & mask(N), r[3], r[10])?;
        let output = Shift::issue(N, 12, true, r[10], r[11])?;
        for (b, ((((((((gate, up), reduce), signs), clip), relu), lookup), product), output)) in
            gate.into_iter()
                .zip(up)
                .zip(reduce)
                .zip(signs)
                .zip(clip)
                .zip(relu)
                .zip(lookup)
                .zip(product)
                .zip(output)
                .enumerate()
        {
            parties[b].push(Lane {
                gate,
                up,
                reduce,
                signs,
                clip,
                relu,
                lookup,
                product,
                output,
            });
        }
    }
    // Count real packed secret payloads, then erase the temporary serialization.
    for party in &parties {
        let mut encoded = Bits::new();
        for lane in party {
            lane.encode(&mut encoded);
        }
        if encoded.len != resource.party_key_bits || encoded.data.len() != resource.party_key_bytes
        {
            return Err(Error::Resource);
        }
    }
    let client = Client {
        binding,
        nonce,
        inputs: Some(inputs),
        outputs: Some(outputs),
    };
    let mut index = 0;
    Ok((
        client,
        parties.map(|keys| {
            let id = index;
            index += 1;
            Party {
                id,
                layout,
                binding,
                nonce,
                lanes,
                keys,
                state: Zeroizing::new(vec![[0; 8]; lanes]),
                last: Zeroizing::new(Vec::new()),
                phase: None,
                alive: true,
                table: *profile.table(),
            }
        }),
    ))
}

pub struct Party {
    id: u8,
    layout: Layout,
    binding: [u8; 32],
    nonce: [u8; 32],
    lanes: usize,
    keys: Vec<Lane>,
    state: Zeroizing<Vec<[u64; 8]>>,
    last: Zeroizing<Vec<u64>>,
    phase: Option<usize>,
    alive: bool,
    table: [i64; 1024],
}
pub enum Step {
    Send(Vec<u8>),
    Done(MaskedOutput),
}

impl Party {
    pub fn cancel(&mut self) {
        self.alive = false;
        self.keys.clear();
        self.state.zeroize();
        self.last.zeroize();
    }
    pub fn start(&mut self, input: &MaskedInputs) -> Result<Vec<u8>, Error> {
        if !self.alive || self.phase.is_some() {
            self.cancel();
            return Err(Error::Consumed);
        }
        if input.binding != self.binding
            || input.nonce != self.nonce
            || input.values.len() != self.lanes
        {
            self.cancel();
            return Err(Error::Context);
        }
        let mut output = Vec::with_capacity(self.lanes * 2);
        for (i, (&values, key)) in input.values.iter().zip(&self.keys).enumerate() {
            self.state[i][0] = values[0];
            self.state[i][1] = values[1];
            output.extend([
                key.gate.start(self.id, values[0]),
                key.up.start(self.id, values[1]),
            ]);
            if self.layout == Layout::Fused {
                output.push(key.reduce.start(self.id, values[0] & mask(INPUT_GATE_BITS)));
                output.extend(key.signs.eval(self.id, values[0] & mask(INPUT_GATE_BITS)));
            }
        }
        Ok(self.frame(0, output))
    }
    fn frame(&mut self, phase: usize, output: Vec<u64>) -> Vec<u8> {
        let mut bits = Bits::new();
        for (i, &x) in output.iter().enumerate() {
            let widths = self.layout.widths()[phase];
            bits.put(x.into(), widths[i % widths.len()] as usize);
        }
        let mut frame = Vec::with_capacity(HEADER + bits.data.len());
        frame.extend(MAGIC);
        frame.extend(self.binding);
        frame.extend(self.nonce);
        frame.extend([self.id, phase as u8]);
        frame.extend((self.lanes as u32).to_le_bytes());
        frame.extend_from_slice(&bits.data);
        self.phase = Some(phase);
        self.last = Zeroizing::new(output);
        frame
    }
    fn open(&self, frame: &[u8], phase: usize) -> Result<Zeroizing<Vec<u64>>, Error> {
        let widths = self.layout.widths()[phase];
        let bit_count = self.lanes * widths.iter().map(|&w| w as usize).sum::<usize>();
        if frame.len() != HEADER + bit_count.div_ceil(8)
            || &frame[..8] != MAGIC
            || frame[8..40] != self.binding
            || frame[40..72] != self.nonce
            || frame[72] != 1 - self.id
            || frame[73] != phase as u8
            || frame[74..78] != (self.lanes as u32).to_le_bytes()
        {
            return Err(Error::Context);
        }
        if bit_count % 8 != 0 && frame[frame.len() - 1] >> (bit_count % 8) != 0 {
            return Err(Error::Context);
        }
        let mut values = Zeroizing::new(Vec::with_capacity(self.last.len()));
        let mut offset = 0;
        for (i, &own) in self.last.iter().enumerate() {
            let width = widths[i % widths.len()];
            let mut peer = 0u64;
            for b in 0..width {
                peer |= u64::from((frame[HEADER + offset / 8] >> (offset % 8)) & 1) << b;
                offset += 1;
            }
            values.push(own.wrapping_add(peer) & mask(width));
        }
        Ok(values)
    }
    pub fn advance(&mut self, peer: &[u8]) -> Result<Step, Error> {
        if !self.alive {
            return Err(Error::Consumed);
        }
        let result = self.advance_inner(peer);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn advance_inner(&mut self, peer: &[u8]) -> Result<Step, Error> {
        let phase = self.phase.ok_or(Error::Context)?;
        let opened = self.open(peer, phase)?;
        if phase + 1 == self.layout.rounds() {
            let output = MaskedOutput {
                binding: self.binding,
                nonce: self.nonce,
                values: opened,
            };
            self.cancel();
            return Ok(Step::Done(output));
        }
        let mut next = Vec::with_capacity(self.lanes * 2);
        for (lane, k) in self.keys.iter().enumerate() {
            let s = &mut self.state[lane];
            let x = &opened[lane * self.layout.widths()[phase].len()..];
            if self.layout == Layout::Fused && phase == 0 {
                next.extend([
                    k.gate.finish(self.id, s[0], x[0]),
                    k.up.finish(self.id, s[1], x[1]),
                    k.reduce.finish(self.id, s[0] & mask(INPUT_GATE_BITS), x[2]),
                ]);
                s[5] = x[3];
                s[6] = x[4];
                continue;
            }
            if self.layout == Layout::Fused && phase == 1 {
                s[2] = x[0];
                s[3] = x[1];
                s[4] = x[2];
                s[7] = k.relu.eval(s[5] as usize, s[2]);
                next.push(k.clip.eval(((s[6] << 1) | s[5]) as usize, s[4] & 1023));
                continue;
            }
            match if self.layout == Layout::Fused {
                phase + 3
            } else {
                phase
            } {
                0 => next.extend([
                    k.gate.finish(self.id, s[0], x[0]),
                    k.up.finish(self.id, s[1], x[1]),
                ]),
                1 => {
                    s[2] = x[0];
                    s[3] = x[1];
                    next.push(k.reduce.start(self.id, s[2] & mask(M)));
                }
                2 => next.push(k.reduce.finish(self.id, s[2] & mask(M), x[0])),
                3 => {
                    s[4] = x[0];
                    next.extend(k.signs.eval(self.id, x[0]));
                }
                4 => {
                    s[5] = x[0];
                    s[6] = x[1];
                    s[7] = k.relu.eval(x[0] as usize, s[2]);
                    next.push(k.clip.eval(((x[1] << 1) | x[0]) as usize, s[4] & 1023));
                }
                5 => {
                    let (value, sign) = k.lookup.start(self.id, x[0], &self.table);
                    s[0] = sign;
                    next.push(value);
                }
                6 => next.push(s[7].wrapping_sub(k.lookup.finish(x[0], s[0])) & mask(N)),
                7 => next.push(k.product.eval(self.id, x[0], s[3])),
                8 => {
                    s[0] = x[0];
                    next.push(k.output.start(self.id, x[0]));
                }
                9 => next.push(k.output.finish(self.id, s[0], x[0])),
                _ => return Err(Error::Context),
            }
        }
        Ok(Step::Send(self.frame(phase + 1, next)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> Context {
        Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [3; 32],
            first_tensor_index: 0,
        }
    }

    #[test]
    fn gap_and_reduce_exhaust_masks_inputs_and_boundaries() {
        for gap in [false, true] {
            for n in 3..=7 {
                for f in 1..n - 1 {
                    for rin in 0..1 << n {
                        let rout = (rin * 7 + 13) & mask(if gap { n } else { n - f });
                        let keys = Shift::issue(n, f, gap, rin, rout).unwrap();
                        for x in 0..1 << n {
                            let v = signed(x, n);
                            if gap && !(-(1i64 << (n - 2))..1 << (n - 2)).contains(&v) {
                                continue;
                            }
                            let masked = (x + rin) & mask(n);
                            let b = keys[0].start(0, masked) ^ keys[1].start(1, masked);
                            let out = keys[0]
                                .finish(0, masked, b)
                                .wrapping_add(keys[1].finish(1, masked, b))
                                .wrapping_sub(rout)
                                & mask(if gap { n } else { n - f });
                            assert_eq!(
                                out,
                                (if gap { (v >> f) as u64 } else { x >> f })
                                    & mask(if gap { n } else { n - f })
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(Shift::bits(32, 7, true).div_ceil(8), 28);
        // Outside-gap inputs have no correctness contract: demonstrate why the
        // admission proof is required rather than silently applying this helper.
        let k = Shift::issue(8, 2, true, 120, 0).unwrap();
        let x = (100 + 120) & 255;
        let b = k[0].start(0, x) ^ k[1].start(1, x);
        assert_ne!((k[0].finish(0, x, b) + k[1].finish(1, x, b)) & 255, 25);
    }

    #[test]
    fn complete_block_exact_and_one_use() {
        let p = Profile::new();
        let mut g = vec![];
        let mut u = vec![];
        for gate in [
            -48.0f64, -16.01, -16.0, -1.0, -0.0001, 0.0, 0.0001, 1.0, 15.99, 16.0, 48.0,
        ] {
            for up in [-128.0f64, -1.0, 0.0, 1.0, 128.0] {
                g.push((gate * 16777216.0).floor() as i64);
                u.push((up * 16777216.0).floor() as i64);
            }
        }
        let (mut c, [mut a, mut b]) = issue_reference(context(), g.len()).unwrap();
        let input = c.encode(&g, &u).unwrap();
        let mut left = a.start(&input).unwrap();
        let mut right = b.start(&input).unwrap();
        let mut bytes = left.len() + right.len();
        for round in 0..ROUNDS {
            match (a.advance(&right).unwrap(), b.advance(&left).unwrap()) {
                (Step::Send(l), Step::Send(r)) => {
                    bytes += l.len() + r.len();
                    left = l;
                    right = r;
                }
                (Step::Done(l), Step::Done(r)) => {
                    assert_eq!(round, ROUNDS - 1);
                    assert_eq!(*l.values, *r.values);
                    let result = c.decode(&l).unwrap();
                    for ((g, u), output) in g.iter().zip(&u).zip(result) {
                        assert_eq!(p.evaluate(*g, *u).unwrap(), output, "g={g}, u={u}");
                    }
                    assert!(c.decode(&l).is_err());
                }
                _ => panic!("phase mismatch"),
            }
        }
        assert_eq!(bytes, resources(g.len()).unwrap().peer_frame_bytes);
        assert!(a.advance(&right).is_err());
        assert!(a.start(&input).is_err());
    }

    #[test]
    fn residual_selection_exhausts_every_table_index_and_tail_boundary() {
        let profile = Profile::new();
        for rin in [0, 1, 511, 1023] {
            let rout = 0x923456789a;
            let keys = Lookup::issue(rin, rout).unwrap();
            for x in 0..1024 {
                let masked = (x + rin) & 1023;
                let (l, sl) = keys[0].start(0, masked, profile.table());
                let (r, sr) = keys[1].start(1, masked, profile.table());
                let opened = l.wrapping_add(r) & mask(N);
                let result = keys[0]
                    .finish(opened, sl)
                    .wrapping_add(keys[1].finish(opened, sr))
                    .wrapping_sub(rout)
                    & mask(N);
                assert_eq!(result, profile.table()[x as usize] as u64);
            }
        }
        for rin in [0, 1, 1023, 1024, 4095, 4096, 8191] {
            let keys = Signs::issue(rin, 1, 1).unwrap();
            for x in -3072i64..=3072 {
                let masked = (x as u64).wrapping_add(rin) & mask(Y);
                let l = keys[0].eval(0, masked);
                let r = keys[1].eval(1, masked);
                assert_eq!(l[0] ^ r[0] ^ 1, u64::from(x >= 0));
                assert_eq!(l[1] ^ r[1] ^ 1, u64::from((-1023..1024).contains(&x)));
            }
        }
    }

    #[test]
    fn malformed_replay_cross_session_cancel_and_invalid_client_burn() {
        assert!(resources(MAX_LANES + 1).is_err());
        for phase in 0..ROUNDS {
            let (mut c, [mut a, mut b]) = issue_reference(context(), 1).unwrap();
            let input = c.encode(&[0], &[0]).unwrap();
            let mut l = a.start(&input).unwrap();
            let mut r = b.start(&input).unwrap();
            for _ in 0..phase {
                let Step::Send(nl) = a.advance(&r).unwrap() else {
                    panic!()
                };
                let Step::Send(nr) = b.advance(&l).unwrap() else {
                    panic!()
                };
                l = nl;
                r = nr;
            }
            let good = r.clone();
            r[40] ^= 1;
            assert!(a.advance(&r).is_err());
            assert!(a.advance(&good).is_err());
            assert!(a.keys.is_empty());
            b.cancel();
            assert!(b.advance(&l).is_err());
        }
        let (mut c, [mut a, mut b]) = issue_reference(context(), 1).unwrap();
        let input = c.encode(&[0], &[0]).unwrap();
        let l = a.start(&input).unwrap();
        let r = b.start(&input).unwrap();
        assert!(matches!(a.advance(&r), Ok(Step::Send(_))));
        assert!(a.advance(&r).is_err());
        assert!(a.keys.is_empty());
        let (mut c2, [mut other, _]) = issue_reference(context(), 1).unwrap();
        assert!(other.start(&input).is_err());
        assert!(other.keys.is_empty());
        assert!(c2.encode(&[i64::MAX], &[0]).is_err());
        assert!(c2.encode(&[0], &[0]).is_err());
        b.cancel();
        assert!(b.advance(&l).is_err());
    }

    #[test]
    fn fused_complete_block_matches_unfused_at_shift_interval_and_signed_boundaries() {
        let profile = Profile::new();
        let mut gates = vec![-(48 << 24), 48 << 24];
        for index in [-3072i64, -1024, -1023, -1, 0, 1, 1023, 1024, 3072] {
            for residue in [-4097, -4096, -1, 0, 1, 4095, 4096] {
                let x = (index << 18) + residue;
                if x.unsigned_abs() <= (GATE_BOUND as u64) << 24 {
                    gates.push(x);
                }
            }
        }
        for layout in [Layout::Unfused, Layout::Fused] {
            for up in [-(128 << 24), -1, 0, 1, 128 << 24] {
                let (mut c, [mut a, mut b]) =
                    issue_reference_with_layout(context(), gates.len(), layout).unwrap();
                let ups = vec![up; gates.len()];
                let input = c.encode(&gates, &ups).unwrap();
                let mut left = a.start(&input).unwrap();
                let mut right = b.start(&input).unwrap();
                let mut bytes = left.len() + right.len();
                for round in 0..layout.rounds() {
                    match (a.advance(&right).unwrap(), b.advance(&left).unwrap()) {
                        (Step::Send(l), Step::Send(r)) => {
                            bytes += l.len() + r.len();
                            left = l;
                            right = r;
                        }
                        (Step::Done(l), Step::Done(r)) => {
                            assert_eq!(round + 1, layout.rounds());
                            assert_eq!(*l.values, *r.values);
                            let output = c.decode(&l).unwrap();
                            for (g, y) in gates.iter().zip(output) {
                                assert_eq!(
                                    y,
                                    profile.evaluate(*g, up).unwrap(),
                                    "{layout:?} {g} {up}"
                                );
                            }
                            assert!(c.decode(&l).is_err());
                        }
                        _ => panic!("phase mismatch"),
                    }
                }
                assert_eq!(
                    bytes,
                    resources_with_layout(gates.len(), layout)
                        .unwrap()
                        .peer_frame_bytes
                );
                assert!(a.advance(&right).is_err());
                assert!(a.keys.is_empty());
            }
        }
        let a = resources_with_layout(64, Layout::Unfused).unwrap();
        let b = resources_with_layout(64, Layout::Fused).unwrap();
        assert_eq!(a.peer_payload_bits, b.peer_payload_bits);
        assert_eq!(a.peer_frame_bytes - b.peer_frame_bytes, 6 * HEADER);
        assert!(a.party_key_bits < b.party_key_bits);
        assert!(resources_with_layout(MAX_LANES, Layout::Fused).is_ok());
    }

    #[test]
    fn fused_predicate_views_match_floor_reduction_including_mask_wraps() {
        for rin in [0, 1, (1 << 18) - 1, (1 << 30) - 1, 1 << 30, (1 << 31) - 1] {
            let keys = Signs::issue_scaled(INPUT_GATE_BITS, 18, rin, 1, 1).unwrap();
            for index in -3072i64..=3072 {
                for low in [0, (1 << 18) - 1] {
                    let value = (index << 18) + low;
                    let x = (value as u64).wrapping_add(rin) & mask(INPUT_GATE_BITS);
                    let a = keys[0].eval(0, x);
                    let b = keys[1].eval(1, x);
                    assert_eq!(a[0] ^ b[0] ^ 1, u64::from(index >= 0));
                    assert_eq!(a[1] ^ b[1] ^ 1, u64::from((-1023..1024).contains(&index)));
                }
            }
        }
    }

    #[test]
    fn fused_phase_forgery_and_layout_mismatch_burn_material() {
        for phase in 0..Layout::Fused.rounds() {
            let (mut c, [mut a, mut b]) =
                issue_reference_with_layout(context(), 1, Layout::Fused).unwrap();
            let input = c.encode(&[1], &[-1]).unwrap();
            let mut left = a.start(&input).unwrap();
            let mut right = b.start(&input).unwrap();
            for _ in 0..phase {
                let Step::Send(l) = a.advance(&right).unwrap() else {
                    panic!()
                };
                let Step::Send(r) = b.advance(&left).unwrap() else {
                    panic!()
                };
                left = l;
                right = r;
            }
            let valid = right.clone();
            right[73] = (phase + 1) as u8;
            assert_eq!(a.advance(&right).err(), Some(Error::Context));
            assert_eq!(a.advance(&valid).err(), Some(Error::Consumed));
            assert!(a.keys.is_empty());
            b.cancel();
            assert!(b.advance(&left).is_err());
        }
        let (mut c, _) = issue_reference(context(), 1).unwrap();
        let (_, [mut a, _]) = issue_reference_with_layout(context(), 1, Layout::Fused).unwrap();
        let input = c.encode(&[0], &[0]).unwrap();
        assert_eq!(a.start(&input).err(), Some(Error::Context));
        assert!(a.keys.is_empty());
    }
}
