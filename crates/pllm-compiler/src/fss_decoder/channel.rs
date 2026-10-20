//! Bounded client/request boundary for the generated native FSS decoder.
//!
//! SHA-256 framing checks detect corruption; they are NOT authentication or a
//! proof of honest evaluation. Authenticated, recipient-bound SDK channels and a
//! session/epoch replay registry are mandatory. This module performs no HTTP,
//! preprocessing, intermediate export, source loading, or worker reconstruction.
use super::{Error, Limits, Program, TokenShares, MASK, MAX_VALUES};
use pllm_garble::shared_rescale_reference::Context;
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const MAX_MANIFEST_BYTES: usize = 1024;
pub const REQUEST_BYTES: usize = 248;
pub const FRAME_HEADER_BYTES: usize = 112;
pub const FRAME_TRAILER_BYTES: usize = 32;
pub const MAX_INPUT_FRAME_BYTES: usize = FRAME_HEADER_BYTES + 16 * 8 + FRAME_TRAILER_BYTES;
pub const MAX_OUTPUT_FRAME_BYTES: usize = FRAME_HEADER_BYTES + 4 * 8 + FRAME_TRAILER_BYTES;
const REQUEST_MAGIC: &[u8; 8] = b"PLLMRQ01";
const INPUT_MAGIC: &[u8; 8] = b"PLLMIN01";
const OUTPUT_MAGIC: &[u8; 8] = b"PLLMOU01";
const SCHEMA: &[u8] = b"pllm.fss.channel.program.v1";
const GPT2: &str = "fusefss.gpt2-complete-ring50-q12.v1";
const ROPE: &str = "fusefss.dense-rope-ring50-q12.v1";
const SCRATCH_BYTES: usize = 4096;

/// Actual-byte identity of the bounded public SDK source contract. Source-file
/// authentication and comparison with locally lowered metadata are separate.
pub fn contract_digest(bytes: &[u8]) -> Result<[u8; 32], Error> {
    if bytes.is_empty() || bytes.len() > 1 << 20 {
        return Err(Error::Resource);
    }
    Ok(Sha256::digest(bytes).into())
}

/// Public bounded geometry. Byte counts are logical capacities, not RSS. The
/// client peak includes both retained exported inputs, both final replies and
/// reconstructed u32 IDs. Extra caller/Python/HTTP copies are separate owners.
/// These bounds are not Budget leases: the SDK reservation must outlive returned
/// frames, imported word owners and callbacks, including after close/cancel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub manifest_bytes: usize,
    pub request_bytes: usize,
    pub input_frame_bytes: usize,
    pub output_frame_bytes: usize,
    pub client_peak_bytes: usize,
    pub worker_input_import_bytes: usize,
    pub worker_output_export_bytes: usize,
}

#[derive(Clone, Copy)]
struct Geometry {
    contract: [u8; 32],
    program: [u8; 32],
    source: [u8; 32],
    input: [u8; 32],
    profile: u8,
    initial: usize,
    selections: usize,
    vocabulary: usize,
}
impl Geometry {
    fn validate(&self) -> Result<(), Error> {
        let hard = Limits::default();
        if self.contract == [0; 32]
            || self.program == [0; 32]
            || self.source == [0; 32]
            || self.input == [0; 32]
            || self.profile > 1
            || self.initial == 0
            || self.initial > hard.max_sequence
            || self.selections == 0
            || self.selections > hard.max_selections
            || self
                .initial
                .checked_add(self.selections - 1)
                .filter(|n| *n <= hard.max_sequence)
                .is_none()
            || self.vocabulary == 0
            || self.vocabulary > MAX_VALUES
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
    fn profile(&self) -> &'static str {
        if self.profile == 0 {
            GPT2
        } else {
            ROPE
        }
    }
}

/// Canonical public manifest derived only from a locally compiled Program.
/// Fixed inline storage avoids attacker-sized JSON allocations during parsing.
/// Public metadata may be copied; private request/frame owners are not Clone.
#[derive(Clone)]
pub struct Manifest {
    geometry: Geometry,
    bytes: [u8; MAX_MANIFEST_BYTES],
    length: usize,
    digest: [u8; 32],
}
impl Manifest {
    pub fn new(
        contract_digest: [u8; 32],
        input_digest: [u8; 32],
        program: &Program,
    ) -> Result<Self, Error> {
        let profile = match program.profile() {
            GPT2 => 0,
            ROPE => 1,
            _ => return Err(Error::Unsupported),
        };
        Self::from_geometry(Geometry {
            contract: contract_digest,
            program: program.digest(),
            source: program.source_digest(),
            input: input_digest,
            profile,
            initial: program.initial_tokens(),
            selections: program.selections(),
            vocabulary: program.vocabulary(),
        })
    }
    fn from_geometry(geometry: Geometry) -> Result<Self, Error> {
        geometry.validate()?;
        let mut manifest = Self {
            geometry,
            bytes: [0; MAX_MANIFEST_BYTES],
            length: 0,
            digest: [0; 32],
        };
        let mut out = PublicWriter {
            bytes: &mut manifest.bytes,
            offset: 0,
        };
        out.literal(b"{\"contract\":\"")?;
        out.hex(&geometry.contract)?;
        out.literal(b"\",\"initial\":")?;
        out.number(geometry.initial)?;
        out.literal(b",\"input_digest\":\"")?;
        out.hex(&geometry.input)?;
        out.literal(b"\",\"profile\":\"")?;
        out.literal(geometry.profile().as_bytes())?;
        out.literal(b"\",\"program\":\"")?;
        out.hex(&geometry.program)?;
        out.literal(b"\",\"schema\":\"")?;
        out.literal(SCHEMA)?;
        out.literal(b"\",\"selections\":")?;
        out.number(geometry.selections)?;
        out.literal(b",\"source\":\"")?;
        out.hex(&geometry.source)?;
        out.literal(b"\",\"vocabulary\":")?;
        out.number(geometry.vocabulary)?;
        out.literal(b"}")?;
        manifest.length = out.offset;
        manifest.digest = Sha256::new()
            .chain_update(b"pllm.fss.channel.program.v1\0")
            .chain_update(manifest.as_bytes())
            .finalize()
            .into();
        Ok(manifest)
    }
    fn parse(bytes: &[u8], expected_contract: [u8; 32]) -> Result<Self, Error> {
        if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
            return Err(Error::Resource);
        }
        let mut input = PublicReader { bytes, offset: 0 };
        input.literal(b"{\"contract\":\"")?;
        let contract = input.hex()?;
        input.literal(b"\",\"initial\":")?;
        let initial = input.number(16)?;
        input.literal(b",\"input_digest\":\"")?;
        let source_input = input.hex()?;
        input.literal(b"\",\"profile\":\"")?;
        let profile = if input.remaining().starts_with(GPT2.as_bytes()) {
            input.literal(GPT2.as_bytes())?;
            0
        } else {
            input.literal(ROPE.as_bytes())?;
            1
        };
        input.literal(b"\",\"program\":\"")?;
        let program = input.hex()?;
        input.literal(b"\",\"schema\":\"")?;
        input.literal(SCHEMA)?;
        input.literal(b"\",\"selections\":")?;
        let selections = input.number(4)?;
        input.literal(b",\"source\":\"")?;
        let source = input.hex()?;
        input.literal(b"\",\"vocabulary\":")?;
        let vocabulary = input.number(MAX_VALUES)?;
        input.literal(b"}")?;
        if input.offset != bytes.len() || contract != expected_contract {
            return Err(Error::Context);
        }
        let manifest = Self::from_geometry(Geometry {
            contract,
            program,
            source,
            input: source_input,
            profile,
            initial,
            selections,
            vocabulary,
        })?;
        if manifest.as_bytes() != bytes {
            return Err(Error::Context);
        }
        Ok(manifest)
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn contract_digest(&self) -> [u8; 32] {
        self.geometry.contract
    }
    pub fn program_digest(&self) -> [u8; 32] {
        self.geometry.program
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.geometry.source
    }
    /// Original source dtype/shape/IEEE payload identity supplied by the trusted
    /// SourceBuilder. This is not an input-token digest or checkpoint-file proof.
    pub fn input_digest(&self) -> [u8; 32] {
        self.geometry.input
    }
    pub fn profile(&self) -> &'static str {
        self.geometry.profile()
    }
    pub fn initial_tokens(&self) -> usize {
        self.geometry.initial
    }
    pub fn selections(&self) -> usize {
        self.geometry.selections
    }
    pub fn vocabulary(&self) -> usize {
        self.geometry.vocabulary
    }
    pub fn resources(&self) -> Resources {
        let input = frame_bytes(self.geometry.initial);
        let output = frame_bytes(self.geometry.selections);
        Resources {
            manifest_bytes: self.length,
            request_bytes: REQUEST_BYTES,
            input_frame_bytes: input,
            output_frame_bytes: output,
            client_peak_bytes: size_of::<ClientRequest>()
                + 2 * input
                + 2 * output
                + 4 * self.geometry.selections
                + SCRATCH_BYTES,
            worker_input_import_bytes: size_of::<Request>()
                + input
                + 8 * self.geometry.initial
                + SCRATCH_BYTES,
            worker_output_export_bytes: size_of::<Request>()
                + size_of::<TokenShares>()
                + 8 * self.geometry.selections
                + output
                + SCRATCH_BYTES,
        }
    }
}

struct PublicWriter<'a> {
    bytes: &'a mut [u8],
    offset: usize,
}
impl PublicWriter<'_> {
    fn literal(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let end = self
            .offset
            .checked_add(bytes.len())
            .ok_or(Error::Resource)?;
        self.bytes
            .get_mut(self.offset..end)
            .ok_or(Error::Resource)?
            .copy_from_slice(bytes);
        self.offset = end;
        Ok(())
    }
    fn hex(&mut self, bytes: &[u8; 32]) -> Result<(), Error> {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for value in bytes {
            self.literal(&[HEX[(*value >> 4) as usize], HEX[(*value & 15) as usize]])?;
        }
        Ok(())
    }
    fn number(&mut self, mut n: usize) -> Result<(), Error> {
        let mut digits = [0; 20];
        let mut offset = digits.len();
        loop {
            offset -= 1;
            digits[offset] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        self.literal(&digits[offset..])
    }
}
struct PublicReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl PublicReader<'_> {
    fn remaining(&self) -> &[u8] {
        &self.bytes[self.offset..]
    }
    fn literal(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if !self.remaining().starts_with(bytes) {
            return Err(Error::Context);
        }
        self.offset += bytes.len();
        Ok(())
    }
    fn hex(&mut self) -> Result<[u8; 32], Error> {
        let bytes = self.remaining().get(..64).ok_or(Error::Context)?;
        let nibble = |b| -> Result<u8, Error> {
            match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err(Error::Context),
            }
        };
        let mut digest = [0; 32];
        for (out, pair) in digest.iter_mut().zip(bytes.chunks_exact(2)) {
            *out = nibble(pair[0])? * 16 + nibble(pair[1])?;
        }
        self.offset += 64;
        Ok(digest)
    }
    fn number(&mut self, maximum: usize) -> Result<usize, Error> {
        let mut count = 0;
        let mut value = 0usize;
        for &byte in self.remaining().iter().take(6) {
            if !byte.is_ascii_digit() {
                break;
            }
            count += 1;
            value = value
                .checked_mul(10)
                .and_then(|n| n.checked_add((byte - b'0') as usize))
                .filter(|n| *n <= maximum)
                .ok_or(Error::Context)?;
        }
        if count == 0 || (count > 1 && self.remaining()[0] == b'0') {
            return Err(Error::Context);
        }
        self.offset += count;
        Ok(value)
    }
}

fn request_bytes(manifest: &Manifest, session: [u8; 32]) -> Result<[u8; REQUEST_BYTES], Error> {
    if session == [0; 32] {
        return Err(Error::Context);
    }
    let g = manifest.geometry;
    let mut bytes = [0; REQUEST_BYTES];
    bytes[..8].copy_from_slice(REQUEST_MAGIC);
    bytes[8..40].copy_from_slice(&manifest.digest);
    bytes[40..72].copy_from_slice(&session);
    bytes[72..76].copy_from_slice(&(g.initial as u32).to_le_bytes());
    bytes[76..80].copy_from_slice(&(g.selections as u32).to_le_bytes());
    bytes[80..84].copy_from_slice(&(g.vocabulary as u32).to_le_bytes());
    bytes[84..88].copy_from_slice(&[50, g.profile, 0, 0]);
    bytes[88..120].copy_from_slice(&g.contract);
    bytes[120..152].copy_from_slice(&g.program);
    bytes[152..184].copy_from_slice(&g.source);
    bytes[184..216].copy_from_slice(&g.input);
    let checksum = Sha256::digest(&bytes[..216]);
    bytes[216..].copy_from_slice(&checksum);
    Ok(bytes)
}

#[derive(Clone, Copy)]
enum Phase {
    Ready,
    Input(u8),
}

/// One local worker/dealer request. Parsing cannot enforce process-wide replay:
/// the authenticated SDK must retain its session/epoch registry. This owner
/// itself accepts only one input and one final output; every failed method burns.
pub struct Request {
    manifest: Manifest,
    bytes: [u8; REQUEST_BYTES],
    context: Context,
    phase: Option<Phase>,
}
impl Request {
    pub fn parse(bytes: &[u8], expected_manifest: &Manifest) -> Result<Self, Error> {
        if bytes.len() != REQUEST_BYTES {
            return Err(Error::Context);
        }
        let session = bytes[40..72].try_into().unwrap();
        let expected = request_bytes(expected_manifest, session)?;
        if bytes != expected {
            return Err(Error::Context);
        }
        let operation = Sha256::new()
            .chain_update(b"pllm.fss.channel.request.v1\0")
            .chain_update(bytes)
            .finalize()
            .into();
        if operation == [0; 32] {
            return Err(Error::Context);
        }
        Ok(Self {
            manifest: expected_manifest.clone(),
            bytes: expected,
            context: Context {
                plan: expected_manifest.program_digest(),
                session,
                operation,
                first_tensor_index: 0,
            },
            phase: Some(Phase::Ready),
        })
    }
    pub fn context(&self) -> Context {
        self.context
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn close(&mut self) {
        self.phase = None;
    }
    pub fn cancel(&mut self) {
        self.close();
    }
    pub fn is_consumed(&self) -> bool {
        self.phase.is_none()
    }
    pub fn take_input(&mut self, role: u8, frame: &[u8]) -> Result<Zeroizing<Vec<u64>>, Error> {
        let Phase::Ready = self.phase.take().ok_or(Error::Consumed)? else {
            return Err(Error::Consumed);
        };
        check_frame(
            frame,
            INPUT_MAGIC,
            self.context,
            role,
            self.manifest.initial_tokens(),
            0,
        )?;
        let mut values = Zeroizing::new(allocate(self.manifest.initial_tokens())?);
        for word in frame[FRAME_HEADER_BYTES..frame.len() - FRAME_TRAILER_BYTES].chunks_exact(8) {
            values.push(u64::from_le_bytes(word.try_into().unwrap()));
        }
        self.phase = Some(Phase::Input(role));
        Ok(values)
    }
    /// Only genuine native final shares enter this constructor. The bridge must
    /// use shares produced by this request's Worker; TokenShares authenticates
    /// native origin/program/role, not a proof against a malicious evaluator.
    pub fn output(&mut self, role: u8, shares: TokenShares) -> Result<Zeroizing<Vec<u8>>, Error> {
        let Phase::Input(input_role) = self.phase.take().ok_or(Error::Consumed)? else {
            return Err(Error::Context);
        };
        if role > 1
            || role != input_role
            || shares.role() != role
            || shares.program_digest() != self.context.plan
            || shares.values().len() != self.manifest.selections()
            || shares.values().iter().any(|v| *v > MASK)
        {
            return Err(Error::Context);
        }
        let mut frame = new_frame(OUTPUT_MAGIC, self.context, role, shares.values().len(), 1)?;
        for (chunk, word) in frame
            [FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + shares.values().len() * 8]
            .chunks_exact_mut(8)
            .zip(shares.values())
        {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        seal(&mut frame);
        Ok(frame)
    }
}

struct ClientState {
    inputs: [Option<Zeroizing<Vec<u8>>>; 2],
    sent: u8,
}
/// Trusted, noncloneable client owner. No IDs or sibling shares are exposed to
/// any evaluator. Take each recipient frame once; final reconstruction consumes
/// this owner. Explicit close/cancel/drop erase every remaining private frame.
pub struct ClientRequest {
    request: Request,
    state: Option<ClientState>,
}
impl ClientRequest {
    pub fn new(
        expected_contract_digest: [u8; 32],
        session: [u8; 32],
        manifests: [&[u8]; 3],
        token_ids: &[u32],
    ) -> Result<Self, Error> {
        Self::with_random(
            expected_contract_digest,
            session,
            manifests,
            token_ids,
            |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
        )
    }
    fn with_random(
        expected_contract: [u8; 32],
        session: [u8; 32],
        manifests: [&[u8]; 3],
        token_ids: &[u32],
        mut random: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        // All public geometry and token validation precedes allocation and RNG.
        if manifests
            .iter()
            .any(|m| m.is_empty() || m.len() > MAX_MANIFEST_BYTES)
        {
            return Err(Error::Resource);
        }
        if manifests[0] != manifests[1] || manifests[0] != manifests[2] {
            return Err(Error::Context);
        }
        let manifest = Manifest::parse(manifests[0], expected_contract)?;
        if token_ids.len() != manifest.initial_tokens()
            || token_ids
                .iter()
                .any(|id| *id as usize >= manifest.vocabulary())
        {
            return Err(Error::Domain);
        }
        let bytes = request_bytes(&manifest, session)?;
        let request = Request::parse(&bytes, &manifest)?;
        // Both exact-capacity owners exist before the first OS-random draw.
        let mut left = new_frame(INPUT_MAGIC, request.context, 0, token_ids.len(), 0)?;
        let mut right = new_frame(INPUT_MAGIC, request.context, 1, token_ids.len(), 0)?;
        for (i, &token) in token_ids.iter().enumerate() {
            let mut entropy = Zeroizing::new([0; 8]);
            random(&mut *entropy)?;
            let a = Zeroizing::new(u64::from_le_bytes(*entropy) & MASK);
            let b = Zeroizing::new((token as u64).wrapping_sub(*a) & MASK);
            let start = FRAME_HEADER_BYTES + i * 8;
            left[start..start + 8].copy_from_slice(&a.to_le_bytes());
            right[start..start + 8].copy_from_slice(&b.to_le_bytes());
        }
        seal(&mut left);
        seal(&mut right);
        Ok(Self {
            request,
            state: Some(ClientState {
                inputs: [Some(left), Some(right)],
                sent: 0,
            }),
        })
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        self.request.manifest_bytes()
    }
    pub fn close(&mut self) {
        self.state = None;
        self.request.close();
    }
    pub fn cancel(&mut self) {
        self.close();
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }
    pub fn take_input(&mut self, role: u8) -> Result<Zeroizing<Vec<u8>>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if role > 1 {
            return Err(Error::Domain);
        }
        let frame = state
            .inputs
            .get_mut(role as usize)
            .and_then(Option::take)
            .ok_or(Error::Consumed)?;
        state.sent |= 1 << role;
        self.state = Some(state);
        Ok(frame)
    }
    pub fn finish(
        mut self,
        response0: &[u8],
        response1: &[u8],
    ) -> Result<Zeroizing<Vec<u32>>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        if state.sent != 3 {
            return Err(Error::Context);
        }
        let count = self.request.manifest.selections();
        check_frame(response0, OUTPUT_MAGIC, self.request.context, 0, count, 1)?;
        check_frame(response1, OUTPUT_MAGIC, self.request.context, 1, count, 1)?;
        let token = |i| -> Result<u32, Error> {
            let start = FRAME_HEADER_BYTES + i * 8;
            let a = Zeroizing::new(u64::from_le_bytes(
                response0[start..start + 8].try_into().unwrap(),
            ));
            let b = Zeroizing::new(u64::from_le_bytes(
                response1[start..start + 8].try_into().unwrap(),
            ));
            let sum = Zeroizing::new(a.wrapping_add(*b) & MASK);
            if *sum >= self.request.manifest.vocabulary() as u64 {
                return Err(Error::Domain);
            }
            Ok(*sum as u32)
        };
        // Every reconstructed ID passes vocabulary checks before output allocation.
        for i in 0..count {
            let _ = Zeroizing::new(token(i)?);
        }
        let mut values = Zeroizing::new(allocate(count)?);
        for i in 0..count {
            values.push(token(i)?);
        }
        Ok(values)
    }
}

impl Drop for ClientRequest {
    fn drop(&mut self) {
        self.close();
    }
}

fn frame_bytes(count: usize) -> usize {
    FRAME_HEADER_BYTES + 8 * count + FRAME_TRAILER_BYTES
}
fn new_frame(
    magic: &[u8; 8],
    context: Context,
    role: u8,
    count: usize,
    kind: u8,
) -> Result<Zeroizing<Vec<u8>>, Error> {
    if role > 1 || count == 0 || count > 16 || (kind == 1 && count > 4) {
        return Err(Error::Domain);
    }
    let mut bytes = Zeroizing::new(allocate(frame_bytes(count))?);
    bytes.resize(frame_bytes(count), 0);
    bytes[..8].copy_from_slice(magic);
    bytes[8..40].copy_from_slice(&context.operation);
    bytes[40..72].copy_from_slice(&context.plan);
    bytes[72..104].copy_from_slice(&context.session);
    bytes[104..108].copy_from_slice(&[role, 50, kind, 0]);
    bytes[108..112].copy_from_slice(&(count as u32).to_le_bytes());
    Ok(bytes)
}
fn seal(bytes: &mut [u8]) {
    let end = bytes.len() - FRAME_TRAILER_BYTES;
    let checksum = Sha256::digest(&bytes[..end]);
    bytes[end..].copy_from_slice(&checksum);
}
fn check_frame(
    bytes: &[u8],
    magic: &[u8; 8],
    context: Context,
    role: u8,
    count: usize,
    kind: u8,
) -> Result<(), Error> {
    if role > 1
        || bytes.len() != frame_bytes(count)
        || &bytes[..8] != magic
        || bytes[8..40] != context.operation
        || bytes[40..72] != context.plan
        || bytes[72..104] != context.session
        || bytes[104..108] != [role, 50, kind, 0]
        || bytes[108..112] != (count as u32).to_le_bytes()
        || bytes[bytes.len() - 32..] != Sha256::digest(&bytes[..bytes.len() - 32])[..]
        || bytes[FRAME_HEADER_BYTES..bytes.len() - 32]
            .chunks_exact(8)
            .any(|word| u64::from_le_bytes(word.try_into().unwrap()) > MASK)
    {
        return Err(Error::Context);
    }
    Ok(())
}

fn allocate<T>(count: usize) -> Result<Vec<T>, Error> {
    #[cfg(test)]
    FAIL_AFTER.with(|fail| {
        let left = fail.get();
        if left == 0 {
            return Err(Error::Allocation);
        }
        if left > 0 {
            fail.set(left - 1);
        }
        Ok(())
    })?;
    super::vector(count)
}
#[cfg(test)]
thread_local! { static FAIL_AFTER: std::cell::Cell<isize> = const { std::cell::Cell::new(-1) }; }

#[cfg(test)]
#[path = "channel_tests.rs"]
mod tests;
