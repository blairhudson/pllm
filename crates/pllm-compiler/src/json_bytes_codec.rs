//! Private, inert byte-to-code-point step. Not a JSON parser or frontend mount.
//!
//! SOURCE ONLY: compilation UNVERIFIED; tests UNRUN. Source basis is limited to
//! these six immutable CPython releases (not all patches in Python >=3.11,<3.14):
//! 3.11.0  deaf509e8fc6e0363bd6f26d52ad42f976ec42f2
//! 3.11.14 cd1c3a6342869b7346c1b5c27b8de9c6ef9c4e69
//! 3.12.0  0fb18b02c8ad56299d6a2910be0bab8ad601ef24
//! 3.12.12 4a5632fbf9bf59477c540e3f53fa7cdbeea3e3f5
//! 3.13.0  60403a5409ff2c3f3b07dd2ca91a7a3e096839c7
//! 3.13.9  8183fa5e3f78ca6ab862de7fb8b14f3d929421e0
//! Source URLs: https://github.com/python/cpython/blob/COMMIT/PATH, substituting
//! one commit above and Lib/json/__init__.py (detect_encoding, loads),
//! Objects/stringlib/codecs.h (utf8_decode, utf16_decode),
//! Objects/unicodeobject.c (unicode_decode_utf8, PyUnicode_DecodeUTF16Stateful,
//! PyUnicode_DecodeUTF32Stateful), Python/codecs.c (PyCodec_SurrogatePassErrors),
//! Lib/encodings/{utf_8_sig,utf_16,utf_32}.py (final decode), and
//! Modules/_codecsmodule.c (final/stateful dispatch). Astra's six-pin source
//! audit reported no relevant semantic differences. This implementation and its
//! source-authored vectors are NOT a verified equivalence or runtime claim.
//! Installed Python and every other patch release remain UNPROVED. No Python
//! runtime, vendored decoder, normalization, String, numeric or lexer dependency.
//! Same-counter funding is NOT raw-byte provenance: this step never certifies
//! that its incoming bytes equal the configured root's original configuration,
//! and never manufactures or adopts OriginalSourcePlan authority.
//!
//! Work units name logical visits, not instructions: each pass dispatches at
//! most N units, reads each byte at most twice (UTF-16 uses one cached unit),
//! and only replay writes, at most N points. Prepay 4N+2N+N plus fixed controls
//! and the shared attempt API's separate +1 fee, without refunds. Detection
//! adds 20 events: four scratch initializations, four guarded prefix probes, five
//! guarded BOM predicates, two length predicates, four zero predicates, one
//! selection. No terminal probe is hidden in the N dispatch allowance.
//! Retained capacity/logical controls only; allocator internals, CPU, physical
//! stack, RSS, legacy C9 depth-16 compatibility and whole-request fit unclaimed.

use crate::original_source::{
    Error as SourceError, OriginalSourcePlan, PrimitiveAttempt, ReserveBootstrapPhase,
    ReservedDecodeFunding,
};
use pllm_garble::shared_allocation::{
    Budget, Error as NativeError, OwnedCharged, OwnedExternalCharge,
};
use std::{mem::size_of, sync::OnceLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Encoding { Utf8, Utf16Le, Utf16Be, Utf32Le, Utf32Be }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeFailure {
    InvalidStart, InvalidContinuation, UnexpectedEnd, TruncatedUnit, OutOfRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecodeError {
    pub kind: DecodeFailure,
    pub start: usize,
    pub end: usize,
    /// UTF-8-sig exception input is the stripped slice. Add this to both spans
    /// for original coordinates. UTF-16/32 spans already include their BOM.
    pub error_input_base: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Counter, Capacity, Overflow, Work, Resource, Replay,
    Decode(DecodeError), Source(SourceError), Native(NativeError),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub max_source_bytes: usize,
    pub max_points: usize,
    pub max_work: u64,
    pub max_working_payload: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Resources {
    pub original_owner_bytes: usize,
    pub point_count: usize,
    pub point_capacity_bytes: usize,
    pub required_working_payload: usize,
    pub working_charge_bytes: usize,
    pub controller_charge_bytes: usize,
    /// Configured path includes begin_primitive_attempt's positive fee. Reserved
    /// path reports incremental codec/transition visits ONLY: its phase already
    /// paid its fee and earlier visits, which remain in the SAME root ledger.
    pub prepaid_work: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Detection { encoding: Encoding, bom_bytes: usize, error_input_base: usize }

/// Field order: output backing, original backing/charge, committed controller;
/// the enclosing OwnedCharged then retires the covering working guard.
pub(crate) struct DecodedJsonText {
    points: Vec<u32>,
    original: OwnedCharged<Vec<u8>>,
    controller: OnceLock<OwnedExternalCharge>,
    detection: Detection,
    resources: Resources,
}

impl DecodedJsonText {
    /// Every point <= 0x10ffff; surrogates are intentionally representable.
    pub(crate) fn codepoints(&self) -> &[u32] { &self.points }
    pub(crate) fn original_owner(&self) -> &OwnedCharged<Vec<u8>> { &self.original }
    pub(crate) fn resources(&self) -> &Resources { &self.resources }
    pub(crate) fn encoding(&self) -> Encoding { self.detection.encoding }
    pub(crate) fn bom_bytes(&self) -> usize { self.detection.bom_bytes }
}

/// Original must retire before covering working charge even on abandonment.
pub(crate) struct CodecQuote {
    original: OwnedCharged<Vec<u8>>,
    attempt: PrimitiveAttempt,
    working: OwnedExternalCharge,
    detection: Detection,
    resources: Resources,
}

struct Incoming {
    original: OwnedCharged<Vec<u8>>,
    controller: Option<OwnedExternalCharge>,
    working: OwnedExternalCharge,
}

// Fixed success-path logical events; alternate error/unwind paths use a subset.
// 8 admission checks; 3 length/work derivations; 1 attempt entry; 20 detector;
// 4 pass init/terminal; 4 checked quote finalizations (points, bytes, payload,
// charge); 3 working growth/inline admission/factory entry; 2 reserve/cap check;
// 1 replay count check; 6 aggregate owner moves (quote/destructure/capture/
// result/controller/publication); 1 commit; 7 retirement visits (partial Vec,
// output Vec, raw Vec, raw guard, controller, working guard, attempt).
// Trivial u8/u32 backing destruction has no element callbacks or hidden loop.
const FIXED_WORK: u64 = 8 + 3 + 1 + 20 + 4 + 4 + 3 + 2 + 1 + 6 + 1 + 7;

fn add(a: usize, b: usize) -> Result<usize, Error> { a.checked_add(b).ok_or(Error::Overflow) }
fn mul(a: usize, b: usize) -> Result<usize, Error> { a.checked_mul(b).ok_or(Error::Overflow) }

/// Explicit named logical slots and move/result overlap, no physical-frame claim.
/// The actual private attempt layout/Arc/mutex/commit controls remain priced by
/// its owner, in the independent controller; never replaced by a plain guard.
/// No inventory loop, callback allocation or source-authority adoption here.
pub(crate) fn codec_control_bytes() -> Result<usize, Error> {
    let n = mul(3, size_of::<CodecQuote>())?; // incoming, return, consuming move
    let n = add(n, mul(2, size_of::<Result<CodecQuote, Error>>())?)?;
    let n = add(n, mul(2, size_of::<Factory>())?)?; // capture and invocation
    let n = add(n, mul(2, size_of::<Incoming>())?)?; // admission/error owners
    let n = add(n, mul(2, size_of::<Decoder<'_>>())?)?; // count and replay
    let n = add(n, mul(3, size_of::<DecodedJsonText>())?)?;
    let n = add(n, mul(2, size_of::<OwnedCharged<DecodedJsonText>>())?)?;
    let n = add(n, size_of::<Result<OwnedCharged<DecodedJsonText>, NativeError>>())?;
    let n = add(n, mul(2, size_of::<OwnedCharged<Vec<u8>>>())?)?;
    let n = add(n, mul(2, size_of::<Vec<u32>>())?)?; // partial and move
    let n = add(n, mul(3, size_of::<OwnedExternalCharge>())?)?;
    let n = add(n, size_of::<Result<OwnedExternalCharge, SourceError>>())?;
    let n = add(n, size_of::<Result<PrimitiveAttempt, SourceError>>())?;
    let n = add(n, mul(3, size_of::<Detection>())?)?;
    let n = add(n, mul(3, size_of::<Resources>())?)?;
    let n = add(n, mul(2, size_of::<Limits>())?)?;
    let n = add(n, mul(4, size_of::<Error>())?)?;
    let n = add(n, mul(2, size_of::<Result<Option<u32>, Error>>())?)?;
    let n = add(n, mul(2, size_of::<Result<usize, Error>>())?)?;
    let n = add(n, mul(32, size_of::<usize>())?)?; // quote/layout/index/span temps
    let n = add(n, mul(8, size_of::<u64>())?)?;
    let n = add(n, mul(8, size_of::<u32>())?)?; // unit/pair/scalar temporaries
    let n = add(n, mul(8, size_of::<&[u8]>())?)?; // original, decoder, factory borrows
    let n = add(n, mul(8, size_of::<bool>())?)?; // guarded detector predicates
    add(n, 8) // four detector scratch bytes, four UTF-32 assembly bytes
}

fn work(n: usize) -> Result<u64, Error> {
    u64::try_from(n).ok().and_then(|n| n.checked_mul(7))
        .and_then(|n| n.checked_add(FIXED_WORK)).ok_or(Error::Overflow)
}

// Private borrowed payment marker, created ONLY after begin/debit succeeded.
// A real attempt/root/controller stays alive throughout count, quote and replay;
// neither a bool, a Budget reference nor this borrow can create source authority.
// Its one pointer fits one of codec_control_bytes' eight fat-borrow slots; the
// tuple return fits the existing Detection/usize slots. Old quotes are unchanged.
struct FundedDecode<'a> { _attempt: &'a PrimitiveAttempt }

impl FundedDecode<'_> {
    fn count(self, bytes: &[u8], limits: Limits) -> Result<(Detection, usize), Error> {
        let detection = detect(bytes);
        let mut decoder = Decoder::new(bytes, detection);
        let mut points = 0usize;
        while decoder.next()?.is_some() {
            points = points.checked_add(1).ok_or(Error::Overflow)?;
            if points > limits.max_points { return Err(Error::Resource); }
        }
        Ok((detection, points))
    }
}

/// New transition ONLY: conservative fixed logical events, no inventory walk.
/// 24 aggregate moves: entry raw/phase/working (3), admission/result (2), funding
/// wrapper (1), quote factory/raw/funding/working/detection/resources (6), count
/// payment/borrow/tuple (3), construct self/factory/working/capture (4), factory
/// decoder/funding/decoded/result (4), publication return (1).
/// 16 metadata predicates (controller association/capacity, original coverage,
/// counter/working association/control floor, length/local-work/payload limits,
/// growth/inline/reserve/capacity/replay/ledger guards; conservative overlaps).
/// 8 quote/layout groups; 4 entry/terminal pairs (source, quote, construct,
/// factory); 3 ledger debit/usable operations; 9 owner/control retirements.
/// These 64 events are ADDED to existing 7N+60, without a second attempt fee.
/// No new per-byte/unit loop. Allocator/physical frame/CPU/RSS still unqualified.
const RESERVED_TRANSITION_WORK: u64 = 24 + 16 + 8 + 4 + 3 + 9;

/// Pending payload and its uncommitted phase share the enclosing working guard.
/// Decoded Vec/raw/guard retire first, phase/root/controller next, working last.
/// No method releases phase, commits it, retypes points or installs configuration.
pub(crate) struct PendingDecodedCustody {
    decoded: DecodedJsonText,
    funding: ReservedDecodeFunding,
}

impl PendingDecodedCustody {
    pub(crate) fn decoded(&self) -> &DecodedJsonText { &self.decoded }
    pub(crate) fn point_capacity(&self) -> usize { self.decoded.points.capacity() }
    pub(crate) fn lexical_identity(&self, budget: &Budget) -> Result<(), SourceError> {
        self.funding.lexical_identity(budget)
    }
    pub(crate) fn lexical_debit(&self, amount: u128) -> Result<(), SourceError> {
        self.funding.lexical_debit(amount)
    }
}

struct PendingFactory {
    decoder: Factory,
    funding: ReservedDecodeFunding,
}

impl PendingFactory {
    fn build(self) -> Result<PendingDecodedCustody, NativeError> {
        let Self { decoder, funding } = self;
        // Existing replay, allocation and partial-output unwind implementation.
        // Its decoded/raw owners retire before this outer funding on failure.
        let decoded = decoder.build()?;
        Ok(PendingDecodedCustody { decoded, funding })
    }
}

pub(crate) struct ReservedCodecQuote {
    factory: PendingFactory,
    working: OwnedExternalCharge,
}

/// Metadata-only calculation, NOT an attempt or authority. Its fields cannot be
/// asserted by callers; only the genuine phase can supply ReservedDecodeFunding.
pub(crate) struct ReservedDecodeAdmission {
    n: usize,
    controls: usize,
    visits: u64,
    controller_charge_bytes: usize,
}

impl ReservedDecodeAdmission {
    pub(crate) fn visits(&self) -> u64 { self.visits }
}

/// Additional named move/result/inline overlap for the new transition. Baseline
/// codec, context, phase and primitive-attempt quotations are NOT changed.
/// Actual local Incoming layout is raw/phase/working; sum plus 9 alignment quanta
/// conservatively covers padding, without naming/exposing that local type.
pub(crate) fn reserved_codec_control_bytes() -> Result<usize, Error> {
    let n = codec_control_bytes()?;
    let n = add(n, mul(3, size_of::<ReservedCodecQuote>())?)?;
    let n = add(n, mul(2, size_of::<Result<ReservedCodecQuote, Error>>())?)?;
    let n = add(n, mul(3, size_of::<PendingFactory>())?)?;
    let n = add(n, mul(3, size_of::<PendingDecodedCustody>())?)?;
    let n = add(n, mul(2, size_of::<OwnedCharged<PendingDecodedCustody>>())?)?;
    let n = add(n, size_of::<Result<OwnedCharged<PendingDecodedCustody>, NativeError>>())?;
    let n = add(n, mul(3, size_of::<ReservedDecodeFunding>())?)?;
    let n = add(n, mul(3, size_of::<ReserveBootstrapPhase>())?)?;
    let n = add(n, mul(3, size_of::<ReservedDecodeAdmission>())?)?;
    let n = add(n, size_of::<Result<ReservedDecodeAdmission, Error>>())?;
    let n = add(n, mul(3, size_of::<OwnedCharged<Vec<u8>>>())?)?;
    let n = add(n, mul(3, size_of::<OwnedExternalCharge>())?)?;
    let alignment = std::mem::align_of::<ReserveBootstrapPhase>()
        .max(std::mem::align_of::<OwnedCharged<Vec<u8>>>())
        .max(std::mem::align_of::<OwnedExternalCharge>());
    let n = add(n, mul(9, alignment)?)?; // three copies, three padding gaps each
    let n = add(n, mul(2, size_of::<Limits>())?)?; // source clamp/argument overlap
    let n = add(n, mul(8, size_of::<usize>())?)?;
    let n = add(n, mul(4, size_of::<u64>())?)?;
    add(n, mul(4, size_of::<&PrimitiveAttempt>())?)
}

pub(crate) fn reserve_decode_admission(
    raw: &OwnedCharged<Vec<u8>>, working: &OwnedExternalCharge, budget: &Budget,
    controller_charge_bytes: usize,
) -> Result<ReservedDecodeAdmission, Error> {
    if raw.charged_bytes() < add(size_of::<OwnedCharged<Vec<u8>>>(), raw.payload().capacity())? {
        return Err(Error::Capacity);
    }
    if !std::ptr::eq(raw.budget(), budget) || !working.is_for(budget) { return Err(Error::Counter); }
    let controls = reserved_codec_control_bytes()?;
    if working.payload_bytes() < controls { return Err(Error::Capacity); }
    let n = raw.payload().len();
    let visits = work(n)?.checked_add(RESERVED_TRANSITION_WORK).ok_or(Error::Overflow)?;
    Ok(ReservedDecodeAdmission { n, controls, visits, controller_charge_bytes })
}

pub(crate) fn quote_reserved(
    original: OwnedCharged<Vec<u8>>, working: OwnedExternalCharge,
    funding: ReservedDecodeFunding, admission: ReservedDecodeAdmission, limits: Limits,
) -> Result<ReservedCodecQuote, Error> {
    // Bind aggregate BEFORE any local limit/error/content work.
    let mut quote = ReservedCodecQuote {
        factory: PendingFactory {
            decoder: Factory { original, detection: Detection { encoding: Encoding::Utf8,
                bom_bytes: 0, error_input_base: 0 }, resources: Resources {
                original_owner_bytes: 0, point_count: 0, point_capacity_bytes: 0,
                required_working_payload: 0, working_charge_bytes: 0,
                controller_charge_bytes: admission.controller_charge_bytes, prepaid_work: admission.visits,
            } }, funding,
        }, working,
    };
    if admission.n > limits.max_source_bytes || admission.visits > limits.max_work { return Err(Error::Work); }
    let (detection, point_count) = FundedDecode {
        _attempt: quote.factory.funding.codec_attempt(),
    }.count(quote.factory.decoder.original.payload(), limits)?;
    let point_capacity_bytes = mul(point_count, size_of::<u32>())?;
    let required_working_payload = add(admission.controls, point_capacity_bytes)?;
    let actual_payload = quote.working.payload_bytes().max(required_working_payload);
    if actual_payload > limits.max_working_payload { return Err(Error::Resource); }
    let inline = size_of::<OwnedCharged<PendingDecodedCustody>>()
        .checked_sub(size_of::<OwnedExternalCharge>()).ok_or(Error::Overflow)?;
    quote.factory.decoder.detection = detection;
    quote.factory.decoder.resources = Resources {
        original_owner_bytes: quote.factory.decoder.original.charged_bytes(), point_count,
        point_capacity_bytes, required_working_payload,
        working_charge_bytes: add(add(actual_payload, inline)?, size_of::<OwnedExternalCharge>())?,
        controller_charge_bytes: admission.controller_charge_bytes, prepaid_work: admission.visits,
    };
    Ok(quote)
}

impl ReservedCodecQuote {
    pub(crate) fn resources(&self) -> &Resources { &self.factory.decoder.resources }
    pub(crate) fn construct(mut self) -> Result<OwnedCharged<PendingDecodedCustody>, Error> {
        self.factory.funding.check_usable().map_err(Error::Source)?;
        let required = self.resources().required_working_payload;
        if self.working.payload_bytes() < required {
            self.working.resize_payload(required).map_err(Error::Native)?;
        }
        // Aggregate field order covers growth failure. try_build covers inline
        // refusal, factory failure and unwind before working charge retirement.
        let Self { factory, working } = self;
        let output = working.try_build(move |_| factory.build()).map_err(Error::Native)?;
        // Sticky check only, NOT commit/install or a future publication permit.
        output.payload().funding.check_usable().map_err(Error::Source)?;
        Ok(output)
    }
}

pub(crate) fn quote_codec(
    original: OwnedCharged<Vec<u8>>,
    working: OwnedExternalCharge,
    root: &OriginalSourcePlan,
    controller: OwnedExternalCharge,
    limits: Limits,
) -> Result<CodecQuote, Error> {
    let mut incoming = Incoming { original, controller: Some(controller), working };
    let original = &incoming.original;
    let working = &incoming.working;
    let controller = incoming.controller.as_ref().ok_or(Error::Capacity)?;
    // Metadata only before all capacity/counter checks and the actual debit.
    let minimum = add(size_of::<OwnedCharged<Vec<u8>>>(), original.payload().capacity())?;
    if original.charged_bytes() < minimum { return Err(Error::Capacity); }
    if !std::ptr::eq(original.budget(), root.budget())
        || !working.is_for(root.budget()) || !controller.is_for(root.budget()) {
        return Err(Error::Counter);
    }
    let controls = codec_control_bytes()?;
    if working.payload_bytes() < controls
        || controller.payload_bytes() < OriginalSourcePlan::primitive_attempt_control_bytes().map_err(Error::Source)? {
        return Err(Error::Capacity);
    }
    let n = original.payload().len();
    let visits = work(n)?;
    let prepaid_work = visits.checked_add(1).ok_or(Error::Overflow)?;
    let controller_charge_bytes = controller.charged_bytes();
    let attempt = root.begin_primitive_attempt(incoming.controller.take().ok_or(Error::Capacity)?, visits).map_err(Error::Source)?;
    // Failures after this point burn the actual source-wide ledger.
    if n > limits.max_source_bytes || prepaid_work > limits.max_work { return Err(Error::Work); }
    let (detection, point_count) = FundedDecode { _attempt: &attempt }
        .count(incoming.original.payload(), limits)?;
    let point_capacity_bytes = mul(point_count, size_of::<u32>())?;
    let required_working_payload = add(controls, point_capacity_bytes)?;
    let actual_payload = incoming.working.payload_bytes().max(required_working_payload);
    if actual_payload > limits.max_working_payload { return Err(Error::Resource); }
    let inline = size_of::<OwnedCharged<DecodedJsonText>>()
        .checked_sub(size_of::<OwnedExternalCharge>()).ok_or(Error::Overflow)?;
    // Excess incoming working capacity is never silently shrunk/recredited.
    let working_charge_bytes = add(add(actual_payload, inline)?, size_of::<OwnedExternalCharge>())?;
    let resources = Resources {
        original_owner_bytes: incoming.original.charged_bytes(), point_count,
        point_capacity_bytes, required_working_payload, working_charge_bytes,
        // The controller survives commit with precisely its incoming charge.
        controller_charge_bytes, prepaid_work,
    };
    let Incoming { original, working, .. } = incoming;
    Ok(CodecQuote { original, attempt, working, detection, resources })
}

struct Factory {
    original: OwnedCharged<Vec<u8>>,
    detection: Detection,
    resources: Resources,
}

impl Factory {
    fn build(self) -> Result<DecodedJsonText, NativeError> {
        #[cfg(test)]
        factory_entry();
        // self (original) was bound before points, so partial points retire first.
        let mut points = Vec::new();
        // Scoped, test-only refusal at the actual shared reservation boundary;
        // not an allocator replacement or evidence of a real allocation failure.
        #[cfg(test)]
        refuse_point_allocation()?;
        points.try_reserve_exact(self.resources.point_count).map_err(|_| NativeError::Allocation)?;
        if points.capacity() != self.resources.point_count { return Err(NativeError::Resource); }
        let mut decoder = Decoder::new(self.original.payload(), self.detection);
        while let Some(point) = decoder.next().map_err(|_| NativeError::Resource)? {
            if points.len() == points.capacity() { return Err(NativeError::Resource); }
            points.push(point);
            #[cfg(test)]
            after_point()?;
        }
        if points.len() != self.resources.point_count { return Err(NativeError::Resource); }
        Ok(DecodedJsonText {
            points, original: self.original, controller: OnceLock::new(),
            detection: self.detection, resources: self.resources,
        })
    }
}

impl CodecQuote {
    pub(crate) fn resources(&self) -> &Resources { &self.resources }
    pub(crate) fn construct(self) -> Result<OwnedCharged<DecodedJsonText>, Error> {
        let Self { original, attempt, mut working, detection, resources } = self;
        // Put original in a concrete capture before any fallible growth. On any
        // failure it retires before working; attempt still burns abandonment.
        let factory = Factory { original, detection, resources };
        if working.payload_bytes() < resources.required_working_payload {
            working.resize_payload(resources.required_working_payload).map_err(Error::Native)?;
        }
        // try_build admits inline/padding BEFORE entering the allocating factory;
        // its documented contract retires closure/partial owners before charge.
        let output = working.try_build(move |_| factory.build()).map_err(Error::Native)?;
        // No fallible output preparation follows commit. OnceLock is private,
        // newly empty, and nobody can publish/access output before this write.
        let controller = attempt.commit().map_err(Error::Source)?;
        // Infallible initialization of this private, newly empty inline slot.
        // No mutation of published resources, original or points is exposed.
        output.payload().controller.get_or_init(|| controller);
        Ok(output)
    }
}

fn detect(bytes: &[u8]) -> Detection {
    let mut p = [0u8; 4]; // four scratch initialization events
    let n = bytes.len();
    // Four guarded prefix-probe events (<=four reads). No detector loop,
    // iterator, repeated starts_with, or data-dependent prefix rescan.
    if n > 0 { p[0] = bytes[0]; }
    if n > 1 { p[1] = bytes[1]; }
    if n > 2 { p[2] = bytes[2]; }
    if n > 3 { p[3] = bytes[3]; }
    let b32be = n >= 4 && p == [0, 0, 0xfe, 0xff];
    let b32le = n >= 4 && p == [0xff, 0xfe, 0, 0];
    let b16be = n >= 2 && p[0] == 0xfe && p[1] == 0xff;
    let b16le = n >= 2 && p[0] == 0xff && p[1] == 0xfe;
    let b8 = n >= 3 && p[0] == 0xef && p[1] == 0xbb && p[2] == 0xbf;
    let four = n >= 4;
    let two = n == 2;
    let z0 = p[0] == 0;
    let z1 = p[1] == 0;
    let z2 = p[2] == 0;
    let z3 = p[3] == 0;
    let (encoding, bom_bytes) = if b32be { (Encoding::Utf32Be, 4) }
        else if b32le { (Encoding::Utf32Le, 4) }
        else if b16be { (Encoding::Utf16Be, 2) }
        else if b16le { (Encoding::Utf16Le, 2) }
        else if b8 { (Encoding::Utf8, 3) }
        else if four && z0 { (if z1 { Encoding::Utf32Be } else { Encoding::Utf16Be }, 0) }
        else if four && z1 { (if z2 && z3 { Encoding::Utf32Le } else { Encoding::Utf16Le }, 0) }
        else if two && z0 { (Encoding::Utf16Be, 0) }
        else if two && z1 { (Encoding::Utf16Le, 0) }
        else { (Encoding::Utf8, 0) };
    Detection { encoding, bom_bytes, error_input_base: if b8 { 3 } else { 0 } }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
    detection: Detection,
    // Non-low UTF-16 lookahead is retained, not reread or recursively decoded.
    cached: Option<u16>,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8], detection: Detection) -> Self {
        Self { bytes, pos: detection.bom_bytes, detection, cached: None }
    }
    fn error(&self, kind: DecodeFailure, start: usize, end: usize) -> Error {
        let base = self.detection.error_input_base;
        Error::Decode(DecodeError { kind, start: start - base, end: end - base, error_input_base: base })
    }
    fn next(&mut self) -> Result<Option<u32>, Error> {
        if self.pos == self.bytes.len() && self.cached.is_none() { return Ok(None); }
        match self.detection.encoding {
            Encoding::Utf8 => self.utf8().map(Some),
            Encoding::Utf16Le => self.utf16(true).map(Some),
            Encoding::Utf16Be => self.utf16(false).map(Some),
            Encoding::Utf32Le => self.utf32(true).map(Some),
            Encoding::Utf32Be => self.utf32(false).map(Some),
        }
    }
    fn utf8(&mut self) -> Result<u32, Error> {
        use DecodeFailure::*;
        let start = self.pos;
        let first = self.bytes[start];
        let width = match first { 0..=0x7f => 1, 0xc2..=0xdf => 2, 0xe0..=0xef => 3, 0xf0..=0xf4 => 4,
            _ => return Err(self.error(InvalidStart, start, start + 1)) };
        if width == 1 { self.pos += 1; return Ok(u32::from(first)); }
        let mut point = u32::from(first & (0x7f >> width));
        let mut i = 1;
        // At most three continuation reads, each byte read once. CPython's
        // initial ED A0..BF rejection is rescued ONLY by a complete surrogate
        // triple; otherwise its original [start,start+1) error survives.
        let mut surrogate_prefix = false;
        while i < width {
            if start + i == self.bytes.len() {
                return Err(self.error(if surrogate_prefix { InvalidContinuation } else { UnexpectedEnd },
                    start, if surrogate_prefix { start + 1 } else { self.bytes.len() }));
            }
            let byte = self.bytes[start + i];
            if byte & 0xc0 != 0x80
                || (i == 1 && ((first == 0xe0 && byte < 0xa0)
                    || (first == 0xf0 && byte < 0x90) || (first == 0xf4 && byte >= 0x90))) {
                return Err(self.error(InvalidContinuation, start, if surrogate_prefix { start + 1 } else { start + i }));
            }
            if i == 1 && first == 0xed && byte >= 0xa0 { surrogate_prefix = true; }
            point = (point << 6) | u32::from(byte & 0x3f);
            i += 1;
        }
        self.pos += width;
        Ok(point) // EACH surrogate triple stays one point, never paired.
    }
    fn unit16(&mut self, le: bool) -> u16 {
        let pair = [self.bytes[self.pos], self.bytes[self.pos + 1]];
        self.pos += 2;
        if le { u16::from_le_bytes(pair) } else { u16::from_be_bytes(pair) }
    }
    fn utf16(&mut self, le: bool) -> Result<u32, Error> {
        let unit = if let Some(unit) = self.cached.take() { unit } else {
            if self.bytes.len() - self.pos < 2 {
                return Err(self.error(DecodeFailure::TruncatedUnit, self.pos, self.bytes.len()));
            }
            self.unit16(le)
        };
        if (0xd800..=0xdbff).contains(&unit) && self.bytes.len() - self.pos >= 2 {
            let next = self.unit16(le);
            if (0xdc00..=0xdfff).contains(&next) {
                return Ok(0x10000 + ((u32::from(unit) - 0xd800) << 10) + (u32::from(next) - 0xdc00));
            }
            self.cached = Some(next);
        }
        // surrogatepass consumes an isolated full unit even before odd tail.
        Ok(u32::from(unit))
    }
    fn utf32(&mut self, le: bool) -> Result<u32, Error> {
        let start = self.pos;
        if self.bytes.len() - start < 4 {
            return Err(self.error(DecodeFailure::TruncatedUnit, start, self.bytes.len()));
        }
        let unit = [self.bytes[start], self.bytes[start + 1], self.bytes[start + 2], self.bytes[start + 3]];
        self.pos += 4;
        let point = if le { u32::from_le_bytes(unit) } else { u32::from_be_bytes(unit) };
        if point > 0x10ffff { return Err(self.error(DecodeFailure::OutOfRange, start, start + 4)); }
        Ok(point)
    }
}

#[cfg(test)]
thread_local! {
    static FACTORY_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PANIC_FACTORY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_AFTER_POINT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PANIC_AFTER_POINT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static REFUSE_POINT_ALLOCATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum ReplayRefusal { PartialResource, PartialPanic, FactoryPanic, Allocation }

/// Thread-scoped fixture, restores even during unwind; no production hook/API.
#[cfg(test)]
pub(crate) struct ReplayRefusalGuard {
    previous: [bool; 4],
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl ReplayRefusalGuard {
    pub(crate) fn arm(mode: ReplayRefusal) -> Self {
        let previous = [
            FAIL_AFTER_POINT.with(|flag| flag.replace(matches!(mode, ReplayRefusal::PartialResource))),
            PANIC_AFTER_POINT.with(|flag| flag.replace(matches!(mode, ReplayRefusal::PartialPanic))),
            PANIC_FACTORY.with(|flag| flag.replace(matches!(mode, ReplayRefusal::FactoryPanic))),
            REFUSE_POINT_ALLOCATION.with(|flag| flag.replace(matches!(mode, ReplayRefusal::Allocation))),
        ];
        Self { previous, _thread: std::marker::PhantomData }
    }
}

#[cfg(test)]
impl Drop for ReplayRefusalGuard {
    fn drop(&mut self) {
        FAIL_AFTER_POINT.with(|flag| flag.set(self.previous[0]));
        PANIC_AFTER_POINT.with(|flag| flag.set(self.previous[1]));
        PANIC_FACTORY.with(|flag| flag.set(self.previous[2]));
        REFUSE_POINT_ALLOCATION.with(|flag| flag.set(self.previous[3]));
    }
}

#[cfg(test)]
fn refuse_point_allocation() -> Result<(), NativeError> {
    if REFUSE_POINT_ALLOCATION.with(|flag| flag.replace(false)) { Err(NativeError::Allocation) }
    else { Ok(()) }
}
#[cfg(test)]
fn factory_entry() {
    FACTORY_ENTRIES.with(|n| n.set(n.get() + 1));
    PANIC_FACTORY.with(|flag| assert!(!flag.replace(false), "source-only codec unwind fixture"));
}
#[cfg(test)]
fn after_point() -> Result<(), NativeError> {
    PANIC_AFTER_POINT.with(|flag| assert!(!flag.replace(false), "source-only partial output unwind"));
    if FAIL_AFTER_POINT.with(|flag| flag.replace(false)) { return Err(NativeError::Resource); }
    Ok(())
}

#[cfg(test)]
#[path = "json_bytes_codec_tests.rs"]
mod tests;
