//! Opt-in, original-counter DecoderPlan import; no source verification.
//!
//! Never call ordinary unbounded JSON/DTO deserialization before reservation.
//! First pass counts immutable source spans without allocation. Second pass and
//! DTO constructors use exact shared buffers. Object backing uses the EXISTING
//! model census's conservative opaque-tree policy, not private B-tree layouts.
//! Qualified for the workspace default scalar-Number/B-tree serde_json closure.
//! Number lexemes are bounded without conversion during scan. After complete
//! reservation, the pinned serde_json parser alone determines kind and rounding.
//! Qualified default/std features have zero numeric scratch backing, NOT zero
//! error backing. float_roundtrip's scratch/bigints and arbitrary_precision's
//! owned strings remain UNQUALIFIED; enabling either requires a new contract.
//! Work counts source peek/read/span events, prepaid complete byte passes,
//! schema/initialization/retirement visits, opaque tree comparison/move allowance,
//! and the existing census's own units. Independent slices are prepaid and never
//! lent/refunded. Pure checked-layout quotation arithmetic/ordinary scalar locals
//! are excluded, as in the underlying resource contracts; this is not CPU fuel.
//! No native/dependency stack, allocator overhead/work, formatting or RSS claim.
//! Incoming byte creation is an upstream gate, NOT retroactively admitted here.

use crate::bounded_index::{layout_bytes, ExactBuffer};
use pllm_garble::shared_allocation::{Error as NativeError, OwnedCharged, OwnedExternalCharge};
use pllm_models::{
    owned_backing::bounded as census, AppliedMethod, DecoderGraph, DecoderMode, DecoderPlan,
    ModelOperation, ModelOperator, StateKind, StateTensor,
};
use pllm_types::{bounded_canonical::borrowed::NativeFrame, Digest};
use serde::{
    de::{self, DeserializeOwned},
    Deserialize,
};
use serde_json::{Map, Number, Value};
use std::{cell::Cell, fmt, mem::size_of};

const DEPTH: usize = 32;
const CENSUS_SLOTS: usize = 17;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_source_bytes: usize,
    pub max_nodes: usize,
    pub max_depth: usize,
    pub max_string_bytes: usize,
    /// Includes sign, decimal point and exponent; checked before any helper.
    pub max_number_bytes: usize,
    /// Aggregate independently prepaid serde numeric-helper work.
    pub max_number_work: u128,
    pub max_work: u128,
    pub max_payload_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Counter,
    OriginalOwner,
    Bootstrap,
    Overflow,
    SourceLimit,
    NodeLimit,
    DepthLimit,
    StringLimit,
    WorkLimit,
    Capacity,
    Syntax,
    Schema,
    DuplicateField,
    UnknownField,
    MissingField,
    UnsupportedNumber,
    NumberLimit,
    NumberWorkLimit,
    InvalidNumber,
    Census,
    Factory,
    Native(NativeError),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded plan input refused")
    }
}
impl std::error::Error for Error {}
impl de::Error for Error {
    fn custom<T: fmt::Display>(_: T) -> Self {
        Self::Schema
    } // never format/allocate
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}
fn wide_add(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn wide_mul(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

#[derive(Clone, Copy, Default, Debug)]
struct Meter {
    used: u128,
    limit: u128,
}
impl Meter {
    fn pay(&mut self, amount: u128) -> Result<(), Error> {
        let next = wide_add(self.used, amount)?;
        if next > self.limit {
            return Err(Error::WorkLimit);
        }
        self.used = next;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Object,
    Array,
    String,
    Number,
    True,
    False,
    Null,
}
#[derive(Clone, Copy)]
struct Span {
    kind: Kind,
    start: usize,
    end: usize,
    next: usize,
    children: usize,
    decoded: usize,
}
impl Span {
    const EMPTY: Self = Self {
        kind: Kind::Null,
        start: 0,
        end: 0,
        next: 0,
        children: 0,
        decoded: 0,
    };
}
#[derive(Clone, Copy)]
struct Frame {
    kind: Kind,
    state: u8,
    span: usize,
    children: usize,
}
impl Frame {
    const EMPTY: Self = Self {
        kind: Kind::Null,
        state: 0,
        span: 0,
        children: 0,
    };
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SourceCounts {
    pub spans: usize,
    pub objects: usize,
    pub array_children: usize,
    pub map_entries: usize,
    pub decoded_string_bytes: usize,
    pub largest_string: usize,
    pub depth: usize,
    pub numbers: usize,
    pub number_bytes: usize,
    map_squares: u128,
    scan_work: u128,
}

// The default policy retains C9's container-count depth semantics and 32 slots.
// Lexical snapshots count VALUE depth from root zero (keys are not values).
// A container at depth 64 needs slot 65 even when it is empty.
struct Scanner<'a, 'm, 's, const SLOTS: usize = DEPTH, const LEXICAL: bool = false> {
    raw: &'a [u8],
    position: usize,
    stack: [Frame; SLOTS],
    active: usize,
    stats: SourceCounts,
    meter: &'m mut Meter,
    output: Option<&'s mut [Span]>,
    limits: Limits,
}
impl<const SLOTS: usize, const LEXICAL: bool> Scanner<'_, '_, '_, SLOTS, LEXICAL> {
    fn peek(&mut self) -> Result<Option<u8>, Error> {
        self.meter.pay(1)?;
        Ok(self.raw.get(self.position).copied())
    }
    fn byte(&mut self) -> Result<u8, Error> {
        self.meter.pay(1)?;
        let byte = *self.raw.get(self.position).ok_or(Error::Syntax)?;
        self.position += 1;
        Ok(byte)
    }
    fn whitespace(&mut self) -> Result<(), Error> {
        while matches!(self.peek()?, Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.byte()?;
        }
        Ok(())
    }
    fn emit(&mut self, mut span: Span) -> Result<usize, Error> {
        self.meter.pay(1)?;
        let index = self.stats.spans;
        self.stats.spans = add(index, 1)?;
        if self.stats.spans > self.limits.max_nodes {
            return Err(Error::NodeLimit);
        }
        span.next = self.stats.spans;
        if let Some(out) = &mut self.output {
            *out.get_mut(index).ok_or(Error::Capacity)? = span;
        }
        Ok(index)
    }
    fn string(&mut self) -> Result<(), Error> {
        let start = self.position;
        if self.byte()? != b'"' {
            return Err(Error::Syntax);
        }
        let mut decoded = 0;
        loop {
            let byte = self.byte()?;
            match byte {
                b'"' => break,
                0..=31 => return Err(Error::Syntax),
                b'\\' => {
                    let escape = self.byte()?;
                    let bytes = match escape {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => 1,
                        b'u' => {
                            let first = self.hex()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                if self.byte()? != b'\\' || self.byte()? != b'u' {
                                    return Err(Error::Syntax);
                                }
                                let second = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return Err(Error::Syntax);
                                }
                                0x10000 + ((u32::from(first) - 0xd800) << 10) + u32::from(second)
                                    - 0xdc00
                            } else {
                                u32::from(first)
                            };
                            char::from_u32(scalar).ok_or(Error::Syntax)?.len_utf8()
                        }
                        _ => return Err(Error::Syntax),
                    };
                    decoded = add(decoded, bytes)?;
                }
                _ => decoded = add(decoded, 1)?,
            }
        }
        if decoded > self.limits.max_string_bytes {
            return Err(Error::StringLimit);
        }
        self.stats.decoded_string_bytes = add(self.stats.decoded_string_bytes, decoded)?;
        self.stats.largest_string = self.stats.largest_string.max(decoded);
        self.emit(Span {
            kind: Kind::String,
            start,
            end: self.position,
            decoded,
            ..Span::EMPTY
        })?;
        Ok(())
    }
    fn hex(&mut self) -> Result<u16, Error> {
        let mut value = 0;
        for _ in 0..4 {
            let byte = self.byte()?;
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(Error::Syntax),
            };
            value = value * 16 + u16::from(digit);
        }
        Ok(value)
    }
    fn value(&mut self) -> Result<(), Error> {
        if LEXICAL {
            if self.active > self.limits.max_depth.min(64) {
                return Err(Error::DepthLimit);
            }
            self.stats.depth = self.stats.depth.max(self.active);
        }
        self.whitespace()?;
        let start = self.position;
        let byte = self.peek()?.ok_or(Error::Syntax)?;
        if byte == b'"' {
            return self.string();
        }
        if byte == b'[' || byte == b'{' {
            self.byte()?;
            let kind = if byte == b'[' {
                Kind::Array
            } else {
                Kind::Object
            };
            let span = self.emit(Span {
                kind,
                start,
                ..Span::EMPTY
            })?;
            if self.active >= SLOTS
                || (!LEXICAL && self.active >= self.limits.max_depth.min(DEPTH))
            {
                return Err(Error::DepthLimit);
            }
            self.stack[self.active] = Frame {
                kind,
                state: 0,
                span,
                children: 0,
            };
            self.active += 1;
            if !LEXICAL {
                self.stats.depth = self.stats.depth.max(self.active);
            }
            if kind == Kind::Object {
                self.stats.objects = add(self.stats.objects, 1)?;
            }
            return Ok(());
        }
        let kind = match byte {
            b'n' => {
                for expected in *b"null" {
                    if self.byte()? != expected {
                        return Err(Error::Syntax);
                    }
                }
                Kind::Null
            }
            b't' => {
                for expected in *b"true" {
                    if self.byte()? != expected {
                        return Err(Error::Syntax);
                    }
                }
                Kind::True
            }
            b'f' => {
                for expected in *b"false" {
                    if self.byte()? != expected {
                        return Err(Error::Syntax);
                    }
                }
                Kind::False
            }
            b'-' | b'0'..=b'9' => {
                self.number()?;
                Kind::Number
            }
            _ => return Err(Error::Syntax),
        };
        self.emit(Span {
            kind,
            start,
            end: self.position,
            ..Span::EMPTY
        })?;
        Ok(())
    }
    fn number(&mut self) -> Result<(), Error> {
        let start = self.position;
        if self.peek()? == Some(b'-') {
            self.byte()?;
        }
        match self.byte()? {
            b'0' => {
                if matches!(self.peek()?, Some(b'0'..=b'9')) {
                    return Err(Error::Syntax);
                }
            }
            b'1'..=b'9' => {
                while matches!(self.peek()?, Some(b'0'..=b'9')) {
                    self.byte()?;
                }
            }
            _ => return Err(Error::Syntax),
        }
        if self.peek()? == Some(b'.') {
            self.byte()?;
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(Error::Syntax);
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.byte()?;
            }
        }
        if matches!(self.peek()?, Some(b'e' | b'E')) {
            self.byte()?;
            if matches!(self.peek()?, Some(b'+' | b'-')) {
                self.byte()?;
            }
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(Error::Syntax);
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.byte()?;
            }
        }
        let length = self.position - start;
        if length > self.limits.max_number_bytes {
            return Err(Error::NumberLimit);
        }
        // No Number/serde/helper construction in this allocation-free pass.
        self.stats.numbers = add(self.stats.numbers, 1)?;
        self.stats.number_bytes = add(self.stats.number_bytes, length)?;
        Ok(())
    }
    fn run(mut self) -> Result<SourceCounts, Error> {
        let before = self.meter.used;
        self.value()?;
        while self.active != 0 {
            self.whitespace()?;
            let top = self.active - 1;
            let frame = self.stack[top];
            let next = self.peek()?.ok_or(Error::Syntax)?;
            let closing = if frame.kind == Kind::Array {
                b']'
            } else {
                b'}'
            };
            if next == closing && (frame.state == 0 || frame.state == 1) {
                self.byte()?;
                if let Some(output) = &mut self.output {
                    let span = &mut output[frame.span];
                    span.end = self.position;
                    span.next = self.stats.spans;
                    span.children = frame.children;
                }
                if frame.kind == Kind::Object {
                    self.stats.map_squares = wide_add(
                        self.stats.map_squares,
                        wide_mul(frame.children as u128, frame.children as u128)?,
                    )?;
                }
                self.active -= 1;
                continue;
            }
            if frame.state == 1 {
                if self.byte()? != b',' {
                    return Err(Error::Syntax);
                }
                self.stack[top].state = 2;
                continue;
            }
            if frame.kind == Kind::Object {
                self.string()?;
                self.whitespace()?;
                if self.byte()? != b':' {
                    return Err(Error::Syntax);
                }
                self.stats.map_entries = add(self.stats.map_entries, 1)?;
            } else {
                self.stats.array_children = add(self.stats.array_children, 1)?;
            }
            self.stack[top].state = 1;
            self.stack[top].children = add(frame.children, 1)?;
            self.value()?;
        }
        self.whitespace()?;
        if self.position != self.raw.len() {
            return Err(Error::Syntax);
        }
        self.stats.scan_work = self.meter.used - before;
        Ok(self.stats)
    }
}
impl<const SLOTS: usize, const LEXICAL: bool> Drop for Scanner<'_, '_, '_, SLOTS, LEXICAL> {
    fn drop(&mut self) {
        for frame in &mut self.stack {
            *frame = Frame::EMPTY;
        }
    }
}

struct CensusPool<'a>([NativeFrame<'a>; CENSUS_SLOTS]);
impl Drop for CensusPool<'_> {
    fn drop(&mut self) {
        for frame in &mut self.0 {
            *frame = NativeFrame::empty();
        }
    }
}

// Pinned serde_json 1.0.151 de.rs:509-671,714-738,834-873;
// number.rs:398-459; error.rs:230-311,315-318. Actual compiled fingerprints in
// target/debug identify default,std only. No string/ignored-value/I/O parser is
// used. With this closure numeric success never grows Deserializer::scratch.
// Alternate features are not silently priced as zero:
// - float_roundtrip: de.rs:674-831 scratch plus lexical/{bignum,math}.rs Vecs;
// - arbitrary_precision: de.rs:937-1050 growing String and owned Number payload.
// This qualification follows existing model/JSON census feature policy; it is
// not runtime feature detection or permission to enable alternate features.

// zmij 1.0.23 lib.rs BUFFER_SIZE=24. This signed finite boundary spelling has
// that same length and bounds CompactFormatter's complete f64 output. It is an
// output/work bound, NOT a parser, approximation or changed numeric format.
const NUMBER_DISPLAY_BYTES: usize = b"-1.7976931348623157e+308".len();
// Default serde's long-integer/fraction counters are i32. Bound their combined
// lengths before helper entry, rather than relying on wrap/overflow behavior.
const SAFE_NUMBER_BYTES: usize = (i32::MAX as usize) / 2;

#[derive(Clone, Copy)]
struct NumberResources {
    parser: usize,
    input_transition: usize,
    result_transitions: usize,
    value_transition: usize,
    error_backing_policy: usize,
    scratch_backing: usize,
}
impl NumberResources {
    fn total(self) -> Result<usize, Error> {
        [
            self.parser,
            self.input_transition,
            self.result_transitions,
            self.value_transition,
            self.error_backing_policy,
            self.scratch_backing,
            size_of::<Self>(),
        ]
        .into_iter()
        .try_fold(0, add)
    }
}
fn number_resources() -> Result<NumberResources, Error> {
    // Conservative PINNED error-layout policy, not sizeof(Error)'s pointer and
    // not a claim of observable private layout. ErrorImpl = code,line,column.
    // ErrorCode has only unit variants plus Message(Box<str>) and Io(io::Error).
    // Sum BOTH payload alternatives plus a usize tag and both position fields;
    // this overbounds the qualified enum/struct layout incl alignment. Numeric
    // SliceRead errors are unit syntax codes: no Message/Io pointee is created.
    // Price old/new ErrorImpl overlap for fix_position as well. Never format it.
    let error = std::alloc::Layout::new::<(usize, Box<str>, std::io::Error, usize, usize)>().size();
    Ok(NumberResources {
        parser: size_of::<serde_json::Deserializer<serde_json::de::SliceRead<'_>>>(),
        input_transition: size_of::<serde_json::de::SliceRead<'_>>() + size_of::<&[u8]>(),
        result_transitions: mul(size_of::<Result<Number, serde_json::Error>>(), 2)?,
        value_transition: size_of::<(Number, u64, i64, f64, usize)>()
            + size_of::<Result<Number, Error>>(),
        error_backing_policy: mul(error, 2)?,
        scratch_backing: 0, // ONLY qualified default/std numeric path above
    })
}
fn number_work(bytes: usize, calls: usize) -> Result<u128, Error> {
    // Six byte-event passes cover peek/consume/arithmetic dispatch and error
    // SliceRead position scans. Float scaling terminates on finite IEEE zero;
    // this full exponent-range+mantissa bound exceeds all default POW10-loop
    // visits (each missing-power negative iteration divides by 1e308).
    let scaling = (f64::MAX_EXP - f64::MIN_EXP) as u128 + u128::from(f64::MANTISSA_DIGITS);
    let dispatch = 1 + 1 + 1 + 1; // entry, Number visitor, end check, return
    wide_add(
        wide_mul(bytes as u128, 6)?,
        wide_mul(calls as u128, scaling + dispatch)?,
    )
}
fn numeric_work(stats: SourceCounts, limits: Limits) -> Result<u128, Error> {
    let total = number_work(stats.number_bytes, stats.numbers)?;
    if total > limits.max_number_work {
        return Err(Error::NumberWorkLimit);
    }
    Ok(total)
}

fn factory() -> Result<(), Error> {
    #[cfg(test)]
    FACTORIES.with(|state| {
        let (count, fail, panic) = state.get();
        let count = count + 1;
        state.set((count, fail, panic));
        assert_ne!(panic, Some(count), "injected plan input factory panic");
        if fail == Some(count) {
            return Err(Error::Factory);
        }
        Ok(())
    })?;
    Ok(())
}
fn exact<T>(count: usize) -> Result<ExactBuffer<T>, Error> {
    layout_bytes::<T>(count).map_err(|_| Error::Overflow)?;
    factory()?;
    ExactBuffer::new(count).map_err(|_| Error::Capacity)
}
fn push<T>(buffer: &mut ExactBuffer<T>, value: T) -> Result<(), Error> {
    buffer.push(value).map_err(|_| Error::Capacity)
}
fn finish<T>(buffer: ExactBuffer<T>) -> Result<Vec<T>, Error> {
    buffer
        .finish()
        .map(|b| b.into_vec())
        .map_err(|_| Error::Capacity)
}

struct Builder<'a> {
    raw: &'a [u8],
    spans: &'a [Span],
    numeric: Cell<Meter>,
}
impl Builder<'_> {
    fn span(&self, n: usize) -> Result<Span, Error> {
        self.spans.get(n).copied().ok_or(Error::Schema)
    }
    fn string(&self, n: usize) -> Result<String, Error> {
        let span = self.span(n)?;
        if span.kind != Kind::String {
            return Err(Error::Schema);
        }
        let mut output = exact::<u8>(span.decoded)?;
        let mut position = span.start + 1;
        while position < span.end - 1 {
            let byte = self.raw[position];
            position += 1;
            if byte != b'\\' {
                push(&mut output, byte)?;
                continue;
            }
            let escaped = self.raw[position];
            position += 1;
            let byte = match escaped {
                b'"' => b'"',
                b'\\' => b'\\',
                b'/' => b'/',
                b'b' => 8,
                b'f' => 12,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'u' => {
                    let first = decode_hex(self.raw, &mut position)?;
                    let scalar = if (0xd800..=0xdbff).contains(&first) {
                        position += 2; // immutable first pass proved second \\u
                        let second = decode_hex(self.raw, &mut position)?;
                        0x10000 + ((u32::from(first) - 0xd800) << 10) + u32::from(second) - 0xdc00
                    } else {
                        u32::from(first)
                    };
                    let character = char::from_u32(scalar).ok_or(Error::Syntax)?;
                    let mut utf8 = [0; 4];
                    for byte in character.encode_utf8(&mut utf8).bytes() {
                        push(&mut output, byte)?;
                    }
                    continue;
                }
                _ => return Err(Error::Syntax),
            };
            push(&mut output, byte)?;
        }
        String::from_utf8(finish(output)?).map_err(|_| Error::Syntax)
    }
    fn unsigned(&self, n: usize) -> Result<u64, Error> {
        self.number(n)?.as_u64().ok_or(Error::Schema)
    }
    fn number(&self, n: usize) -> Result<Number, Error> {
        let span = self.span(n)?;
        if span.kind != Kind::Number {
            return Err(Error::Schema);
        }
        let raw = &self.raw[span.start..span.end];
        // Debit ONLY the disjoint prepaid numeric slice, before helper/factory.
        let mut meter = self.numeric.get();
        meter.pay(number_work(raw.len(), 1)?)?;
        self.numeric.set(meter);
        factory()?;
        #[cfg(test)]
        NUMBER_CALLS.with(|calls| calls.set(calls.get() + 1));
        // Full original working guard is held by try_build through this call,
        // through ErrorImpl retirement, and through all initialized DTO cleanup.
        serde_json::from_slice::<Number>(raw).map_err(|_| Error::InvalidNumber)
    }
    fn optional_unsigned(&self, n: usize) -> Result<Option<u64>, Error> {
        if self.span(n)?.kind == Kind::Null {
            Ok(None)
        } else {
            self.unsigned(n).map(Some)
        }
    }
    fn boolean(&self, n: usize) -> Result<bool, Error> {
        match self.span(n)?.kind {
            Kind::True => Ok(true),
            Kind::False => Ok(false),
            _ => Err(Error::Schema),
        }
    }
    fn enumeration<T: DeserializeOwned>(&self, n: usize) -> Result<T, Error> {
        // serde_json also accepts a one-entry externally tagged unit enum map.
        let span = self.span(n)?;
        let name = if span.kind == Kind::Object && span.children == 1 {
            let key = n + 1;
            let value = self.span(key)?.next;
            if self.span(value)?.kind != Kind::Null {
                return Err(Error::Schema);
            }
            key
        } else {
            n
        };
        let text = self.string(name)?;
        T::deserialize(de::value::StrDeserializer::<Error>::new(&text))
    }
    fn optional_state(&self, n: usize) -> Result<Option<StateKind>, Error> {
        if self.span(n)?.kind == Kind::Null {
            Ok(None)
        } else {
            self.enumeration(n).map(Some)
        }
    }
    fn digest(&self, n: usize) -> Result<Digest, Error> {
        Digest::deserialize(de::value::StringDeserializer::<Error>::new(self.string(n)?))
    }
    fn array<T>(
        &self,
        n: usize,
        build: impl Fn(&Self, usize) -> Result<T, Error>,
    ) -> Result<Vec<T>, Error> {
        let span = self.span(n)?;
        if span.kind != Kind::Array {
            return Err(Error::Schema);
        }
        let mut values = exact::<T>(span.children)?;
        let mut current = n + 1;
        while current < span.next {
            push(&mut values, build(self, current)?)?;
            current = self.span(current)?.next;
        }
        finish(values)
    }
    fn strings(&self, n: usize) -> Result<Vec<String>, Error> {
        self.array(n, Self::string)
    }
    fn dimensions(&self, n: usize) -> Result<Vec<u64>, Error> {
        self.array(n, Self::unsigned)
    }
    fn operations(&self, n: usize) -> Result<Vec<ModelOperation>, Error> {
        self.array(n, Self::operation)
    }
    fn states(&self, n: usize) -> Result<Vec<StateTensor>, Error> {
        self.array(n, Self::state)
    }
    fn methods(&self, n: usize) -> Result<Vec<AppliedMethod>, Error> {
        self.array(n, Self::method)
    }
    fn json(&self, n: usize) -> Result<Value, Error> {
        let span = self.span(n)?;
        Ok(match span.kind {
            Kind::Null => Value::Null,
            Kind::True => Value::Bool(true),
            Kind::False => Value::Bool(false),
            Kind::String => Value::String(self.string(n)?),
            Kind::Number => Value::Number(self.number(n)?),
            Kind::Array => Value::Array(self.array(n, Self::json)?),
            Kind::Object => {
                let mut object = Map::new();
                let mut current = n + 1;
                while current < span.next {
                    let key = self.string(current)?;
                    let value = self.span(current)?.next;
                    let item = self.json(value)?;
                    factory()?; // composite real map-insertion allocation site
                    object.insert(key, item); // ordinary last-value-wins duplicate semantics
                    current = self.span(value)?.next;
                }
                Value::Object(object)
            }
        })
    }
}

fn decode_hex(raw: &[u8], position: &mut usize) -> Result<u16, Error> {
    let mut value = 0;
    for _ in 0..4 {
        let byte = *raw.get(*position).ok_or(Error::Syntax)?;
        *position += 1;
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return Err(Error::Syntax),
        };
        value = value * 16 + u16::from(digit);
    }
    Ok(value)
}

macro_rules! field {
    (required, $value:expr) => {
        $value.ok_or(Error::MissingField)?
    };
    (optional, $value:expr) => {
        $value.unwrap_or(None)
    };
}
// Concrete partial owners, not a second model DTO or alternate public schema.
// Literals construct the genuine model type: model field additions fail compile.
macro_rules! dto {
    ($method:ident, $ty:ident, $partial:ident, { $( $field:ident : $type:ty => $build:ident / $kind:ident ),* $(,)? }) => {
        #[derive(Default)] struct $partial { $( $field: Option<$type>, )* }
        impl Builder<'_> {
            fn $method(&self, n: usize) -> Result<$ty, Error> {
                let span = self.span(n)?;
                if span.kind != Kind::Object { return Err(Error::Schema); }
                let mut fields = $partial::default(); let mut current = n+1;
                while current < span.next {
                    let key = self.string(current)?;
                    let value = self.span(current)?.next;
                    match key.as_str() { $( stringify!($field) => {
                        if fields.$field.is_some() { return Err(Error::DuplicateField); }
                        fields.$field = Some(self.$build(value)?);
                    }, )* _ => return Err(Error::UnknownField) }
                    current = self.span(value)?.next;
                }
                Ok($ty { $( $field: field!($kind, fields.$field), )* })
            }
        }
    }
}
dto!(plan, DecoderPlan, PartialPlan, {
    schema_version: String => string / required, model_family: String => string / required,
    adapter: String => string / required, config_digest: Digest => digest / required,
    transformations: Vec<AppliedMethod> => methods / required,
    prefill: DecoderGraph => graph / required, decode: DecoderGraph => graph / required,
    token_feedback: bool => boolean / required,
});
dto!(graph, DecoderGraph, PartialGraph, {
    mode: DecoderMode => enumeration / required, batch: u64 => unsigned / required,
    query_sequence: u64 => unsigned / required, maximum_key_sequence: u64 => unsigned / required,
    operations: Vec<ModelOperation> => operations / required,
    state_inputs: Vec<StateTensor> => states / required, state_outputs: Vec<StateTensor> => states / required,
    output: String => string / required,
});
dto!(operation, ModelOperation, PartialOperation, {
    id: String => string / required, operator: ModelOperator => enumeration / required,
    layer: Option<u64> => optional_unsigned / optional, state_kind: Option<StateKind> => optional_state / optional,
    inputs: Vec<String> => strings / required, output_shape: Vec<u64> => dimensions / required,
    attributes: Value => json / required,
});
dto!(state, StateTensor, PartialState, {
    id: String => string / required, layer: Option<u64> => optional_unsigned / optional,
    kind: StateKind => enumeration / required, shape: Vec<u64> => dimensions / required,
    maximum_sequence: u64 => unsigned / required,
});
dto!(method, AppliedMethod, PartialMethod, {
    component: String => string / required, implementation: String => string / required,
    method_id: String => string / required, input_digest: Digest => digest / required,
    configuration_digest: Digest => digest / required,
});

/// Actual source/return/bootstrap controls; numeric layout is NOT reservation.
/// Check under a real working charge BEFORE making Scanner/frame pools.
pub fn bootstrap_bytes() -> Result<usize, Error> {
    let resources = census::resources_for_slots(CENSUS_SLOTS).map_err(|_| Error::Overflow)?;
    let terms = [
        size_of::<Scanner<'_, '_, '_>>(),
        resources.logical_workspace_bytes,
        size_of::<ImportQuote<'_>>(),
        size_of::<Construct<'_>>(),
        size_of::<Limits>(),
        size_of::<SourceCounts>(),
        size_of::<Meter>(),
        size_of::<Builder<'_>>(),
        size_of::<Result<OwnedCharged<DecoderPlan>, NativeError>>(),
        size_of::<Result<ImportQuote<'_>, Error>>(),
        size_of::<Result<DecoderPlan, Error>>(),
        size_of::<Option<Error>>(),
        size_of::<OwnedCharged<DecoderPlan>>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<[u8; 4]>(),
        size_of::<ExactBuffer<Span>>(),
        size_of::<Vec<Span>>(),
        size_of::<Result<Vec<Span>, Error>>(),
        size_of::<Result<SourceCounts, Error>>(),
        size_of::<Result<&str, std::str::Utf8Error>>(),
        size_of::<census::Resources>(),
        size_of::<NumberResources>(),
    ];
    terms.into_iter().try_fold(0, add)
}

fn partial_frame_slots() -> Result<usize, Error> {
    // One complete set of concrete partial/helper/result slots along Builder's
    // ACTIVE source path. Completed siblings live in separately priced array/
    // map/string backings, never in additional active partial frames. This is
    // logical Rust control ownership, NOT dependency/native stack accounting.
    let terms = [
        size_of::<PartialPlan>(),
        size_of::<PartialGraph>(),
        size_of::<PartialOperation>(),
        size_of::<PartialState>(),
        size_of::<PartialMethod>(),
        size_of::<ExactBuffer<String>>(),
        size_of::<ExactBuffer<u64>>(),
        size_of::<ExactBuffer<ModelOperation>>(),
        size_of::<ExactBuffer<StateTensor>>(),
        size_of::<ExactBuffer<AppliedMethod>>(),
        size_of::<Result<DecoderPlan, Error>>(),
        size_of::<Result<DecoderGraph, Error>>(),
        size_of::<Result<ModelOperation, Error>>(),
        size_of::<Result<StateTensor, Error>>(),
        size_of::<Result<AppliedMethod, Error>>(),
        size_of::<Result<Vec<String>, Error>>(),
        size_of::<Result<Vec<u64>, Error>>(),
        size_of::<Result<Vec<ModelOperation>, Error>>(),
        size_of::<Result<Vec<StateTensor>, Error>>(),
        size_of::<Result<Vec<AppliedMethod>, Error>>(),
        size_of::<de::value::StrDeserializer<'_, Error>>(),
        size_of::<de::value::StringDeserializer<Error>>(),
    ];
    terms.into_iter().try_fold(json_frame_slots()?, add)
}

// Model-neutral controls of the SAME Builder::json/string/array implementation.
fn json_frame_slots() -> Result<usize, Error> {
    [
        size_of::<ExactBuffer<Value>>(),
        size_of::<ExactBuffer<u8>>(),
        size_of::<String>(),
        size_of::<Map<String, Value>>(),
        size_of::<Value>(),
        size_of::<Result<String, Error>>(),
        size_of::<Result<Value, Error>>(),
        size_of::<Result<Vec<Value>, Error>>(),
        size_of::<std::string::FromUtf8Error>(),
        size_of::<std::str::Bytes<'_>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<(&Builder<'_>, usize)>(),
        size_of::<(usize, Span, Number)>(),
        size_of::<[u8; 4]>(),
    ]
    .into_iter()
    .try_fold(0, add)
}

fn active_frames(stats: SourceCounts, slots: usize) -> Result<usize, Error> {
    // Builder recursively descends only into child source spans. Array dispatch
    // and its typed wrapper share one ExactBuffer; string/enum/numeric helpers
    // are leaf calls whose concrete slots are all in partial_frame_slots.
    // Container depth plus leaf and outer-return slots bounds the active path.
    // Two complete sets at EACH level cover origin/destination result moves;
    // no backing retirement/reuse credit is taken anywhere in this quote.
    mul(mul(add(stats.depth, 2)?, 2)?, slots)
}

fn partial_frames(stats: SourceCounts) -> Result<usize, Error> {
    active_frames(stats, partial_frame_slots()?)
}

fn source_backing(stats: SourceCounts, array_slot: usize) -> Result<usize, Error> {
    let trees = mul(
        add(stats.objects, stats.map_entries)?,
        pllm_models::owned_backing::checked_tree_node_bytes::<(String, Value)>()
            .map_err(|_| Error::Overflow)?,
    )?;
    [
        mul(stats.array_children, array_slot)?,
        stats.decoded_string_bytes,
        trees,
    ]
    .into_iter()
    .try_fold(0, add)
}

fn payload(stats: SourceCounts) -> Result<usize, Error> {
    // Each array has exactly ONE of these element types. Summing all is a
    // schema-bound upper bound, including every overwritten duplicate subtree.
    let array_slot = [
        size_of::<String>(),
        size_of::<u64>(),
        size_of::<ModelOperation>(),
        size_of::<StateTensor>(),
        size_of::<AppliedMethod>(),
        size_of::<Value>(),
    ]
    .into_iter()
    .try_fold(0, add)?;
    let terms = [
        bootstrap_bytes()?,
        layout_bytes::<Span>(stats.spans).map_err(|_| Error::Overflow)?,
        source_backing(stats, array_slot)?,
        partial_frames(stats)?,
        number_resources()?.total()?,
    ];
    terms.into_iter().try_fold(0, add)
}

fn build_work(stats: SourceCounts, raw_bytes: usize) -> Result<u128, Error> {
    // Replay measured scan, span initialization/retirement, string
    // unescape/copy/UTF8 passes and byte-to-static schema matches. Numeric helper
    // work is a separate disjoint slice, never part of this free allowance.
    // Fixed schema match alternatives: every field of all five concrete DTOs.
    let fields = 8u128 + 8 + 7 + 5 + 5;
    shared_build_work(stats, raw_bytes, fields)
}

fn shared_build_work(
    stats: SourceCounts,
    raw_bytes: usize,
    string_and_dispatch_passes: u128,
) -> Result<u128, Error> {
    let fields = string_and_dispatch_passes;
    let strings = wide_mul(stats.decoded_string_bytes as u128, fields)?;
    let source_passes = wide_mul(raw_bytes as u128, 3)?;
    let span_visits = wide_mul(stats.spans as u128, fields + 2 + 2 + 1)?;
    let array_visits = wide_mul(stats.array_children as u128, 2)?;
    // Comparison/move allowance for opaque object insertion. Sum per-object
    // squared entry counts, NOT whole-document quadratic work. String keys only;
    // no user comparator or arbitrary serde/custom Deserialize is invoked.
    let key_comparisons = wide_mul(
        stats.map_squares,
        wide_add(stats.largest_string as u128, 1)?,
    )?;
    let entry_and_separator_moves = wide_add(stats.map_squares, stats.map_squares)?;
    let tree = wide_add(key_comparisons, entry_and_separator_moves)?;
    [
        stats.scan_work,
        strings,
        source_passes,
        span_visits,
        array_visits,
        tree,
        (2 * DEPTH + 2 * CENSUS_SLOTS) as u128,
    ]
    .into_iter()
    .try_fold(0, wide_add)
}

fn census_work(stats: SourceCounts) -> Result<usize, Error> {
    // bounded::Census: dispatch + transformations header + two graph/operation/
    // two-state headers. Per object charge EVERY possible method, operation
    // (element/input-header/JSON-header), state, and native root-reset owner.
    let headers = 1u128 + 1 + 2 * (1 + 1 + 2);
    let dto = wide_add(
        wide_mul(stats.objects as u128, 1 + 3 + 1 + 1)?,
        stats.array_children as u128,
    )?;
    // Native cursor: enter+exit per value, array edges, object selection /
    // candidate / pending-child per entry. Qualified integer/f64 Number display
    // is bounded by the pinned CompactFormatter/zmij output above.
    usize::try_from(wide_add(headers, wide_add(dto, json_cursor_work(stats)?)?)?)
        .map_err(|_| Error::Overflow)
}

fn json_cursor_work(stats: SourceCounts) -> Result<u128, Error> {
    wide_add(
        wide_mul(stats.spans as u128, 2 + NUMBER_DISPLAY_BYTES as u128)?,
        wide_add(
            stats.array_children as u128,
            wide_mul(stats.map_entries as u128, 3)?,
        )?,
    )
}

/// Quote borrows the ORIGINAL immutable byte owner and owns the ORIGINAL working
/// charge. No surrogate capacity baseline or caller-provided DTO is accepted.
/// No mutable-exporter/raw-Python importer or source-verification authority.
///
/// Actual public-type lifetime restriction (not a privacy failure):
/// ```compile_fail
/// use pllm_compiler::plan_input::{quote, ImportQuote, Limits};
/// use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged};
/// fn escape<'a>(root: &OwnedBudget, limits: Limits) -> ImportQuote<'a> {
///     let owner: OwnedCharged<Vec<u8>> = root.charge_external(2).unwrap()
///         .try_build(|_| Ok(b"{}".to_vec())).unwrap();
///     let work = root.charge_external(pllm_compiler::plan_input::bootstrap_bytes().unwrap()).unwrap();
///     quote(&owner, work, limits).unwrap() // cannot return reference to local owner
/// }
/// ```
pub struct ImportQuote<'a> {
    original: &'a OwnedCharged<Vec<u8>>,
    charge: OwnedExternalCharge,
    limits: Limits,
    counts: SourceCounts,
    meter: Meter,
    required: usize,
}
impl ImportQuote<'_> {
    pub fn counts(&self) -> SourceCounts {
        self.counts
    }
    /// Required working payload before try_build's additional return-inline
    /// charge. Complete construction reservation is retained on success.
    pub fn required_working_payload(&self) -> usize {
        self.required
    }
    pub fn original_capacity_bytes(&self) -> usize {
        self.original.payload().capacity()
    }
    /// Includes already-retained excess working reservation and try_build's
    /// actual inline return/padding. Original byte owner remains separately live.
    pub fn working_charge_bytes(&self) -> Result<usize, Error> {
        add(
            self.required.max(self.charge.payload_bytes()),
            size_of::<OwnedCharged<DecoderPlan>>(),
        )
    }
    /// These two original owner charges only; enclosing root/siblings already
    /// live on the same counter and must not be replaced by a guessed baseline.
    pub fn original_owners_peak_bytes(&self) -> Result<usize, Error> {
        add(self.original.charged_bytes(), self.working_charge_bytes()?)
    }
    pub fn work_used(&self) -> u128 {
        self.meter.used
    }
    pub fn total_work_bound(&self) -> Result<u128, Error> {
        wide_add(
            self.meter.used,
            wide_add(
                build_work(self.counts, self.original.payload().len())?,
                wide_add(
                    census_work(self.counts)? as u128,
                    numeric_work(self.counts, self.limits)?,
                )?,
            )?,
        )
    }

    /// All growth on original counter precedes factories. Partial initialized
    /// owners die inside try_build BEFORE its charge refunds, including unwind.
    /// Returned fixed errors contain no source borrow/String/hidden allocation.
    /// Caller still owns original charged bytes throughout error retirement.
    pub fn construct(self) -> Result<OwnedCharged<DecoderPlan>, Error> {
        let Self {
            original,
            mut charge,
            limits,
            counts,
            mut meter,
            required,
        } = self;
        let construction_work = wide_add(
            build_work(counts, original.payload().len())?,
            wide_add(census_work(counts)? as u128, numeric_work(counts, limits)?)?,
        )?;
        meter.pay(construction_work)?; // disjoint prepaid construction allowance
        charge
            .resize_payload(charge.payload_bytes().max(required))
            .map_err(Error::Native)?;
        let mut failure = None;
        let state = Construct {
            original,
            limits,
            counts,
            meter,
        };
        let result = charge.try_build(|_| match state.run() {
            Ok(plan) => Ok(plan),
            Err(error) => {
                failure = Some(error);
                Err(NativeError::Resource)
            }
        });
        match result {
            Ok(owner) => Ok(owner),
            Err(native) => Err(failure.unwrap_or(Error::Native(native))),
        }
    }
}

pub fn quote(
    original: &OwnedCharged<Vec<u8>>,
    charge: OwnedExternalCharge,
    limits: Limits,
) -> Result<ImportQuote<'_>, Error> {
    check_original(original, &charge)?;
    let (counts, meter) = scan_original(original, limits, bootstrap_bytes()?, &charge)?;
    let raw = original.payload();
    let required = payload(counts)?;
    if required > limits.max_payload_bytes {
        return Err(Error::Capacity);
    }
    // Refuse full preconstruction work without entering any real allocation.
    let mut check = meter;
    check.pay(wide_add(
        build_work(counts, raw.len())?,
        wide_add(census_work(counts)? as u128, numeric_work(counts, limits)?)?,
    )?)?;
    Ok(ImportQuote {
        original,
        charge,
        limits,
        counts,
        meter,
        required,
    })
}

fn check_original(
    original: &OwnedCharged<Vec<u8>>,
    charge: &OwnedExternalCharge,
) -> Result<(), Error> {
    if !charge.is_for(original.budget()) {
        return Err(Error::Counter);
    }
    let incoming = add(
        size_of::<OwnedCharged<Vec<u8>>>(),
        original.payload().capacity(),
    )?;
    if original.charged_bytes() < incoming {
        return Err(Error::OriginalOwner);
    }
    Ok(())
}

fn scan_original(
    original: &OwnedCharged<Vec<u8>>,
    limits: Limits,
    bootstrap: usize,
    charge: &OwnedExternalCharge,
) -> Result<(SourceCounts, Meter), Error> {
    scan_original_policy::<DEPTH, false>(original, limits, bootstrap, charge)
}

fn scan_original_policy<const SLOTS: usize, const LEXICAL: bool>(
    original: &OwnedCharged<Vec<u8>>,
    limits: Limits,
    bootstrap: usize,
    charge: &OwnedExternalCharge,
) -> Result<(SourceCounts, Meter), Error> {
    if charge.payload_bytes() < bootstrap {
        return Err(Error::Bootstrap);
    }
    if bootstrap > limits.max_payload_bytes {
        return Err(Error::Capacity);
    }
    if !LEXICAL && limits.max_number_bytes > SAFE_NUMBER_BYTES {
        return Err(Error::NumberLimit);
    }
    let raw = original.payload();
    if raw.len() > limits.max_source_bytes {
        return Err(Error::SourceLimit);
    }
    let mut meter = Meter {
        used: 0,
        limit: limits.max_work,
    };
    meter.pay((2 * SLOTS) as u128)?; // fixed scanner initialization AND retirement
    meter.pay(raw.len() as u128)?; // complete UTF8 pass, BEFORE call
    std::str::from_utf8(raw).map_err(|_| Error::Syntax)?;
    let counts = Scanner::<SLOTS, LEXICAL> {
        raw,
        position: 0,
        stack: [Frame::EMPTY; SLOTS],
        active: 0,
        stats: SourceCounts::default(),
        meter: &mut meter,
        output: None,
        limits,
    }
    .run()?;
    Ok((counts, meter))
}

struct Construct<'a> {
    original: &'a OwnedCharged<Vec<u8>>,
    limits: Limits,
    counts: SourceCounts,
    meter: Meter,
}
impl Construct<'_> {
    fn run(self) -> Result<DecoderPlan, Error> {
        let spans = replay(self.original, self.counts, self.limits)?;
        let builder = Builder {
            raw: self.original.payload(),
            spans: &spans,
            numeric: Cell::new(Meter {
                used: 0,
                limit: numeric_work(self.counts, self.limits)?,
            }),
        };
        let plan = builder.plan(0)?;
        // Genuine iterative census, no recursive or caller-baseline fallback.
        // Construction/cleanup and census allowances are already debited and
        // disjoint. This dependency enforces its own prepaid slice each step.
        let resources = census::resources_for_slots(CENSUS_SLOTS).map_err(|_| Error::Census)?;
        let prepaid_census = census_work(self.counts)?;
        let limits = census::Limits {
            max_backing_bytes: self.limits.max_payload_bytes,
            max_work: prepaid_census,
            max_workspace_bytes: resources.logical_workspace_bytes,
            max_json_nodes: self.limits.max_nodes,
            max_json_payload_bytes: add(
                self.counts.decoded_string_bytes,
                mul(self.counts.spans, NUMBER_DISPLAY_BYTES)?,
            )?,
            max_json_string_bytes: self.limits.max_string_bytes,
            max_json_key_bytes: self.limits.max_string_bytes,
            max_json_depth: 16,
        };
        resources.check(limits).map_err(|_| Error::Census)?;
        let backing = {
            let mut frames = CensusPool([const { NativeFrame::empty() }; CENSUS_SLOTS]);
            let mut session =
                census::Census::new(&mut frames.0, limits).map_err(|_| Error::Census)?;
            let report = session.plan(&plan).map_err(|_| Error::Census)?;
            report.backing_bytes()
        };
        if self.meter.used > self.meter.limit {
            return Err(Error::WorkLimit);
        }
        if backing > payload(self.counts)? {
            return Err(Error::Capacity);
        }
        Ok(plan)
    }
}

// One scanner replay/initialization path for both DTO and arbitrary-Value input.
fn replay(
    original: &OwnedCharged<Vec<u8>>,
    counts: SourceCounts,
    limits: Limits,
) -> Result<Vec<Span>, Error> {
    replay_policy::<DEPTH, false>(original, counts, limits)
}

fn replay_policy<const SLOTS: usize, const LEXICAL: bool>(
    original: &OwnedCharged<Vec<u8>>,
    counts: SourceCounts,
    limits: Limits,
) -> Result<Vec<Span>, Error> {
    let mut buffer = exact::<Span>(counts.spans)?;
    for _ in 0..counts.spans {
        push(&mut buffer, Span::EMPTY)?;
    }
    let mut spans = finish(buffer)?;
    let mut prepaid = Meter {
        used: 0,
        // Same immutable input gives the measured scan exactly. Span init/drop
        // has its own prepaid construction credit, never lent to this walker.
        limit: counts.scan_work,
    };
    Scanner::<SLOTS, LEXICAL> {
        raw: original.payload(),
        position: 0,
        stack: [Frame::EMPTY; SLOTS],
        active: 0,
        stats: SourceCounts::default(),
        meter: &mut prepaid,
        output: Some(&mut spans),
        limits,
    }
    .run()?;
    Ok(spans)
}

#[cfg(test)]
thread_local! {
    static FACTORIES: std::cell::Cell<(usize, Option<usize>, Option<usize>)> = const { std::cell::Cell::new((0,None,None)) };
    static NUMBER_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn allocation_factories_for_test() -> usize {
    FACTORIES.with(|state| state.get().0)
}

#[cfg(test)]
#[path = "plan_input_tests.rs"]
mod tests;

/// Model-neutral immutable JSON import; same scanner, builder and qualifications.
#[path = "json_value_input.rs"]
pub mod json;
