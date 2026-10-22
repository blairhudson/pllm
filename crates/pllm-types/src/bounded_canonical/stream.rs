//! Direct schema events and borrowed JSON fragments under one global budget.
//!
//! Run the identical, immutable schema traversal first against `io::sink()`, call
//! `finish`, then replay against a writer or `DomainHash`. This module does not
//! retain or authenticate the traversal. All failures can leave output partial;
//! after any failure the session is poisoned and cannot be finished successfully.
//!
//! Work units are: one per public event (including `finish`, not constructors or
//! accessors), one per visited fragment value, one per
//! candidate examined during fragment key selection, one per byte pair examined
//! by key comparison (plus one for equal-prefix termination), one per raw string
//! byte emitted, and one per output byte. This is an explicit algorithmic work
//! budget, not a wall-clock or arbitrary external writer/allocator CPU bound.

use super::{Error, Output};
use serde_json::ser::{CompactFormatter, Formatter};
use serde_json::{Number, Value};
use sha2::{Digest as _, Sha256};
use std::{
    cmp::Ordering,
    fmt, io,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
pub struct StreamLimits {
    /// JSON values, including containers; object keys are not nodes.
    pub max_nodes: usize,
    /// Root value is depth one; scalar leaves also contribute to peak depth.
    pub max_depth: usize,
    /// Encoded JSON bytes (not domain bytes).
    pub max_bytes: usize,
    /// Raw UTF-8 bytes per string, including object keys, before escaping.
    pub max_string_bytes: usize,
    /// Additional raw UTF-8 cap on keys, checked before comparisons.
    pub max_key_bytes: usize,
    /// Sum of raw UTF-8 lengths of emitted keys and string values.
    pub max_total_string_bytes: usize,
    /// Shared algorithmic work budget; see the module's precise unit definition.
    pub max_work: usize,
    /// Entire supplied stack slice, including unused slots (not live depth).
    pub max_stack_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamUsage {
    pub nodes: usize,
    pub depth: usize,
    pub bytes: usize,
    pub string_bytes: usize,
    pub work: usize,
}

#[derive(Debug)]
pub enum StreamError {
    NodeLimit,
    DepthLimit,
    ByteLimit,
    StringLimit,
    KeyLimit,
    TotalStringLimit,
    WorkLimit,
    StackLimit,
    StackTooSmall,
    Structure,
    KeyOrder,
    NonFiniteNumber,
    Poisoned,
    Io(io::Error),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => fmt::Display::fmt(error, f),
            error => write!(f, "canonical stream error: {error:?}"),
        }
    }
}
impl std::error::Error for StreamError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Array,
    Object,
}

/// One live container, shared by typed events and iterative fragment traversal.
#[derive(Clone, Copy, Debug)]
pub struct StreamFrame<'a> {
    kind: Kind,
    emitted: bool,
    waiting: bool,
    previous: Option<&'a str>,
    fragment: Option<&'a Value>,
    cursor: usize,
}

impl StreamFrame<'_> {
    pub const fn empty() -> Self {
        Self {
            kind: Kind::Array,
            emitted: false,
            waiting: false,
            previous: None,
            fragment: None,
            cursor: 0,
        }
    }
    pub const fn storage_bytes() -> usize {
        size_of::<Self>()
    }
}

/// Actual reusable traversal scratch, included in Session's concrete size.
/// Outer survives key selection and nested value emission; ordered_key and value
/// run sequentially and share inner. The iterator itself lives here, not in an
/// unquoted `for` loop. Borrowed input pointees are not owned workspace.
struct Transient<'a> {
    outer: StreamFrame<'a>,
    inner: StreamFrame<'a>,
    selected: Option<(&'a str, &'a Value)>,
    fragment_base: usize,
    object_iter: Option<serde_json::map::Iter<'a>>,
}

impl Transient<'_> {
    const fn empty() -> Self {
        Self {
            outer: StreamFrame::empty(),
            inner: StreamFrame::empty(),
            selected: None,
            fragment_base: 0,
            object_iter: None,
        }
    }
}

/// Concrete owned state sizes, not a guessed total process-stack allowance.
/// Session includes its meter and borrowed writer/stack references; stack capacity
/// is separate. Reusable traversal scratch (both frame snapshots, selected entry,
/// fragment base and concrete object iterator) is inside Session, not an extra
/// allowance to add. The entire supplied frame slice is measured separately; an
/// owned DomainHash is separate from its borrowed writer ref. Input pointees,
/// ordinary borrowed aliases, and scalar evaluation locals are excluded.
/// Emitter and format arguments are transient. Dependency formatter,
/// SHA compression, `std` and writer *call stacks* are not measurable by size_of;
/// these accessors deliberately do not claim a complete machine-stack bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resources {
    pub session_bytes: usize,
    pub stack_bytes: usize,
    pub emitter_bytes: usize,
    pub format_arguments_bytes: usize,
    pub compact_formatter_bytes: usize,
    pub escape_buffer_bytes: usize,
    pub numeric_value_bytes: usize,
    pub hash_state_bytes: usize,
    pub hash_result_bytes: usize,
    /// Private numeric-library scratch and compiled call frames have no public
    /// Rust size accessor through serde_json. Never substitute a guessed value.
    pub dependency_call_stack_bytes: Option<usize>,
}

/// Public, allocation-free SHA-256 writer; exact legacy domain separator.
/// Domain hashing happens at construction and is outside Session JSON counters.
/// Finalize only after the associated session has successfully finished.
pub struct DomainHash {
    hash: Sha256,
}

impl DomainHash {
    pub fn new(domain: &str) -> Self {
        let mut hash = Sha256::new();
        hash.update(domain.as_bytes());
        hash.update([0]);
        Self { hash }
    }
    pub fn finish(self) -> [u8; 32] {
        self.hash.finalize().into()
    }
    pub const fn storage_bytes() -> usize {
        size_of::<Self>()
    }
}

impl io::Write for DomainHash {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum MeterFailure {
    Byte,
    Work,
}

struct Meter<'w, W> {
    writer: &'w mut W,
    limits: StreamLimits,
    usage: StreamUsage,
    failure: Option<MeterFailure>,
}

impl<W: io::Write> io::Write for Meter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(total) = self
            .usage
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limits.max_bytes)
        else {
            self.failure = Some(MeterFailure::Byte);
            return Err(io::ErrorKind::InvalidData.into());
        };
        let Some(work) = self
            .usage
            .work
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limits.max_work)
        else {
            self.failure = Some(MeterFailure::Work);
            return Err(io::ErrorKind::InvalidData.into());
        };
        self.writer.write_all(bytes)?;
        self.usage.bytes = total;
        self.usage.work = work;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// No heap, descriptor Vec, recursion, or generic serializer. Keys and fragments
/// must remain borrowed for the stack lifetime. Fixed arrays work as the stack.
pub struct Session<'a, 'w, W> {
    meter: Meter<'w, W>,
    stack: &'w mut [StreamFrame<'a>],
    transient: Transient<'a>,
    depth: usize,
    root: bool,
    failed: bool,
}

impl<'a, 'w, W: io::Write> Session<'a, 'w, W> {
    pub fn new(
        writer: &'w mut W,
        stack: &'w mut [StreamFrame<'a>],
        limits: StreamLimits,
    ) -> Result<Self, StreamError> {
        if stack
            .len()
            .checked_mul(size_of::<StreamFrame<'a>>())
            .is_none_or(|n| n > limits.max_stack_bytes)
        {
            return Err(StreamError::StackLimit);
        }
        Ok(Self {
            meter: Meter {
                writer,
                limits,
                usage: StreamUsage::default(),
                failure: None,
            },
            stack,
            transient: Transient::empty(),
            depth: 0,
            root: false,
            failed: false,
        })
    }

    /// Exact completed counters on success; not a partial-I/O byte receipt after
    /// failure (the underlying writer may have accepted an uncounted prefix).
    pub fn usage(&self) -> StreamUsage {
        self.meter.usage
    }

    pub const fn storage_bytes() -> usize {
        size_of::<Self>()
    }

    /// Diagnostic breakdown only: already included in storage_bytes/session_bytes.
    pub const fn transient_storage_bytes() -> usize {
        size_of::<Transient<'a>>()
    }

    /// Account state before constructing a session or allocating a caller stack.
    pub fn resources_for_slots(slots: usize) -> Result<Resources, StreamError> {
        let stack_bytes = slots
            .checked_mul(size_of::<StreamFrame<'a>>())
            .ok_or(StreamError::StackLimit)?;
        Ok(Self::resource_sizes(stack_bytes))
    }

    fn resource_sizes(stack_bytes: usize) -> Resources {
        Resources {
            session_bytes: size_of::<Self>(),
            stack_bytes,
            emitter_bytes: size_of::<Output<'_, Meter<'_, W>>>(),
            format_arguments_bytes: size_of::<fmt::Arguments<'_>>(),
            compact_formatter_bytes: size_of::<CompactFormatter>(),
            escape_buffer_bytes: size_of::<[u8; 6]>(),
            numeric_value_bytes: size_of::<Number>(),
            hash_state_bytes: DomainHash::storage_bytes(),
            hash_result_bytes: size_of::<[u8; 32]>(),
            dependency_call_stack_bytes: None,
        }
    }

    pub fn resources(&self) -> Resources {
        Self::resource_sizes(size_of_val(self.stack))
    }

    /// Consumes the session; exactly one complete root is required.
    pub fn finish(mut self) -> Result<StreamUsage, StreamError> {
        self.step(|s| {
            if s.root && s.depth == 0 {
                Ok(())
            } else {
                Err(StreamError::Structure)
            }
        })?;
        Ok(self.meter.usage)
    }

    fn step<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, StreamError>,
    ) -> Result<T, StreamError> {
        if self.failed {
            return Err(StreamError::Poisoned);
        }
        // Preserve poison even if a caller catches a writer/formatter unwind.
        // A scalar value may have consumed its root slot before output failed.
        self.failed = true;
        let result = self.work(1).and_then(|()| operation(self));
        self.failed = result.is_err();
        result
    }

    fn work(&mut self, amount: usize) -> Result<(), StreamError> {
        self.meter.usage.work = self
            .meter
            .usage
            .work
            .checked_add(amount)
            .filter(|n| *n <= self.meter.limits.max_work)
            .ok_or(StreamError::WorkLimit)?;
        Ok(())
    }

    fn emit(
        &mut self,
        operation: impl FnOnce(&mut Output<'_, Meter<'w, W>>) -> Result<(), Error>,
    ) -> Result<(), StreamError> {
        let mut output = Output {
            writer: &mut self.meter,
            bytes: 0,
            limit: usize::MAX,
            exceeded: false,
        };
        let result = operation(&mut output);
        match self.meter.failure {
            Some(MeterFailure::Byte) => Err(StreamError::ByteLimit),
            Some(MeterFailure::Work) => Err(StreamError::WorkLimit),
            None => result.map_err(|e| match e {
                Error::Io(e) => StreamError::Io(e),
                _ => StreamError::ByteLimit,
            }),
        }
    }

    fn raw(&mut self, bytes: &[u8]) -> Result<(), StreamError> {
        self.emit(|o| o.raw(bytes))
    }

    fn check_text(&self, text: &str, key: bool) -> Result<(), StreamError> {
        if key && text.len() > self.meter.limits.max_key_bytes {
            return Err(StreamError::KeyLimit);
        }
        if text.len() > self.meter.limits.max_string_bytes {
            return Err(StreamError::StringLimit);
        }
        Ok(())
    }

    fn text(&mut self, text: &str, key: bool) -> Result<(), StreamError> {
        self.check_text(text, key)?;
        self.meter.usage.string_bytes = self
            .meter
            .usage
            .string_bytes
            .checked_add(text.len())
            .filter(|n| *n <= self.meter.limits.max_total_string_bytes)
            .ok_or(StreamError::TotalStringLimit)?;
        self.work(text.len())?;
        self.emit(|o| o.string(text))
    }

    fn compare(&mut self, left: &str, right: &str) -> Result<Ordering, StreamError> {
        // Both lengths checked before any potentially long prefix comparison.
        self.check_text(left, true)?;
        self.check_text(right, true)?;
        for (a, b) in left.bytes().zip(right.bytes()) {
            self.work(1)?;
            match a.cmp(&b) {
                Ordering::Equal => {}
                order => return Ok(order),
            }
        }
        self.work(1)?;
        Ok(left.len().cmp(&right.len()))
    }

    fn value(&mut self) -> Result<(), StreamError> {
        if self.depth == 0 {
            if self.root {
                return Err(StreamError::Structure);
            }
            self.root = true;
        } else {
            self.transient.inner = self.stack[self.depth - 1];
            match self.transient.inner.kind {
                Kind::Object if !self.transient.inner.waiting => {
                    return Err(StreamError::Structure)
                }
                Kind::Object => self.stack[self.depth - 1].waiting = false,
                Kind::Array => {
                    if self.transient.inner.emitted {
                        self.raw(b",")?;
                    }
                    self.stack[self.depth - 1].emitted = true;
                }
            }
        }
        if self.meter.usage.nodes >= self.meter.limits.max_nodes {
            return Err(StreamError::NodeLimit);
        }
        let depth = self.depth.checked_add(1).ok_or(StreamError::DepthLimit)?;
        if depth > self.meter.limits.max_depth {
            return Err(StreamError::DepthLimit);
        }
        self.meter.usage.nodes += 1;
        self.meter.usage.depth = self.meter.usage.depth.max(depth);
        Ok(())
    }

    fn begin(&mut self, kind: Kind) -> Result<(), StreamError> {
        self.value()?;
        if self.depth == self.stack.len() {
            return Err(StreamError::StackTooSmall);
        }
        self.raw(if kind == Kind::Object { b"{" } else { b"[" })?;
        self.stack[self.depth] = StreamFrame {
            kind,
            ..StreamFrame::empty()
        };
        self.depth += 1;
        Ok(())
    }

    fn end(&mut self, kind: Kind) -> Result<(), StreamError> {
        if self.depth == 0
            || self.stack[self.depth - 1].kind != kind
            || self.stack[self.depth - 1].waiting
        {
            return Err(StreamError::Structure);
        }
        self.raw(if kind == Kind::Object { b"}" } else { b"]" })?;
        self.depth -= 1;
        Ok(())
    }

    fn ordered_key(&mut self, key: &'a str) -> Result<(), StreamError> {
        self.check_text(key, true)?;
        if self.depth == 0 {
            return Err(StreamError::Structure);
        }
        self.transient.inner = self.stack[self.depth - 1];
        if self.transient.inner.kind != Kind::Object || self.transient.inner.waiting {
            return Err(StreamError::Structure);
        }
        if let Some(previous) = self.transient.inner.previous {
            if self.compare(previous, key)? != Ordering::Less {
                return Err(StreamError::KeyOrder);
            }
        }
        if self.transient.inner.emitted {
            self.raw(b",")?;
        }
        self.text(key, true)?;
        self.raw(b":")?;
        self.stack[self.depth - 1].previous = Some(key);
        self.stack[self.depth - 1].emitted = true;
        self.stack[self.depth - 1].waiting = true;
        Ok(())
    }

    pub fn begin_object(&mut self) -> Result<(), StreamError> {
        self.step(|s| s.begin(Kind::Object))
    }
    pub fn end_object(&mut self) -> Result<(), StreamError> {
        self.step(|s| s.end(Kind::Object))
    }
    pub fn begin_array(&mut self) -> Result<(), StreamError> {
        self.step(|s| s.begin(Kind::Array))
    }
    pub fn end_array(&mut self) -> Result<(), StreamError> {
        self.step(|s| s.end(Kind::Array))
    }
    pub fn key(&mut self, key: &'a str) -> Result<(), StreamError> {
        self.step(|s| s.ordered_key(key))
    }
    pub fn null(&mut self) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.raw(b"null")
        })
    }
    pub fn bool(&mut self, value: bool) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.raw(if value { b"true" } else { b"false" })
        })
    }
    pub fn string(&mut self, value: &str) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.text(value, false)
        })
    }
    pub fn number(&mut self, value: &Number) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.emit(|o| o.number(value))
        })
    }
    fn formatted(
        &mut self,
        operation: impl FnOnce(&mut Meter<'w, W>) -> io::Result<()>,
    ) -> Result<(), StreamError> {
        let result = operation(&mut self.meter);
        match self.meter.failure {
            Some(MeterFailure::Byte) => Err(StreamError::ByteLimit),
            Some(MeterFailure::Work) => Err(StreamError::WorkLimit),
            None => result.map_err(StreamError::Io),
        }
    }
    pub fn u64(&mut self, value: u64) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.formatted(|out| CompactFormatter.write_u64(out, value))
        })
    }
    pub fn i64(&mut self, value: i64) -> Result<(), StreamError> {
        self.step(|s| {
            s.value()?;
            s.formatted(|out| CompactFormatter.write_i64(out, value))
        })
    }
    pub fn f64(&mut self, value: f64) -> Result<(), StreamError> {
        self.step(|s| {
            if !value.is_finite() {
                return Err(StreamError::NonFiniteNumber);
            }
            s.value()?;
            s.formatted(|out| CompactFormatter.write_f64(out, value))
        })
    }

    /// Iterative canonical fragment traversal, using the same stack and counters.
    pub fn json(&mut self, value: &'a Value) -> Result<(), StreamError> {
        self.step(|s| s.fragment(value))
    }

    fn fragment_value(&mut self, value: &'a Value) -> Result<(), StreamError> {
        self.work(1)?;
        match value {
            Value::Null => {
                self.value()?;
                self.raw(b"null")
            }
            Value::Bool(value) => {
                self.value()?;
                self.raw(if *value { b"true" } else { b"false" })
            }
            Value::String(value) => {
                self.value()?;
                self.text(value, false)
            }
            Value::Number(value) => {
                self.value()?;
                self.emit(|o| o.number(value))
            }
            Value::Array(_) => {
                self.begin(Kind::Array)?;
                self.stack[self.depth - 1].fragment = Some(value);
                Ok(())
            }
            Value::Object(_) => {
                self.begin(Kind::Object)?;
                self.stack[self.depth - 1].fragment = Some(value);
                Ok(())
            }
        }
    }

    fn fragment(&mut self, value: &'a Value) -> Result<(), StreamError> {
        self.transient.fragment_base = self.depth;
        self.fragment_value(value)?;
        while self.depth > self.transient.fragment_base {
            self.transient.outer = self.stack[self.depth - 1];
            match self
                .transient
                .outer
                .fragment
                .ok_or(StreamError::Structure)?
            {
                Value::Array(values) => {
                    if let Some(value) = values.get(self.transient.outer.cursor) {
                        self.stack[self.depth - 1].cursor += 1;
                        self.fragment_value(value)?;
                    } else {
                        self.end(Kind::Array)?;
                    }
                }
                Value::Object(values) => {
                    self.transient.selected = None;
                    self.transient.object_iter = Some(values.iter());
                    // Even the final empty selection scan is charged. Long keys
                    // are capped before either prefix or candidate comparison.
                    while let Some((key, value)) = self
                        .transient
                        .object_iter
                        .as_mut()
                        .and_then(|entries| entries.next())
                    {
                        self.work(1)?;
                        self.check_text(key, true)?;
                        if let Some(previous) = self.transient.outer.previous {
                            if self.compare(key, previous)? != Ordering::Greater {
                                continue;
                            }
                        }
                        if let Some((old, _)) = self.transient.selected {
                            if self.compare(key, old)? != Ordering::Less {
                                continue;
                            }
                        }
                        self.transient.selected = Some((key, value));
                    }
                    self.transient.object_iter = None;
                    if let Some((key, value)) = self.transient.selected {
                        self.ordered_key(key)?;
                        self.fragment_value(value)?;
                    } else {
                        self.end(Kind::Object)?;
                    }
                }
                _ => return Err(StreamError::Structure),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
