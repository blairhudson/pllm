//! Bounded structural traversal of existing, immutable JSON; no parser or sink.
//!
//! Complements `Session`: that API preflights/emits canonical bytes, whereas this
//! cursor exposes original values and String owners for census/validation. Root
//! depth is one, as in Session. Sorted mode shares Session's conservative key
//! selection (quadratic per object). Native mode instead retains one concrete
//! serde map iterator per depth for LINEAR census/preflight, with no sorting buffer.
//! Equality uses this same cursor, never recursive Value equality or encoded-byte
//! equality. Both complete inputs are preflighted before returning even `false`.
//!
//! No allocation, user callbacks, input cloning or owner-capacity inference.
//! Callers supply the entire fixed workspace and must charge its unused slots.
//! Quotes are logical state/Result/iterator sizes, NOT Budget authority, allocator
//! capacity, native stack, dependency call-stack, RSS or full admission proofs.
//! Existing census's conservative opaque-map policy remains the caller's policy;
//! a borrowed map iterator does not disclose its backing capacity.
//!
//! The pinned serde_json 1.0.151 leaf Number equality is used directly: integer
//! versus float stays distinct and floating signed zeros compare equal under the
//! current default scalar-Number closure. No numeric conversion is performed.
//! Sorted equality never relies on object insertion order.
//! Number bytes use the pinned CompactFormatter's primitive entry points, without
//! format_args! or its private argument-descriptor backing. This metering contract
//! requires the current default scalar-Number feature closure. Alternate serde
//! feature closures (including arbitrary_precision/preserve_order) are unvalidated
//! and need separate work/backing-policy qualification; leaf equality never
//! coerces Numbers, regardless of feature representation. Number's
//! private itoa/zmij buffers and formatting call-stack have no public bound here;
//! neither is silently included in the concrete formatter adapter's size.
//!
//! Integration: quote before constructing/reserving caller frame storage, drain
//! Traversal to `Ok(None)` for a census, and discard partial totals on any error.
//! Value events expose original Vec/String capacities and object lengths; key
//! events expose original String capacities. Charge those owners with the existing
//! census policy, not payload `Usage::bytes`. Legacy zero-based JSON depth caps
//! need an explicit translation to this Session-compatible one-based depth.
//! Use `preflight`/`structural_eq` when a retained immutable-input binding is needed.
//! Consumer state, counters, policies, callbacks and allocations remain separate.

use super::selection::{KeyMeter, Selection};
use serde_json::{
    ser::{CompactFormatter, Formatter},
    Number, Value,
};
use std::{cmp::Ordering, fmt, io, mem::size_of};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Order {
    Sorted,
    Native,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Total visited values, excluding keys; repeated passes count again.
    pub max_nodes: usize,
    pub max_depth: usize,
    /// Raw UTF-8 keys/string values plus Number Display bytes, NOT canonical
    /// JSON bytes, owned backing, or source-document length. Passes count again.
    pub max_bytes: usize,
    pub max_string_bytes: usize,
    pub max_key_bytes: usize,
    /// One per cursor transition, selection candidate, compared byte pair plus
    /// termination, formatted number byte, and leaf comparison. Not CPU time.
    pub max_work: usize,
    /// Entire quoted logical envelope, including all supplied frame slots.
    pub max_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub nodes: usize,
    pub depth: usize,
    pub bytes: usize,
    pub work: usize,
    pub object_candidates: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Overflow,
    NodeLimit,
    DepthLimit,
    ByteLimit,
    StringLimit,
    KeyLimit,
    WorkLimit,
    WorkspaceLimit,
    WorkspaceTooSmall,
    NumberFormat,
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "borrowed JSON resource arithmetic overflow",
            Self::NodeLimit => "borrowed JSON node limit exceeded",
            Self::DepthLimit => "borrowed JSON depth limit exceeded",
            Self::ByteLimit => "borrowed JSON payload byte limit exceeded",
            Self::StringLimit => "borrowed JSON string limit exceeded",
            Self::KeyLimit => "borrowed JSON key limit exceeded",
            Self::WorkLimit => "borrowed JSON work limit exceeded",
            Self::WorkspaceLimit => "borrowed JSON workspace limit exceeded",
            Self::WorkspaceTooSmall => "borrowed JSON workspace too small",
            Self::NumberFormat => "borrowed JSON number formatting failed",
            Self::Poisoned => "borrowed JSON traversal is poisoned",
        })
    }
}
impl std::error::Error for Error {}

/// Original immutable owners; a key event retains String capacity access for a
/// census. Container events precede children. There are no closing events: array
/// and object lengths are available on the borrowed value. Keys are not nodes.
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    Value {
        value: &'a Value,
        depth: usize,
        /// Zero except for Number. Length, not its owned allocation capacity.
        number_bytes: usize,
    },
    Key {
        key: &'a String,
        depth: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    value: Option<&'a Value>,
    entered: bool,
    index: usize,
    previous: Option<&'a str>,
    pending: Option<&'a Value>,
}

impl Frame<'_> {
    pub const fn empty() -> Self {
        Self {
            value: None,
            entered: false,
            index: 0,
            previous: None,
            pending: None,
        }
    }
    pub const fn storage_bytes() -> usize {
        size_of::<Self>()
    }
}

/// Native-order workspace: unlike Copy sorted frames, each depth owns an actual
/// opaque iterator value. Initialize arrays with `[const { NativeFrame::empty() };
/// N]` only AFTER checking the native quote; never price this as a sorted Frame.
pub struct NativeFrame<'a> {
    frame: Frame<'a>,
    iterator: Option<serde_json::map::Iter<'a>>,
}

impl NativeFrame<'_> {
    pub const fn empty() -> Self {
        Self {
            frame: Frame::empty(),
            iterator: None,
        }
    }
    pub const fn storage_bytes() -> usize {
        size_of::<Self>()
    }
}

enum Workspace<'a, 'w> {
    Sorted(&'w mut [Frame<'a>]),
    Native(&'w mut [NativeFrame<'a>]),
}

impl<'a> Workspace<'a, '_> {
    fn len(&self) -> usize {
        match self {
            Self::Sorted(frames) => frames.len(),
            Self::Native(frames) => frames.len(),
        }
    }
    fn frame(&self, index: usize) -> &Frame<'a> {
        match self {
            Self::Sorted(frames) => &frames[index],
            Self::Native(frames) => &frames[index].frame,
        }
    }
    fn frame_mut(&mut self, index: usize) -> &mut Frame<'a> {
        match self {
            Self::Sorted(frames) => &mut frames[index],
            Self::Native(frames) => &mut frames[index].frame,
        }
    }
    fn reset(&mut self, index: usize, value: &'a Value) {
        let frame = Frame {
            value: Some(value),
            ..Frame::empty()
        };
        match self {
            Self::Sorted(frames) => frames[index] = frame,
            Self::Native(frames) => {
                frames[index] = NativeFrame {
                    frame,
                    iterator: None,
                }
            }
        }
    }
}

/// Includes the concrete map iterator and selected-entry state rather than an
/// opaque allowance. No iterator is held on an unaccounted recursive call frame.
struct Cursor<'a, 'w> {
    workspace: Workspace<'a, 'w>,
    depth: usize,
    current: Frame<'a>,
    selection: Selection<'a>,
}

struct Meter {
    limits: Limits,
    usage: Usage,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

impl Meter {
    fn work(&mut self, amount: usize) -> Result<(), Error> {
        let next = add(self.usage.work, amount)?;
        if next > self.limits.max_work {
            return Err(Error::WorkLimit);
        }
        self.usage.work = next;
        Ok(())
    }
    fn bytes(&mut self, amount: usize) -> Result<(), Error> {
        let next = add(self.usage.bytes, amount)?;
        if next > self.limits.max_bytes {
            return Err(Error::ByteLimit);
        }
        self.usage.bytes = next;
        Ok(())
    }
    fn text_limit(&self, text: &str, key: bool) -> Result<(), Error> {
        if key && text.len() > self.limits.max_key_bytes {
            return Err(Error::KeyLimit);
        }
        if text.len() > self.limits.max_string_bytes {
            return Err(Error::StringLimit);
        }
        Ok(())
    }
    fn text(&mut self, text: &str, key: bool) -> Result<(), Error> {
        self.text_limit(text, key)?;
        self.bytes(text.len())
    }
    fn compare(&mut self, a: &str, b: &str, key: bool) -> Result<Ordering, Error> {
        self.text_limit(a, key)?;
        self.text_limit(b, key)?;
        // Charge the conservative complete common span BEFORE std's byte
        // comparison. This avoids an unpriced iterator or per-byte callback.
        self.work(add(a.len().min(b.len()), 1)?)?;
        Ok(a.cmp(b))
    }
}

impl KeyMeter for Meter {
    type Error = Error;
    fn candidate(&mut self) -> Result<(), Error> {
        self.work(1)?;
        self.usage.object_candidates = add(self.usage.object_candidates, 1)?;
        Ok(())
    }
    fn check_key(&self, key: &str) -> Result<(), Error> {
        self.text_limit(key, true)
    }
    fn compare_keys(&mut self, a: &str, b: &str) -> Result<Ordering, Error> {
        self.compare(a, b, true)
    }
}

struct NumberCounter<'m> {
    meter: &'m mut Meter,
    bytes: usize,
    failure: Option<Error>,
}

impl io::Write for NumberCounter<'_> {
    fn write(&mut self, text: &[u8]) -> io::Result<usize> {
        let result = self.charge(text.len());
        if let Err(error) = result {
            self.failure = Some(error);
            Err(io::ErrorKind::InvalidData.into())
        } else {
            Ok(text.len())
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl NumberCounter<'_> {
    fn charge(&mut self, len: usize) -> Result<(), Error> {
        let bytes = add(self.bytes, len)?;
        self.meter.bytes(len)?;
        self.meter.work(len)?;
        self.bytes = bytes;
        Ok(())
    }
    fn number(&mut self, number: &Number) -> Result<(), Error> {
        // Source: serde_json 1.0.151 number.rs Display and ser.rs primitive
        // CompactFormatter methods use the same itoa/zmij encodings. No private
        // core::fmt::rt::Argument array is constructed by this path.
        let result = if let Some(value) = number.as_u64() {
            CompactFormatter.write_u64(self, value)
        } else if let Some(value) = number.as_i64() {
            CompactFormatter.write_i64(self, value)
        } else if let Some(value) = number.as_f64() {
            CompactFormatter.write_f64(self, value)
        } else {
            return Err(Error::NumberFormat);
        };
        match result {
            Ok(()) => Ok(()),
            Err(_) => Err(self.failure.unwrap_or(Error::NumberFormat)),
        }
    }
}

impl<'a, 'w> Cursor<'a, 'w> {
    fn new(root: &'a Value, mut workspace: Workspace<'a, 'w>) -> Result<Self, Error> {
        if workspace.len() == 0 {
            return Err(Error::WorkspaceTooSmall);
        }
        workspace.reset(0, root);
        Ok(Self {
            workspace,
            depth: 1,
            current: Frame::empty(),
            selection: Selection::empty(),
        })
    }

    fn push(&mut self, value: &'a Value, meter: &Meter) -> Result<(), Error> {
        let depth = add(self.depth, 1)?;
        if depth > meter.limits.max_depth {
            return Err(Error::DepthLimit);
        }
        if depth > self.workspace.len() {
            return Err(Error::WorkspaceTooSmall);
        }
        self.workspace.reset(self.depth, value);
        self.depth = depth;
        Ok(())
    }

    fn next(&mut self, meter: &mut Meter) -> Result<Option<Event<'a>>, Error> {
        while self.depth != 0 {
            meter.work(1)?;
            self.current = *self.workspace.frame(self.depth - 1);
            let value = self
                .current
                .value
                .expect("only initialized frames are live");
            if !self.current.entered {
                if self.depth > meter.limits.max_depth {
                    return Err(Error::DepthLimit);
                }
                let nodes = add(meter.usage.nodes, 1)?;
                if nodes > meter.limits.max_nodes {
                    return Err(Error::NodeLimit);
                }
                meter.usage.nodes = nodes;
                meter.usage.depth = meter.usage.depth.max(self.depth);
                self.workspace.frame_mut(self.depth - 1).entered = true;
                let mut number_bytes = 0;
                match value {
                    Value::String(text) => meter.text(text, false)?,
                    Value::Number(number) => {
                        let mut output = NumberCounter {
                            meter,
                            bytes: 0,
                            failure: None,
                        };
                        output.number(number)?;
                        number_bytes = output.bytes;
                    }
                    Value::Object(values) => {
                        if let Workspace::Native(frames) = &mut self.workspace {
                            frames[self.depth - 1].iterator = Some(values.iter());
                        }
                    }
                    _ => {}
                }
                return Ok(Some(Event::Value {
                    value,
                    depth: self.depth,
                    number_bytes,
                }));
            }
            match value {
                Value::Array(values) => {
                    if let Some(child) = values.get(self.current.index) {
                        self.workspace.frame_mut(self.depth - 1).index =
                            add(self.current.index, 1)?;
                        self.push(child, meter)?;
                    } else {
                        self.depth -= 1;
                    }
                }
                Value::Object(values) => {
                    if let Some(child) = self.current.pending {
                        self.workspace.frame_mut(self.depth - 1).pending = None;
                        self.push(child, meter)?;
                        continue;
                    }
                    let selected = match &mut self.workspace {
                        Workspace::Sorted(_) => {
                            self.selection.run(values, self.current.previous, meter)?;
                            self.selection.selected()
                        }
                        Workspace::Native(frames) => {
                            let selected = frames[self.depth - 1]
                                .iterator
                                .as_mut()
                                .expect("entered native object owns its iterator")
                                .next();
                            if let Some((key, _)) = selected {
                                meter.candidate()?;
                                meter.check_key(key)?;
                            }
                            selected
                        }
                    };
                    if let Some((key, child)) = selected {
                        meter.text(key, true)?;
                        self.workspace.frame_mut(self.depth - 1).previous = Some(key);
                        self.workspace.frame_mut(self.depth - 1).pending = Some(child);
                        return Ok(Some(Event::Key {
                            key,
                            depth: self.depth,
                        }));
                    }
                    self.depth -= 1;
                }
                _ => self.depth -= 1,
            }
        }
        Ok(None)
    }
}

/// Retained state plus the maximum named transition phase, not compiled stack
/// layout. State includes meter, cursor snapshots, borrowed slices and concrete
/// map iterator(s). Each phase explicitly prices its argument/Result copies.
/// Unused supplied frame slots count in full. No borrowed input backing is added.
/// As with Session resources, ordinary scalar evaluation locals/reference aliases
/// are excluded. Private numeric-library buffers are explicitly unknown, not zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resources {
    pub order: Order,
    pub frame_size: usize,
    pub frame_alignment: usize,
    pub frame_bytes: usize,
    pub state_bytes: usize,
    pub operation_result_bytes: usize,
    pub visit_result_bytes: usize,
    pub formatter_bytes: usize,
    pub format_arguments_bytes: usize,
    pub format_argument_backing_bytes: usize,
    pub format_result_bytes: usize,
    pub unit_result_bytes: usize,
    pub leaf_result_bytes: usize,
    pub arithmetic_result_bytes: usize,
    pub layout_bytes: usize,
    pub layout_result_bytes: usize,
    pub quote_bytes: usize,
    pub cursor_result_bytes: usize,
    pub ordering_result_bytes: usize,
    pub iterator_bytes: usize,
    pub iterator_option_bytes: usize,
    pub iterator_entry_result_bytes: usize,
    pub entry_binding_bytes: usize,
    pub event_pair_bytes: usize,
    pub optional_event_pair_bytes: usize,
    pub same_event_arguments_bytes: usize,
    pub comparison_arguments_bytes: usize,
    pub cursor_arguments_bytes: usize,
    pub workspace_argument_bytes: usize,
    pub frame_transition_bytes: usize,
    pub number_arguments_bytes: usize,
    pub number_write_arguments_bytes: usize,
    pub selection_arguments_bytes: usize,
    pub number_probe_bytes: usize,
    pub formatter_write_result_bytes: usize,
    pub formatter_error_bytes: usize,
    pub constructor_result_bytes: usize,
    pub completion_value_bytes: usize,
    pub phases: Phases,
    pub logical_workspace_bytes: usize,
    pub dependency_buffer_bytes: Option<usize>,
    pub dependency_call_stack_bytes: Option<usize>,
}

/// Known logical transition slots, in addition to retained frames/state. Each
/// phase names its own return/argument copies; no phase borrows unused "padding"
/// from a different result type. These are conservative source-level coexistence
/// envelopes, NOT an ABI/compiler-temporary or machine-stack layout assertion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Phases {
    pub quotation: usize,
    pub construction: usize,
    pub entry: usize,
    pub number: usize,
    pub selection: usize,
    pub delivery: usize,
    pub equality: usize,
    pub completion: usize,
}

impl Phases {
    pub fn maximum(&self) -> usize {
        self.quotation
            .max(self.construction)
            .max(self.entry)
            .max(self.number)
            .max(self.selection)
            .max(self.delivery)
            .max(self.equality)
            .max(self.completion)
    }
}

fn times(bytes: usize, count: usize) -> Result<usize, Error> {
    bytes.checked_mul(count).ok_or(Error::Overflow)
}

fn resources(
    frame_slots: usize,
    order: Order,
    state_bytes: usize,
    result_bytes: usize,
    constructor_result_bytes: usize,
    completion_value_bytes: usize,
    equality: bool,
) -> Result<Resources, Error> {
    // Layout checks isize/size/alignment constraints before caller construction.
    let layout = match order {
        Order::Sorted => std::alloc::Layout::array::<Frame<'_>>(frame_slots),
        Order::Native => std::alloc::Layout::array::<NativeFrame<'_>>(frame_slots),
    }
    .map_err(|_| Error::Overflow)?;
    let (frame_size, frame_alignment) = match order {
        Order::Sorted => (size_of::<Frame<'_>>(), std::mem::align_of::<Frame<'_>>()),
        Order::Native => (
            size_of::<NativeFrame<'_>>(),
            std::mem::align_of::<NativeFrame<'_>>(),
        ),
    };
    let mut quote = Resources {
        order,
        frame_size,
        frame_alignment,
        frame_bytes: layout.size(),
        state_bytes,
        operation_result_bytes: result_bytes,
        visit_result_bytes: size_of::<Result<Option<Event<'_>>, Error>>(),
        formatter_bytes: add(
            size_of::<NumberCounter<'_>>(),
            size_of::<CompactFormatter>(),
        )?,
        // Borrowed numeric counting no longer uses format_args!: no hidden
        // core::fmt::rt::Argument pointee array to guess from Arguments' size.
        format_arguments_bytes: 0,
        format_argument_backing_bytes: 0,
        format_result_bytes: size_of::<io::Result<()>>(),
        unit_result_bytes: size_of::<Result<(), Error>>(),
        leaf_result_bytes: size_of::<Result<bool, Error>>(),
        arithmetic_result_bytes: size_of::<Result<usize, Error>>(),
        layout_bytes: size_of::<std::alloc::Layout>(),
        layout_result_bytes: size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
        quote_bytes: add(
            size_of::<Resources>(),
            size_of::<Result<Resources, Error>>(),
        )?,
        cursor_result_bytes: size_of::<Result<Cursor<'_, '_>, Error>>(),
        ordering_result_bytes: size_of::<Result<Ordering, Error>>(),
        iterator_bytes: size_of::<serde_json::map::Iter<'_>>(),
        iterator_option_bytes: size_of::<Option<serde_json::map::Iter<'_>>>(),
        iterator_entry_result_bytes: size_of::<Option<(&String, &Value)>>(),
        entry_binding_bytes: size_of::<(&String, &Value)>(),
        event_pair_bytes: size_of::<(Event<'_>, Event<'_>)>(),
        optional_event_pair_bytes: size_of::<(Option<Event<'_>>, Option<Event<'_>>)>(),
        same_event_arguments_bytes: size_of::<(Event<'_>, Event<'_>, &mut Meter)>(),
        // Trait forwarding -> Meter::compare (with key flag) -> str::cmp.
        comparison_arguments_bytes: add(
            add(
                size_of::<(&mut Meter, &str, &str)>(),
                size_of::<(&mut Meter, &str, &str, bool)>(),
            )?,
            size_of::<(&str, &str)>(),
        )?,
        cursor_arguments_bytes: size_of::<(&mut Cursor<'_, '_>, &mut Meter)>(),
        workspace_argument_bytes: add(size_of::<Workspace<'_, '_>>(), size_of::<&Value>())?,
        // reset constructs a Frame; native reset additionally constructs a
        // NativeFrame. Both slots are explicit, not mistaken for live input.
        frame_transition_bytes: add(size_of::<Frame<'_>>(), frame_size)?,
        number_arguments_bytes: add(
            size_of::<(&Number, &mut NumberCounter<'_>)>(),
            size_of::<u64>(),
        )?,
        number_write_arguments_bytes: add(
            size_of::<(&mut NumberCounter<'_>, &[u8])>(),
            size_of::<(&mut NumberCounter<'_>, usize)>(),
        )?,
        selection_arguments_bytes: size_of::<(
            &mut Selection<'_>,
            &serde_json::Map<String, Value>,
            Option<&str>,
            &mut Meter,
        )>(),
        number_probe_bytes: add(
            add(size_of::<Option<u64>>(), size_of::<Option<i64>>())?,
            size_of::<Option<f64>>(),
        )?,
        formatter_write_result_bytes: size_of::<io::Result<usize>>(),
        formatter_error_bytes: size_of::<io::Error>(),
        constructor_result_bytes,
        completion_value_bytes,
        phases: Phases::default(),
        logical_workspace_bytes: 0,
        dependency_buffer_bytes: None,
        dependency_call_stack_bytes: None,
    };
    // Quote creation/check forwarding: local Resources + returned Resources,
    // producer/caller Result slots, Layout and its Result, checked sum results,
    // and the by-value Limits argument. No temporary accounting-term array.
    let phase = add(times(quote.quote_bytes, 2)?, quote.layout_bytes)?;
    let phase = add(phase, quote.layout_result_bytes)?;
    let phase = add(phase, times(quote.arithmetic_result_bytes, 2)?)?;
    quote.phases.quotation = add(phase, size_of::<Limits>())?;

    // Cursor::new return + caller ? Result; constructor result; Workspace/root
    // argument copy and frame initializer/reset slots. Equality builds two cursors
    // sequentially: their retained owners already reside in state_bytes.
    let phase = add(
        times(quote.cursor_result_bytes, 2)?,
        times(quote.constructor_result_bytes, 2)?,
    )?;
    let phase = add(phase, quote.workspace_argument_bytes)?;
    let phase = add(phase, quote.frame_transition_bytes)?;
    quote.phases.construction = add(phase, quote.unit_result_bytes)?;

    // Scalar/container entry: producer Event -> Option -> return Result,
    // text_limit/bytes/text unit returns, node/byte checked arithmetic, and
    // native iterator construction on object entry. These are separate logical
    // slots, not space borrowed from Result<Cursor> or Resources padding.
    let phase = add(size_of::<Event<'_>>(), size_of::<Option<Event<'_>>>())?;
    let phase = add(phase, quote.visit_result_bytes)?;
    let phase = add(phase, times(quote.unit_result_bytes, 3)?)?;
    let phase = add(phase, times(quote.arithmetic_result_bytes, 2)?)?;
    let phase = add(phase, quote.iterator_bytes)?;
    quote.phases.entry = add(phase, quote.iterator_option_bytes)?;

    // NumberCounter local, primitive argument/probe slots, formatter return and
    // caller result, Write::write return + caller write_all propagation, builtin
    // io error binding, NumberCounter::number and Cursor ? unit result copies.
    let phase = add(quote.formatter_bytes, quote.number_arguments_bytes)?;
    let phase = add(phase, quote.number_write_arguments_bytes)?;
    let phase = add(phase, quote.number_probe_bytes)?;
    let phase = add(phase, times(quote.format_result_bytes, 2)?)?;
    let phase = add(phase, times(quote.formatter_write_result_bytes, 2)?)?;
    let phase = add(phase, quote.formatter_error_bytes)?;
    let phase = add(phase, times(quote.unit_result_bytes, 2)?)?;
    let phase = add(phase, quote.arithmetic_result_bytes)?;
    quote.phases.number = add(phase, quote.visit_result_bytes)?;

    // Iterator construction/Some temporary, next/and_then/selected()/local
    // Option-entry copies; candidate/best/final tuple bindings. Sorted comparisons
    // additionally return through Meter, KeyMeter forwarding, and selector ?.
    let phase = add(quote.iterator_bytes, quote.iterator_option_bytes)?;
    let phase = add(phase, times(quote.iterator_entry_result_bytes, 4)?)?;
    let phase = add(phase, times(quote.entry_binding_bytes, 3)?)?;
    // Meter leaf -> KeyMeter forwarding -> Selection::run -> caller ?.
    let phase = add(phase, times(quote.unit_result_bytes, 4)?)?;
    let phase = add(phase, quote.arithmetic_result_bytes)?;
    let phase = add(phase, quote.selection_arguments_bytes)?;
    quote.phases.selection = if order == Order::Sorted {
        add(
            add(phase, times(quote.ordering_result_bytes, 3)?)?,
            quote.comparison_arguments_bytes,
        )?
    } else {
        phase
    };

    // Cursor return -> next_event local result -> preflight/consumer ? result;
    // caller/callee cursor argument slots plus push's frame construction.
    let phase = add(
        times(quote.visit_result_bytes, 3)?,
        times(quote.cursor_arguments_bytes, 2)?,
    )?;
    let phase = add(phase, quote.frame_transition_bytes)?;
    let phase = add(phase, times(quote.unit_result_bytes, 2)?)?;
    quote.phases.delivery = add(phase, quote.arithmetic_result_bytes)?;

    if equality {
        // match optional-event tuple, caller Some bindings, actual same_event
        // arguments (including meter reference), and callee match Event tuple.
        let phase = add(
            quote.optional_event_pair_bytes,
            times(quote.event_pair_bytes, 2)?,
        )?;
        let phase = add(phase, quote.same_event_arguments_bytes)?;
        let phase = add(phase, times(quote.leaf_result_bytes, 2)?)?;
        let phase = add(phase, quote.comparison_arguments_bytes)?;
        let phase = add(phase, times(quote.ordering_result_bytes, 2)?)?;
        let phase = add(phase, quote.unit_result_bytes)?;
        quote.phases.equality = add(phase, quote.arithmetic_result_bytes)?;
    }
    // Completed token/value plus producer and caller Result slots.
    quote.phases.completion = add(
        quote.completion_value_bytes,
        times(quote.operation_result_bytes, 2)?,
    )?;
    quote.logical_workspace_bytes = add(
        add(quote.frame_bytes, quote.state_bytes)?,
        quote.phases.maximum(),
    )?;
    Ok(quote)
}

fn check_resources(resources: Resources, limits: Limits) -> Result<(), Error> {
    if resources.logical_workspace_bytes > limits.max_workspace_bytes {
        Err(Error::WorkspaceLimit)
    } else {
        Ok(())
    }
}

impl Resources {
    /// Check BEFORE constructing caller arrays. Construction rechecks the quote
    /// for the actual typed workspace/mode; this check grants no reservation.
    pub fn check(&self, limits: Limits) -> Result<(), Error> {
        check_resources(*self, limits)
    }
}

/// Stateful fallible event iterator. After a limit/error, every later call returns
/// Poisoned; partial census output must not be promoted to a successful preflight.
pub struct Traversal<'a, 'w> {
    cursor: Cursor<'a, 'w>,
    meter: Meter,
    failed: bool,
}

impl<'a, 'w> Traversal<'a, 'w> {
    /// Quote before allocating a caller stack or constructing this traversal.
    /// Also covers the optional preflight token and its return Result.
    pub fn resources_for_slots(slots: usize) -> Result<Resources, Error> {
        Self::resources_for_order(slots, Order::Sorted)
    }

    pub fn resources_for_native_slots(slots: usize) -> Result<Resources, Error> {
        Self::resources_for_order(slots, Order::Native)
    }

    fn resources_for_order(slots: usize, order: Order) -> Result<Resources, Error> {
        resources(
            slots,
            order,
            size_of::<Self>(),
            size_of::<Result<Preflight<'a>, Error>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Preflight<'a>>(),
            false,
        )
    }

    pub fn new(
        value: &'a Value,
        workspace: &'w mut [Frame<'a>],
        limits: Limits,
    ) -> Result<Self, Error> {
        check_resources(Self::resources_for_slots(workspace.len())?, limits)?;
        Ok(Self {
            cursor: Cursor::new(value, Workspace::Sorted(workspace))?,
            meter: Meter {
                limits,
                usage: Usage::default(),
            },
            failed: false,
        })
    }

    /// Native iteration is for census/preflight only. Unlike sorted mode, its
    /// event order follows the resolved map backend; equality always uses sorted
    /// mode. Per-depth opaque iterators are priced with NativeFrame's actual type.
    pub fn new_native(
        value: &'a Value,
        workspace: &'w mut [NativeFrame<'a>],
        limits: Limits,
    ) -> Result<Self, Error> {
        check_resources(Self::resources_for_native_slots(workspace.len())?, limits)?;
        Ok(Self {
            cursor: Cursor::new(value, Workspace::Native(workspace))?,
            meter: Meter {
                limits,
                usage: Usage::default(),
            },
            failed: false,
        })
    }

    pub fn usage(&self) -> Usage {
        self.meter.usage
    }

    pub fn next_event(&mut self) -> Result<Option<Event<'a>>, Error> {
        if self.failed {
            return Err(Error::Poisoned);
        }
        self.failed = true;
        let result = self.cursor.next(&mut self.meter);
        self.failed = result.is_err();
        result
    }
}

/// Binds successful structural preflight to its original immutable input. Not a
/// canonical-byte, owned-backing, admission or semantic-validation certificate.
///
/// ```compile_fail
/// use pllm_types::bounded_canonical::borrowed::{preflight, Frame, Limits};
/// let mut value = serde_json::json!(null);
/// let mut frames = [Frame::empty(); 1];
/// let limits = Limits { max_nodes: 1, max_depth: 1, max_bytes: 100,
///     max_string_bytes: 100, max_key_bytes: 100, max_work: 100,
///     max_workspace_bytes: usize::MAX };
/// let admitted = preflight(&value, &mut frames, limits).unwrap();
/// value = serde_json::json!(true); // cannot mutate original while bound
/// assert!(admitted.value().is_null());
/// ```
#[derive(Debug)]
pub struct Preflight<'a> {
    value: &'a Value,
    usage: Usage,
}

impl<'a> Preflight<'a> {
    pub fn value(&self) -> &'a Value {
        self.value
    }
    pub fn usage(&self) -> Usage {
        self.usage
    }
}

pub fn preflight<'a>(
    value: &'a Value,
    workspace: &mut [Frame<'a>],
    limits: Limits,
) -> Result<Preflight<'a>, Error> {
    let mut traversal = Traversal::new(value, workspace, limits)?;
    while traversal.next_event()?.is_some() {}
    Ok(Preflight {
        value,
        usage: traversal.usage(),
    })
}

pub fn preflight_native<'a>(
    value: &'a Value,
    workspace: &mut [NativeFrame<'a>],
    limits: Limits,
) -> Result<Preflight<'a>, Error> {
    let mut traversal = Traversal::new_native(value, workspace, limits)?;
    while traversal.next_event()?.is_some() {}
    Ok(Preflight {
        value,
        usage: traversal.usage(),
    })
}

/// Comparison result bound to both original immutable inputs, not to later values
/// with matching bytes/digests. Copying its boolean/usage alone carries no proof.
///
/// ```compile_fail
/// use pllm_types::bounded_canonical::borrowed::{structural_eq, Frame, Limits};
/// let mut left = serde_json::json!([1]);
/// let right = serde_json::json!([1]);
/// let mut a = [Frame::empty(); 2];
/// let mut b = [Frame::empty(); 2];
/// let limits = Limits { max_nodes: 100, max_depth: 2, max_bytes: 100,
///     max_string_bytes: 100, max_key_bytes: 100, max_work: 1000,
///     max_workspace_bytes: usize::MAX };
/// let comparison = structural_eq(&left, &right, &mut a, &mut b, limits).unwrap();
/// left[0] = serde_json::json!(2);
/// assert_eq!(comparison.left(), &right); // original left is still borrowed
/// ```
#[derive(Debug)]
pub struct Equality<'a> {
    left: &'a Value,
    right: &'a Value,
    pub equal: bool,
    /// Shared aggregate meter, INCLUDING both preflights and comparison replay.
    pub usage: Usage,
}

impl<'a> Equality<'a> {
    pub fn left(&self) -> &'a Value {
        self.left
    }
    pub fn right(&self) -> &'a Value {
        self.right
    }
}

struct EqualityState<'a, 'w> {
    left: Cursor<'a, 'w>,
    right: Cursor<'a, 'w>,
    meter: Meter,
    left_event: Option<Event<'a>>,
    right_event: Option<Event<'a>>,
}

pub fn equality_resources_for_slots(left: usize, right: usize) -> Result<Resources, Error> {
    resources(
        add(left, right)?,
        Order::Sorted,
        size_of::<EqualityState<'_, '_>>(),
        size_of::<Result<Equality<'_>, Error>>(),
        0, // no Traversal wrapper is constructed by structural_eq
        size_of::<Equality<'_>>(),
        true,
    )
}

fn same_event(a: Event<'_>, b: Event<'_>, meter: &mut Meter) -> Result<bool, Error> {
    meter.work(1)?;
    Ok(match (a, b) {
        (Event::Key { key: a, .. }, Event::Key { key: b, .. }) => {
            meter.compare(a, b, true)?.is_eq()
        }
        (
            Event::Value {
                value: a,
                number_bytes: ab,
                ..
            },
            Event::Value {
                value: b,
                number_bytes: bb,
                ..
            },
        ) => match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => {
                meter.work(add(ab.min(bb), 1)?)?;
                a == b // leaf Number equality, NOT recursive Value::PartialEq
            }
            (Value::String(a), Value::String(b)) => meter.compare(a, b, false)?.is_eq(),
            (Value::Array(a), Value::Array(b)) => a.len() == b.len(),
            (Value::Object(a), Value::Object(b)) => a.len() == b.len(),
            _ => false,
        },
        _ => false,
    })
}

/// Structural Value equality with a single aggregate limit set across both full
/// preflights and comparison replay (keys sorted, arrays positional). `false`
/// cannot bypass an oversized/deep malformed suffix. Limits remain opt-in and do
/// not alter Session, canonical bytes/digests, or legacy census APIs.
pub fn structural_eq<'a>(
    left: &'a Value,
    right: &'a Value,
    left_workspace: &mut [Frame<'a>],
    right_workspace: &mut [Frame<'a>],
    limits: Limits,
) -> Result<Equality<'a>, Error> {
    check_resources(
        equality_resources_for_slots(left_workspace.len(), right_workspace.len())?,
        limits,
    )?;
    let mut state = EqualityState {
        left: Cursor::new(left, Workspace::Sorted(left_workspace))?,
        right: Cursor::new(right, Workspace::Sorted(right_workspace))?,
        meter: Meter {
            limits,
            usage: Usage::default(),
        },
        left_event: None,
        right_event: None,
    };
    while state.left.next(&mut state.meter)?.is_some() {}
    while state.right.next(&mut state.meter)?.is_some() {}
    // Cursors only retain immutable input references; reset without new storage.
    state.left.workspace.reset(0, left);
    state.left.depth = 1;
    state.right.workspace.reset(0, right);
    state.right.depth = 1;
    loop {
        state.left_event = state.left.next(&mut state.meter)?;
        state.right_event = state.right.next(&mut state.meter)?;
        let equal = match (state.left_event, state.right_event) {
            (None, None) => {
                return Ok(Equality {
                    left,
                    right,
                    equal: true,
                    usage: state.meter.usage,
                })
            }
            (Some(a), Some(b)) => same_event(a, b, &mut state.meter)?,
            _ => false,
        };
        if !equal {
            return Ok(Equality {
                left,
                right,
                equal: false,
                usage: state.meter.usage,
            });
        }
    }
}

#[cfg(test)]
#[path = "borrowed_tests.rs"]
mod tests;
