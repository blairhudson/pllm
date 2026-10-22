//! Allocation-free canonical JSON for borrowed values and explicit schema views.
//!
//! This is deliberately not a generic `Serialize` implementation: arbitrary map
//! serializers can require unbounded buffering, and user serializers can allocate.
//! Existing JSON attributes stay borrowed through [`Node::Json`]. Typed schemas
//! compose borrowed [`Node::Object`] and [`Node::Array`] slices, including stack
//! arrays. Object keys are selected lexicographically without sorting scratch
//! (quadratic comparisons per object); duplicate schema keys are rejected.
//!
//! Admission traverses iteratively, counting exact encoded bytes before hashing
//! or writing. A node is a JSON value, not an object key; root depth is one.
//! Workspace is exactly one [`Frame`] per active depth, measured using
//! `size_of::<Frame>()`, not an estimated byte allowance. Input storage, caller
//! unused slice capacity, sinks, schema-view construction, and fixed-size walker,
//! number formatter, and SHA-256 state are outside this traversal-workspace budget.
//! No heap allocation, recursion, heap-backed number/string/output buffers, or
//! authority decisions occur here. Errors from an external writer can leave that
//! writer partially written; admission/workspace failures never touch it.

use serde_json::{Number, Value};
use std::{fmt, io, mem::size_of};

pub mod borrowed;
mod selection;
mod stream;
pub use stream::{
    DomainHash, Resources, Session, StreamError, StreamFrame, StreamLimits, StreamUsage,
};

/// Borrowed, JSON-only schema writer hooks. No arbitrary serializer callbacks.
#[derive(Clone, Copy, Debug)]
pub enum Node<'a> {
    Json(&'a Value),
    Null,
    Bool(bool),
    Number(&'a Number),
    String(&'a str),
    Array(&'a [Node<'a>]),
    Object(&'a [(&'a str, Node<'a>)]),
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_nodes: usize,
    pub max_depth: usize,
    pub max_bytes: usize,
    /// Maximum live frame storage, excluding unused caller-provided slots.
    pub max_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    pub nodes: usize,
    pub depth: usize,
    pub bytes: usize,
    pub workspace_bytes: usize,
}

#[derive(Debug)]
pub enum Error {
    NodeLimit,
    DepthLimit,
    ByteLimit,
    WorkspaceLimit,
    WorkspaceTooSmall,
    DuplicateKey,
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeLimit => f.write_str("canonical JSON node limit exceeded"),
            Self::DepthLimit => f.write_str("canonical JSON depth limit exceeded"),
            Self::ByteLimit => f.write_str("canonical JSON byte limit exceeded"),
            Self::WorkspaceLimit => f.write_str("canonical JSON workspace limit exceeded"),
            Self::WorkspaceTooSmall => f.write_str("canonical JSON workspace too small"),
            Self::DuplicateKey => f.write_str("duplicate canonical JSON object key"),
            Self::Io(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for Error {}

/// Caller-owned traversal slot. Initialize slices using `Frame::empty()`.
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    node: Node<'a>,
    entered: bool,
    index: usize,
    previous_key: Option<&'a str>,
}

impl<'a> Frame<'a> {
    pub const fn empty() -> Self {
        Self {
            node: Node::Null,
            entered: false,
            index: 0,
            previous_key: None,
        }
    }

    fn new(node: Node<'a>) -> Self {
        Self {
            node,
            ..Self::empty()
        }
    }
}

/// Immutable input plus its successful preflight. Cannot be constructed unchecked.
#[derive(Debug)]
pub struct Admitted<'a> {
    root: Node<'a>,
    limits: Limits,
    usage: Usage,
}

pub fn admit<'a>(
    root: Node<'a>,
    limits: Limits,
    workspace: &mut [Frame<'a>],
) -> Result<Admitted<'a>, Error> {
    let usage = walk(root, limits, workspace, &mut io::sink())?;
    Ok(Admitted {
        root,
        limits,
        usage,
    })
}

impl<'a> Admitted<'a> {
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Writes only after admission; a short workspace fails before touching output.
    pub fn write<W: io::Write>(
        &self,
        workspace: &mut [Frame<'a>],
        output: &mut W,
    ) -> Result<(), Error> {
        if workspace.len() < self.usage.depth {
            return Err(Error::WorkspaceTooSmall);
        }
        walk(self.root, self.limits, workspace, output)?;
        Ok(())
    }

    /// Exact legacy `domain.as_bytes() || [0] || canonical_json` SHA-256.
    /// Returns raw digest bytes, avoiding even the legacy hex String allocation.
    /// Domain bytes are not JSON bytes and are not charged to `max_bytes`.
    pub fn hash(&self, domain: &str, workspace: &mut [Frame<'a>]) -> Result<[u8; 32], Error> {
        if workspace.len() < self.usage.depth {
            return Err(Error::WorkspaceTooSmall);
        }
        let mut hash = DomainHash::new(domain);
        self.write(workspace, &mut hash)?;
        Ok(hash.finish())
    }
}

struct Output<'a, W> {
    writer: &'a mut W,
    bytes: usize,
    limit: usize,
    exceeded: bool,
}

impl<W: io::Write> io::Write for Output<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(total) = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit)
        else {
            self.exceeded = true;
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        };
        self.writer.write_all(bytes)?;
        self.bytes = total;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl<W: io::Write> Output<'_, W> {
    fn result(&self, result: io::Result<()>) -> Result<(), Error> {
        result.map_err(|error| {
            if self.exceeded {
                Error::ByteLimit
            } else {
                Error::Io(error)
            }
        })
    }

    fn raw(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let result = io::Write::write_all(self, bytes);
        self.result(result)
    }

    fn string(&mut self, text: &str) -> Result<(), Error> {
        // Every UTF-8 byte survives or expands. Reject oversized strings without
        // scanning their contents, before accounting for additional escapes.
        if text
            .len()
            .checked_add(2)
            .is_none_or(|minimum| minimum > self.limit - self.bytes)
        {
            return Err(Error::ByteLimit);
        }
        // serde_json's default compact formatter: preserve UTF-8, escape only
        // quote, backslash, and U+0000..U+001F, with lowercase hex digits.
        self.raw(b"\"")?;
        let mut start = 0;
        for (index, byte) in text.bytes().enumerate() {
            let escape: &[u8] = match byte {
                b'"' => b"\\\"",
                b'\\' => b"\\\\",
                b'\x08' => b"\\b",
                b'\x0c' => b"\\f",
                b'\n' => b"\\n",
                b'\r' => b"\\r",
                b'\t' => b"\\t",
                0..=31 => {
                    self.raw(&text.as_bytes()[start..index])?;
                    const HEX: &[u8; 16] = b"0123456789abcdef";
                    self.raw(&[
                        b'\\',
                        b'u',
                        b'0',
                        b'0',
                        HEX[(byte >> 4) as usize],
                        HEX[(byte & 15) as usize],
                    ])?;
                    start = index + 1;
                    continue;
                }
                _ => continue,
            };
            self.raw(&text.as_bytes()[start..index])?;
            self.raw(escape)?;
            start = index + 1;
        }
        self.raw(&text.as_bytes()[start..])?;
        self.raw(b"\"")
    }

    fn number(&mut self, number: &Number) -> Result<(), Error> {
        // Display is exactly what legacy Number::to_string uses, without String.
        let result = io::Write::write_fmt(self, format_args!("{number}"));
        self.result(result)
    }
}

fn normalized(node: Node<'_>) -> Node<'_> {
    match node {
        Node::Json(Value::Null) => Node::Null,
        Node::Json(Value::Bool(value)) => Node::Bool(*value),
        Node::Json(Value::Number(value)) => Node::Number(value),
        Node::Json(Value::String(value)) => Node::String(value),
        node => node,
    }
}

fn next_entry<'a>(
    node: Node<'a>,
    previous: Option<&str>,
) -> Result<Option<(&'a str, Node<'a>)>, Error> {
    fn select<'a>(
        entries: impl Iterator<Item = (&'a str, Node<'a>)>,
        previous: Option<&str>,
    ) -> Result<Option<(&'a str, Node<'a>)>, Error> {
        let mut best: Option<(&str, Node<'a>)> = None;
        let mut duplicate = false;
        for (key, value) in entries {
            if previous.is_some_and(|previous| key <= previous) {
                continue;
            }
            match best {
                None => {
                    best = Some((key, value));
                    duplicate = false;
                }
                Some((old, _)) if key < old => {
                    best = Some((key, value));
                    duplicate = false;
                }
                Some((old, _)) if key == old => duplicate = true,
                _ => {}
            }
        }
        if duplicate {
            return Err(Error::DuplicateKey);
        }
        Ok(best)
    }
    match node {
        Node::Object(entries) => select(entries.iter().copied(), previous),
        Node::Json(Value::Object(entries)) => select(
            entries.iter().map(|(k, v)| (k.as_str(), Node::Json(v))),
            previous,
        ),
        _ => unreachable!("only object nodes select entries"),
    }
}

fn walk<'a, W: io::Write>(
    root: Node<'a>,
    limits: Limits,
    workspace: &mut [Frame<'a>],
    writer: &mut W,
) -> Result<Usage, Error> {
    fn check_push(
        depth: usize,
        nodes: usize,
        limits: Limits,
        slots: usize,
    ) -> Result<usize, Error> {
        if nodes >= limits.max_nodes {
            return Err(Error::NodeLimit);
        }
        if depth > limits.max_depth {
            return Err(Error::DepthLimit);
        }
        let bytes = depth
            .checked_mul(size_of::<Frame<'_>>())
            .ok_or(Error::WorkspaceLimit)?;
        if bytes > limits.max_workspace_bytes {
            return Err(Error::WorkspaceLimit);
        }
        if depth > slots {
            return Err(Error::WorkspaceTooSmall);
        }
        Ok(bytes)
    }
    let mut usage = Usage {
        nodes: 1,
        depth: 1,
        bytes: 0,
        workspace_bytes: check_push(1, 0, limits, workspace.len())?,
    };
    workspace[0] = Frame::new(normalized(root));
    let mut depth = 1;
    let mut output = Output {
        writer,
        bytes: 0,
        limit: limits.max_bytes,
        exceeded: false,
    };
    while depth != 0 {
        let frame = &mut workspace[depth - 1];
        let node = frame.node;
        if !frame.entered {
            frame.entered = true;
            // Reject obviously over-wide containers before key-selection scans.
            // Every immediate child requires at least one additional value node.
            let children = match node {
                Node::Array(values) => values.len(),
                Node::Object(values) => values.len(),
                Node::Json(Value::Array(values)) => values.len(),
                Node::Json(Value::Object(values)) => values.len(),
                _ => 0,
            };
            if children > limits.max_nodes - usage.nodes {
                return Err(Error::NodeLimit);
            }
            match node {
                Node::Null => output.raw(b"null")?,
                Node::Bool(value) => output.raw(if value { b"true" } else { b"false" })?,
                Node::Number(value) => output.number(value)?,
                Node::String(value) => output.string(value)?,
                Node::Array(_) | Node::Json(Value::Array(_)) => {
                    output.raw(b"[")?;
                    continue;
                }
                Node::Object(_) | Node::Json(Value::Object(_)) => {
                    output.raw(b"{")?;
                    continue;
                }
                _ => unreachable!("scalar JSON nodes were normalized"),
            }
            depth -= 1;
            continue;
        }
        let (child, closing) = match node {
            Node::Array(values) => (values.get(frame.index).copied(), b"]"),
            Node::Json(Value::Array(values)) => (values.get(frame.index).map(Node::Json), b"]"),
            Node::Object(_) | Node::Json(Value::Object(_)) => {
                let entry = next_entry(node, frame.previous_key)?;
                if let Some((key, _)) = entry {
                    if frame.index > 0 {
                        output.raw(b",")?;
                    }
                    output.string(key)?;
                    output.raw(b":")?;
                    frame.previous_key = Some(key);
                }
                (entry.map(|(_, value)| value), b"}")
            }
            _ => unreachable!("scalar nodes are popped on entry"),
        };
        let Some(child) = child else {
            output.raw(closing)?;
            depth -= 1;
            continue;
        };
        if matches!(node, Node::Array(_) | Node::Json(Value::Array(_))) && frame.index > 0 {
            output.raw(b",")?;
        }
        frame.index += 1;
        let next_depth = depth.checked_add(1).ok_or(Error::DepthLimit)?;
        let bytes = check_push(next_depth, usage.nodes, limits, workspace.len())?;
        usage.nodes += 1;
        usage.depth = usage.depth.max(next_depth);
        usage.workspace_bytes = usage.workspace_bytes.max(bytes);
        workspace[depth] = Frame::new(normalized(child));
        depth = next_depth;
    }
    usage.bytes = output.bytes;
    Ok(usage)
}

#[cfg(test)]
#[path = "bounded_canonical_tests.rs"]
mod tests;
