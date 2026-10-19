//! Small fallibly allocated, canonical ASCII/integer JSON reader. No serde DOM.
use super::{reserve, Error, Result, MAX_DESCRIPTOR_BYTES};

pub(super) enum Value {
    Int(i64),
    Bool(bool),
    String(Vec<u8>),
    Array(Vec<Value>),
    Object(Vec<(Vec<u8>, Value)>),
}

#[cfg(test)]
impl Value {
    // Sum allocated vector capacities, including inline metadata cells. Every
    // parser allocation is monotonically retained in this DOM. During a single
    // Vec growth, an old block can coexist with the new block; its capacity is
    // at most the final largest block. No generic parser/RSS claim is implied.
    pub(super) fn logical_heap(&self) -> (usize, usize) {
        fn combine(parts: impl Iterator<Item = (usize, usize)>, own: usize) -> (usize, usize) {
            parts.fold((own, own), |(total, largest), (bytes, block)| {
                (total + bytes, largest.max(block))
            })
        }
        match self {
            Self::Int(_) | Self::Bool(_) => (0, 0),
            Self::String(v) => (v.capacity(), v.capacity()),
            Self::Array(v) => combine(
                v.iter().map(Self::logical_heap),
                v.capacity() * std::mem::size_of::<Value>(),
            ),
            Self::Object(v) => combine(
                v.iter().map(|(key, value)| {
                    let (bytes, largest) = value.logical_heap();
                    (key.capacity() + bytes, key.capacity().max(largest))
                }),
                v.capacity() * std::mem::size_of::<(Vec<u8>, Value)>(),
            ),
        }
    }
}

impl Value {
    pub fn int(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            _ => Err(Error::Type),
        }
    }
    pub fn uint(&self) -> Result<u64> {
        u64::try_from(self.int()?).map_err(|_| Error::Range)
    }
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(v) => std::str::from_utf8(v).map_err(|_| Error::Encoding),
            _ => Err(Error::Type),
        }
    }
    pub fn boolean(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(Error::Type),
        }
    }
    pub fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(v) => Ok(v),
            _ => Err(Error::Type),
        }
    }
    pub fn fields(&self, expected: &[&str]) -> Result<()> {
        match self {
            Self::Object(v)
                if v.len() == expected.len()
                    && v.iter().zip(expected).all(|((k, _), e)| k == e.as_bytes()) =>
            {
                Ok(())
            }
            _ => Err(Error::Metadata),
        }
    }
    pub fn get(&self, key: &str) -> Result<&Value> {
        match self {
            Self::Object(v) => v
                .iter()
                .find(|(k, _)| k == key.as_bytes())
                .map(|(_, v)| v)
                .ok_or(Error::Metadata),
            _ => Err(Error::Type),
        }
    }
    pub fn eq_string(&self, expected: &str) -> Result<()> {
        if self.string()? == expected {
            Ok(())
        } else {
            Err(Error::Metadata)
        }
    }
}

pub(super) fn parse(bytes: &[u8]) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > MAX_DESCRIPTOR_BYTES {
        return Err(Error::DescriptorBytes);
    }
    let mut p = Parser {
        bytes,
        pos: 0,
        nodes: 0,
    };
    let value = p.value(0)?;
    if p.pos != bytes.len() {
        return Err(Error::Encoding);
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    nodes: usize,
}
impl Parser<'_> {
    fn take(&mut self, byte: u8) -> Result<()> {
        if self.bytes.get(self.pos) != Some(&byte) {
            return Err(Error::Encoding);
        }
        self.pos += 1;
        Ok(())
    }
    fn string(&mut self) -> Result<Vec<u8>> {
        self.take(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.bytes.get(self.pos).ok_or(Error::Encoding)?;
            self.pos += 1;
            if c == b'"' {
                return Ok(out);
            }
            let c = match c {
                b'\\' => {
                    let c = *self.bytes.get(self.pos).ok_or(Error::Encoding)?;
                    self.pos += 1;
                    match c {
                        b'"' | b'\\' => c,
                        b'b' => 8,
                        b'f' => 12,
                        b'n' => 10,
                        b'r' => 13,
                        b't' => 9,
                        b'u' => {
                            self.take(b'0')?;
                            self.take(b'0')?;
                            let mut n = 0u8;
                            for _ in 0..2 {
                                let c = *self.bytes.get(self.pos).ok_or(Error::Encoding)?;
                                self.pos += 1;
                                n = n.checked_mul(16).ok_or(Error::Encoding)?
                                    + match c {
                                        b'0'..=b'9' => c - b'0',
                                        b'a'..=b'f' => c - b'a' + 10,
                                        _ => return Err(Error::Encoding),
                                    };
                            }
                            if !(n < 32 || n == 127) || [8, 9, 10, 12, 13].contains(&n) {
                                return Err(Error::Encoding);
                            }
                            n
                        }
                        _ => return Err(Error::Encoding),
                    }
                }
                32..=126 => c,
                _ => return Err(Error::Encoding),
            };
            reserve(&mut out, 1)?;
            out.push(c);
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        // Source expressions have at most depth 12; envelope adds six levels.
        if depth > 20 || self.nodes >= 50_000 {
            return Err(Error::ExpressionBudget);
        }
        self.nodes += 1;
        match self.bytes.get(self.pos).copied().ok_or(Error::Encoding)? {
            b'"' => Ok(Value::String(self.string()?)),
            b'[' => {
                self.pos += 1;
                let mut values = Vec::new();
                if self.bytes.get(self.pos) != Some(&b']') {
                    loop {
                        let v = self.value(depth + 1)?;
                        reserve(&mut values, 1)?;
                        values.push(v);
                        if self.bytes.get(self.pos) != Some(&b',') {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                self.take(b']')?;
                Ok(Value::Array(values))
            }
            b'{' => {
                self.pos += 1;
                let mut values: Vec<(Vec<u8>, Value)> = Vec::new();
                if self.bytes.get(self.pos) != Some(&b'}') {
                    loop {
                        let key = self.string()?;
                        if values.last().is_some_and(|(k, _)| *k >= key) {
                            return Err(Error::Encoding);
                        }
                        self.take(b':')?;
                        let v = self.value(depth + 1)?;
                        reserve(&mut values, 1)?;
                        values.push((key, v));
                        if self.bytes.get(self.pos) != Some(&b',') {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                self.take(b'}')?;
                Ok(Value::Object(values))
            }
            b't' | b'f' => {
                let yes = self.bytes[self.pos] == b't';
                for &c in if yes {
                    b"true".as_slice()
                } else {
                    b"false".as_slice()
                } {
                    self.take(c)?;
                }
                Ok(Value::Bool(yes))
            }
            b'-' | b'0'..=b'9' => {
                let start = self.pos;
                let negative = self.bytes[self.pos] == b'-';
                if negative {
                    self.pos += 1;
                }
                let first = *self.bytes.get(self.pos).ok_or(Error::Encoding)?;
                if !first.is_ascii_digit() {
                    return Err(Error::Encoding);
                }
                self.pos += 1;
                while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
                let token = &self.bytes[start..self.pos];
                if token.len() > 17
                    || (first == b'0' && (token.len() > 1 + usize::from(negative) || negative))
                {
                    return Err(Error::Encoding);
                }
                let n = std::str::from_utf8(token)
                    .map_err(|_| Error::Encoding)?
                    .parse()
                    .map_err(|_| Error::Range)?;
                Ok(Value::Int(n))
            }
            _ => Err(Error::Encoding),
        }
    }
}
