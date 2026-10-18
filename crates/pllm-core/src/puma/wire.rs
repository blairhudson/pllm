//! Bounded peer codecs. Authentication and confidential transport belong to the
//! enclosing role session. Decoding alone never authenticates a peer.
//!
//! Encoded buffers retain their original one-use owner until zeroization and
//! deallocation. Their additional byte copies belong to the enclosing transport
//! allowance, separate from the primitive's 256 KiB logical-payload ceiling.

use super::{prss, Context, Error, Header, Message, MessagePhase, Party, MAX_LANES};
use zeroize::Zeroizing;

pub const MAX_FRAME_BYTES: usize = 128 + 2 * MAX_LANES * 8;

pub struct Encoded<T> {
    pub(crate) bytes: Zeroizing<Vec<u8>>,
    pub(crate) _owner: T,
}

impl<T> Encoded<T> {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) fn allocate(n: usize) -> Result<Zeroizing<Vec<u8>>, Error> {
    if n > MAX_FRAME_BYTES {
        return Err(Error::Bounds);
    }
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    if v.capacity() != n {
        return Err(Error::Allocation);
    }
    v.resize(n, 0);
    Ok(v)
}

fn party(n: u8) -> Result<Party, Error> {
    match n {
        0 => Ok(Party::P0),
        1 => Ok(Party::P1),
        2 => Ok(Party::P2),
        _ => Err(Error::Role),
    }
}

fn phase(n: u8) -> Result<MessagePhase, Error> {
    use MessagePhase::*;
    [
        Multiply,
        BooleanAnd,
        ArithmeticToBoolean,
        BooleanOtMask,
        BooleanOtZero,
        BooleanOtOne,
        BooleanOtReshare,
        BooleanPpaMask,
        BooleanPpaReveal,
        BooleanPpaRotate,
        MaskedInput,
        Correlation,
        Reshare,
    ]
    .get(n as usize)
    .copied()
    .ok_or(Error::Phase)
}

impl Message {
    pub fn encode(self) -> Result<Encoded<Self>, Error> {
        let bytes = self.encode_bytes()?;
        Ok(Encoded {
            bytes,
            _owner: self,
        })
    }

    pub(crate) fn encode_bytes(&self) -> Result<Zeroizing<Vec<u8>>, Error> {
        let mut bytes = allocate(44 + self.words.len() * 8)?;
        bytes[..4].copy_from_slice(b"PUM1");
        bytes[4..36].copy_from_slice(&self.header.binding);
        bytes[36] = self.header.from as u8;
        bytes[37] = self.header.to as u8;
        bytes[38] = self.header.phase as u8;
        bytes[40..44].copy_from_slice(&(self.words.len() as u32).to_le_bytes());
        for (out, value) in bytes[44..].chunks_exact_mut(8).zip(self.words.iter()) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Public envelope and length validation precede the fallible payload copy.
    /// The consuming runtime still checks the exact current operation/phase.
    pub fn decode(raw: &[u8], sender: Party, recipient: Party) -> Result<Self, Error> {
        if raw.len() < 44 || raw.len() > MAX_FRAME_BYTES || &raw[..4] != b"PUM1" || raw[39] != 0 {
            return Err(Error::Shape);
        }
        let n = u32::from_le_bytes(raw[40..44].try_into().unwrap()) as usize;
        if n == 0 || n > 2 * MAX_LANES || raw.len() != 44 + n * 8 {
            return Err(Error::Shape);
        }
        let header = Header {
            binding: raw[4..36].try_into().unwrap(),
            from: party(raw[36])?,
            to: party(raw[37])?,
            phase: phase(raw[38])?,
        };
        if header.from != sender || header.to != recipient || sender == recipient {
            return Err(Error::Role);
        }
        let mut words = super::zero_words(n)?;
        for (out, data) in words.iter_mut().zip(raw[44..].chunks_exact(8)) {
            *out = u64::from_le_bytes(data.try_into().unwrap());
        }
        Ok(Self { header, words })
    }
}

impl prss::PublicContribution {
    pub fn encode(self) -> [u8; 93] {
        let mut out = [0; 93];
        out[..4].copy_from_slice(b"PUP1");
        out[4..36].copy_from_slice(&self.context.session);
        out[36..68].copy_from_slice(&self.context.schedule);
        out[68..76].copy_from_slice(&self.context.phase.to_le_bytes());
        out[76] = self.from as u8;
        out[77..].copy_from_slice(&self.value.to_le_bytes());
        out
    }

    pub fn decode(raw: &[u8], sender: Party) -> Result<Self, Error> {
        if raw.len() != 93 || &raw[..4] != b"PUP1" {
            return Err(Error::Shape);
        }
        let from = party(raw[76])?;
        if from != sender {
            return Err(Error::Role);
        }
        Ok(Self {
            context: Context::new(
                raw[4..36].try_into().unwrap(),
                raw[36..68].try_into().unwrap(),
                u64::from_le_bytes(raw[68..76].try_into().unwrap()),
            )?,
            from,
            value: u128::from_le_bytes(raw[77..].try_into().unwrap()),
        })
    }
}

impl prss::SeedMessage {
    pub fn encode(self) -> Result<Encoded<Self>, Error> {
        let mut bytes = allocate(54)?;
        bytes[..4].copy_from_slice(b"PUS1");
        bytes[4..36].copy_from_slice(&self.setup);
        bytes[36] = self.from as u8;
        bytes[37] = self.to as u8;
        bytes[38..].copy_from_slice(&*self.seed);
        Ok(Encoded {
            bytes,
            _owner: self,
        })
    }

    pub fn decode(raw: &[u8], sender: Party, recipient: Party) -> Result<Self, Error> {
        if raw.len() != 54 || &raw[..4] != b"PUS1" {
            return Err(Error::Shape);
        }
        let from = party(raw[36])?;
        let to = party(raw[37])?;
        if from != sender || to != recipient || from.next().next() != to {
            return Err(Error::Role);
        }
        Ok(Self {
            setup: raw[4..36].try_into().unwrap(),
            from,
            to,
            seed: Zeroizing::new(raw[38..].try_into().unwrap()),
        })
    }
}
