//! Exact, bounded clear-slot algebra for distinct interleaved K and distributed V.
//!
//! Source: `papers/cachemir.pdf`, SHA-256
//! `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`,
//! §§5.1–5.4, pp.6–8, Figures 5–6; contract §6.2 in
//! `paper/methods/encrypted-computation-contracts.md`.
//! This is NOT encrypted state, CKKS, a numerical/security profile, a decoder,
//! encrypted prefill, a rotation-cost implementation, or a refresh policy.
//! Appendix C rectangular VMM and Appendix D RoPE are deliberately absent.
//!
//! Logical vectors are `[head, feature]`, with equal query/KV head counts;
//! no implicit GQA expansion, batch, layer, or checkpoint binding is supplied.
//! Caller buffers hold flat `[ciphertext, slot]` integers, not ciphertexts.
//! Unlike `ckks_layout`, this does not admit cryptographic parameters. Like
//! `coefficient_layout`, it borrows exact caller-sized buffers, allocates nothing,
//! and rejects every error before mutation. Semantic cache dimensions remain
//! distinct from this physical packing (see `kv_cache`'s head/capacity axes).

pub const MAX_SLOTS: usize = 32768;
pub const MAX_CAPACITY: usize = 32768;
pub const MAX_STORAGE_WORDS: usize = 16 * 1024 * 1024;
pub const MAX_READ_WORK: usize = 16 * 1024 * 1024;
/// Exact test-domain bound, not a paper approximation range.
pub const MAX_ABS_INPUT: i64 = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    Geometry,
    Overflow,
    ResourceLimit,
    BufferLength,
    NonZeroStorage,
    InputRange,
    Capacity,
    Index,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Address {
    pub ciphertext: usize,
    pub slot: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    slots: usize,
    heads: usize,
    head_dim: usize,
    capacity: usize,
    width: usize,
    t: usize,
    group_tokens: usize,
}

fn product(a: usize, b: usize) -> Result<usize, LayoutError> {
    a.checked_mul(b).ok_or(LayoutError::Overflow)
}

impl Geometry {
    /// `width = heads * head_dim` is the paper's d, not per-head width.
    /// Power-of-two slots/width/head counts cover the source packing algorithm;
    /// nondivisible or padded/rectangular interpretations are not guessed.
    pub fn new(
        slots: usize,
        heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, LayoutError> {
        let width = product(heads, head_dim)?;
        if !slots.is_power_of_two()
            || !heads.is_power_of_two()
            || !head_dim.is_power_of_two()
            || width == 0
            || width > slots
            || slots % width != 0
            || capacity == 0
        {
            return Err(LayoutError::Geometry);
        }
        if slots > MAX_SLOTS || capacity > MAX_CAPACITY {
            return Err(LayoutError::ResourceLimit);
        }
        let g = Self {
            slots,
            heads,
            head_dim,
            capacity,
            width,
            t: slots / width,
            group_tokens: slots / heads,
        };
        let k = g.key_layout().storage_words()?;
        let v = g.value_layout().storage_words()?;
        if k.checked_add(v).ok_or(LayoutError::Overflow)? > MAX_STORAGE_WORDS
            || v > MAX_READ_WORK
            || product(capacity, width)? > MAX_READ_WORK
        {
            return Err(LayoutError::ResourceLimit);
        }
        Ok(g)
    }

    pub fn slots(self) -> usize {
        self.slots
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn head_dim(self) -> usize {
        self.head_dim
    }
    pub fn width(self) -> usize {
        self.width
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn tokens_per_key_tile(self) -> usize {
        self.t
    }
    pub fn tokens_per_value_group(self) -> usize {
        self.group_tokens
    }
    pub fn key_layout(self) -> KeyLayout {
        KeyLayout(self)
    }
    pub fn value_layout(self) -> ValueLayout {
        ValueLayout(self)
    }

    fn check_index(self, token: usize, head: usize, feature: usize) -> Result<(), LayoutError> {
        if token >= self.capacity || head >= self.heads || feature >= self.head_dim {
            return Err(LayoutError::Index);
        }
        Ok(())
    }

    // Embed one head-local slot in head-interleaved blocks of t, not a contiguous
    // head segment. Source §5.3's two-head example fixes this ordering.
    fn embed(self, local: usize, head: usize) -> usize {
        (local / self.t * self.heads + head) * self.t + local % self.t
    }

    pub fn score_words(self) -> usize {
        self.capacity.div_ceil(self.group_tokens) * self.slots
    }

    /// Compact score blocks, preserving t-slot head interleaving. Blocks beyond
    /// S/H are merged by reads, never normalized independently or truncated.
    pub fn score_address(self, token: usize, head: usize) -> Result<Address, LayoutError> {
        self.check_index(token, head, 0)?;
        Ok(Address {
            ciphertext: token / self.group_tokens,
            slot: self.embed(token % self.group_tokens, head),
        })
    }

    pub fn output_slot(self, head: usize, feature: usize) -> Result<usize, LayoutError> {
        self.check_index(0, head, feature)?;
        Ok(self.embed(feature * self.t, head))
    }
}

/// Retained K: t tokens per tile, all heads in each tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyLayout(Geometry);

impl KeyLayout {
    pub fn ciphertexts(self, active: usize) -> Result<usize, LayoutError> {
        if active > self.0.capacity {
            return Err(LayoutError::Capacity);
        }
        Ok(active.div_ceil(self.0.t))
    }

    pub fn storage_words(self) -> Result<usize, LayoutError> {
        product(self.ciphertexts(self.0.capacity)?, self.0.slots)
    }

    /// Single head: slot[f*t+r] = K[j*t+r,f]. Multiple heads reorder
    /// feature blocks as [f0h0, f0h1, ..., f1h0, ...].
    pub fn address(
        self,
        token: usize,
        head: usize,
        feature: usize,
    ) -> Result<Address, LayoutError> {
        let g = self.0;
        g.check_index(token, head, feature)?;
        Ok(Address {
            ciphertext: token / g.t,
            slot: g.embed(feature * g.t + token % g.t, head),
        })
    }

    pub fn validity_mask(self, active: usize, mask: &mut [u8]) -> Result<(), LayoutError> {
        self.ciphertexts(active)?;
        if mask.len() != self.storage_words()? {
            return Err(LayoutError::BufferLength);
        }
        let g = self.0;
        for (i, value) in mask.iter_mut().enumerate() {
            let token = i / g.slots * g.t + i % g.t;
            *value = u8::from(token < active);
        }
        Ok(())
    }
}

/// Retained V: head_dim ciphertexts per S/H-token group, NOT K tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueLayout(Geometry);

impl ValueLayout {
    pub fn ciphertexts(self, active: usize) -> Result<usize, LayoutError> {
        let g = self.0;
        if active > g.capacity {
            return Err(LayoutError::Capacity);
        }
        product(active.div_ceil(g.group_tokens), g.head_dim)
    }

    pub fn storage_words(self) -> Result<usize, LayoutError> {
        product(self.ciphertexts(self.0.capacity)?, self.0.slots)
    }

    /// Figure 6 reconstructed explicitly: in one head, V ciphertext c,
    /// slot x stores V[(x+c*t) mod (S/H), floor(x/t)]. Thus ct0 has
    /// tokens 0,1 at feature0, tokens 2,3 at feature1 when t=2; ct1
    /// shifts token indices by 2 with wraparound. Inverting gives c below.
    /// Multi-head embedding applies independently to these head-local slots.
    pub fn address(
        self,
        token: usize,
        head: usize,
        feature: usize,
    ) -> Result<Address, LayoutError> {
        let g = self.0;
        g.check_index(token, head, feature)?;
        let local = token % g.group_tokens;
        let c = (local / g.t + g.head_dim - feature) % g.head_dim;
        Ok(Address {
            ciphertext: token / g.group_tokens * g.head_dim + c,
            slot: g.embed(feature * g.t + local % g.t, head),
        })
    }

    pub fn validity_mask(self, active: usize, mask: &mut [u8]) -> Result<(), LayoutError> {
        self.ciphertexts(active)?;
        if mask.len() != self.storage_words()? {
            return Err(LayoutError::BufferLength);
        }
        let g = self.0;
        for (i, value) in mask.iter_mut().enumerate() {
            let ct = i / g.slots;
            let slot = i % g.slots;
            let feature = slot / g.t / g.heads;
            let local = feature * g.t + slot % g.t;
            let token =
                ct / g.head_dim * g.group_tokens + (local + ct % g.head_dim * g.t) % g.group_tokens;
            *value = u8::from(token < active);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainedDescriptor {
    pub geometry: Geometry,
    pub active_tokens: usize,
    pub allocated_key_ciphertexts: usize,
    pub allocated_value_ciphertexts: usize,
    pub live_key_ciphertexts: usize,
    pub live_value_ciphertexts: usize,
}

/// Logical allocation delta; caller storage is reserved at construction, while
/// these counts describe which previously empty ciphertext-shaped tiles go live.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendDescriptor {
    pub token: usize,
    pub key_tile: usize,
    pub key_lane: usize,
    pub new_key_ciphertexts: usize,
    pub value_group: usize,
    pub new_value_ciphertexts: usize,
}

/// Exclusive borrow enforces separate K/V storage and no overlapping inputs.
/// Construction admits only zero storage; no unvalidated retained imports.
pub struct RetainedCache<'a> {
    geometry: Geometry,
    active: usize,
    keys: &'a mut [i64],
    values: &'a mut [i64],
}

fn input_range(input: &[i64]) -> Result<(), LayoutError> {
    if input
        .iter()
        .any(|&x| !(-MAX_ABS_INPUT..=MAX_ABS_INPUT).contains(&x))
    {
        return Err(LayoutError::InputRange);
    }
    Ok(())
}

fn offset(g: Geometry, a: Address) -> usize {
    a.ciphertext * g.slots + a.slot
}

impl<'a> RetainedCache<'a> {
    pub fn new(
        geometry: Geometry,
        keys: &'a mut [i64],
        values: &'a mut [i64],
    ) -> Result<Self, LayoutError> {
        if keys.len() != geometry.key_layout().storage_words()?
            || values.len() != geometry.value_layout().storage_words()?
        {
            return Err(LayoutError::BufferLength);
        }
        if keys.iter().chain(values.iter()).any(|&v| v != 0) {
            return Err(LayoutError::NonZeroStorage);
        }
        Ok(Self {
            geometry,
            active: 0,
            keys,
            values,
        })
    }

    pub fn descriptor(&self) -> RetainedDescriptor {
        let g = self.geometry;
        RetainedDescriptor {
            geometry: g,
            active_tokens: self.active,
            allocated_key_ciphertexts: g.capacity.div_ceil(g.t),
            allocated_value_ciphertexts: g.capacity.div_ceil(g.group_tokens) * g.head_dim,
            live_key_ciphertexts: self.active.div_ceil(g.t),
            live_value_ciphertexts: self.active.div_ceil(g.group_tokens) * g.head_dim,
        }
    }

    pub fn append_descriptor(&self) -> Result<AppendDescriptor, LayoutError> {
        let g = self.geometry;
        if self.active == g.capacity {
            return Err(LayoutError::Capacity);
        }
        Ok(AppendDescriptor {
            token: self.active,
            key_tile: self.active / g.t,
            key_lane: self.active % g.t,
            new_key_ciphertexts: usize::from(self.active % g.t == 0),
            value_group: self.active / g.group_tokens,
            new_value_ciphertexts: if self.active % g.group_tokens == 0 {
                g.head_dim
            } else {
                0
            },
        })
    }

    /// Atomically append one [head,feature] K/V pair into previously empty slots.
    /// No repacking of old tokens, allocations, or reads of clear model state.
    pub fn append(&mut self, key: &[i64], value: &[i64]) -> Result<(), LayoutError> {
        let g = self.geometry;
        if self.active == g.capacity {
            return Err(LayoutError::Capacity);
        }
        if key.len() != g.width || value.len() != g.width {
            return Err(LayoutError::BufferLength);
        }
        input_range(key)?;
        input_range(value)?;
        for h in 0..g.heads {
            for f in 0..g.head_dim {
                // Geometry and capacity are admitted before mutation. These
                // infallible addresses cannot fail in this bounded loop.
                let k = g.key_layout().address(self.active, h, f).unwrap();
                let v = g.value_layout().address(self.active, h, f).unwrap();
                self.keys[offset(g, k)] = key[h * g.head_dim + f];
                self.values[offset(g, v)] = value[h * g.head_dim + f];
            }
        }
        self.active += 1;
        Ok(())
    }

    /// Derived views borrow retained slots immutably. No retained-level mutation
    /// is expressible; actual ciphertext copy/level/error policy remains absent.
    pub fn derived_view(&self) -> DerivedView<'_> {
        DerivedView {
            retained: self.descriptor(),
            keys: self.keys,
            values: self.values,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedDescriptor {
    pub retained: RetainedDescriptor,
    pub score_words: usize,
    pub output_words: usize,
}

pub struct DerivedView<'a> {
    retained: RetainedDescriptor,
    keys: &'a [i64],
    values: &'a [i64],
}

impl DerivedView<'_> {
    pub fn descriptor(&self) -> DerivedDescriptor {
        let g = self.retained.geometry;
        DerivedDescriptor {
            retained: self.retained,
            score_words: g.score_words(),
            output_words: g.slots,
        }
    }

    pub fn key_slots(&self) -> &[i64] {
        self.keys
    }
    pub fn value_slots(&self) -> &[i64] {
        self.values
    }

    /// Executable exact QK: replicate query conceptually into K's lanes, reduce
    /// per-head features, and compact one score copy into head-interleaved blocks.
    /// All padding is zero, even for empty/partial tiles and final score blocks.
    pub fn query_key(&self, query: &[i64], scores: &mut [i64]) -> Result<(), LayoutError> {
        let g = self.retained.geometry;
        if query.len() != g.width || scores.len() != g.score_words() {
            return Err(LayoutError::BufferLength);
        }
        input_range(query)?;
        // |dot| <= MAX_SLOTS * 2^40 <= 2^55: i64 cannot overflow.
        scores.fill(0);
        for token in 0..self.retained.active_tokens {
            for h in 0..g.heads {
                let mut sum = 0;
                for f in 0..g.head_dim {
                    let a = g.key_layout().address(token, h, f).unwrap();
                    sum += query[h * g.head_dim + f] * self.keys[offset(g, a)];
                }
                scores[offset(g, g.score_address(token, h).unwrap())] = sum;
            }
        }
        Ok(())
    }

    /// Consume caller-supplied compact attention weights (NOT softmax). Execute
    /// Figure 6's distributed diagonal product and within-t reduction directly.
    /// For each physical V slot, the matching score is shifted by c*t within
    /// its head-local group; accumulated groups share one interleaved output.
    /// Invalid score slots are ignored, preventing padded values entering sums.
    pub fn attention_value(&self, weights: &[i64], output: &mut [i64]) -> Result<(), LayoutError> {
        let g = self.retained.geometry;
        if weights.len() != g.score_words() || output.len() != g.slots {
            return Err(LayoutError::BufferLength);
        }
        input_range(weights)?;
        output.fill(0);
        let groups = self.retained.active_tokens.div_ceil(g.group_tokens);
        for group in 0..groups {
            for c in 0..g.head_dim {
                for f in 0..g.head_dim {
                    for h in 0..g.heads {
                        for r in 0..g.t {
                            let local = (f * g.t + r + c * g.t) % g.group_tokens;
                            let token = group * g.group_tokens + local;
                            if token < self.retained.active_tokens {
                                let v =
                                    (group * g.head_dim + c) * g.slots + g.embed(f * g.t + r, h);
                                let s = group * g.slots + g.embed(local, h);
                                output[g.embed(f * g.t, h)] += weights[s] * self.values[v];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "interleaved_kv_layout_tests.rs"]
mod tests;
