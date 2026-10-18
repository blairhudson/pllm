use super::*;

enum State {
    Ready,
    Open {
        operation: Operation,
        own_open: Words,
        output_component: Words,
    },
    Reshare {
        operation: Operation,
        own_reshare: Words,
        output_component: Words,
    },
    Complete(ReplicatedShare),
    Retired,
}

/// Party-local scalar CPU protocol. One operation may be live at a time.
/// Every state-changing failure retires the runtime and wipes retained buffers;
/// cancellation is idempotent. Output extraction is one-use. No restart API can
/// rewind the consumed sequence counter.
pub struct PartyRuntime {
    context: Context,
    party: Party,
    limits: Limits,
    next_sequence: u64,
    state: State,
    payloads: PayloadUsage,
}

impl PartyRuntime {
    pub fn new(context: Context, party: Party, limits: Limits) -> Self {
        Self {
            context,
            party,
            limits,
            next_sequence: 0,
            state: State::Ready,
            payloads: PayloadUsage::new(limits.max_bytes),
        }
    }

    pub fn status(&self) -> Status {
        match self.state {
            State::Ready => Status::Ready,
            State::Open { .. } => Status::AwaitingOpen,
            State::Reshare { .. } => Status::AwaitingReshare,
            State::Complete(_) => Status::Complete,
            State::Retired => Status::Retired,
        }
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }
    pub fn payload_usage(&self) -> PayloadUsage {
        self.payloads.clone()
    }
    pub fn cancel(&mut self) {
        self.state = State::Retired;
    }

    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }

    /// Burns the ordinal before allocating or using correlated material.
    /// A/B send one MaskedInput frame to one another; H sends one Correlation
    /// frame to each evaluator. Only H draws fresh private splitting randomness.
    pub fn begin(
        &mut self,
        operation: Operation,
        input: ReplicatedShare,
        material: LocalMaterial,
    ) -> Result<[Option<Message>; 2], Error> {
        self.begin_with_randomness(operation, input, material, &mut OsEntropy)
    }

    pub(super) fn begin_with_randomness(
        &mut self,
        operation: Operation,
        input: ReplicatedShare,
        material: LocalMaterial,
        entropy: &mut impl Entropy,
    ) -> Result<[Option<Message>; 2], Error> {
        let result = self.begin_inner(operation, input, material, entropy);
        self.checked(result)
    }

    fn begin_inner(
        &mut self,
        operation: Operation,
        mut input: ReplicatedShare,
        mut material: LocalMaterial,
        entropy: &mut impl Entropy,
    ) -> Result<[Option<Message>; 2], Error> {
        if !matches!(self.state, State::Ready) {
            return Err(if self.status() == Status::Retired {
                Error::Retired
            } else {
                Error::Phase
            });
        }
        if operation.context != self.context
            || operation.sequence != self.next_sequence
            || operation.sequence >= self.limits.max_operations
            || input.operation != operation
            || material.operation != operation
        {
            return Err(Error::Binding);
        }
        if input.party != self.party || material.party != self.party {
            return Err(Error::Role);
        }
        self.next_sequence += 1;
        self.limits.admit(operation.lanes)?;
        input.charge_to(&self.payloads)?;
        material.charge_to(&self.payloads)?;
        let n = operation.lanes;
        match material.data {
            MaterialData::Evaluator { mask, output } => {
                let a = self.party == operation.evaluator_a();
                let peer = if a {
                    operation.evaluator_b()
                } else {
                    operation.evaluator_a()
                };
                let mut opening = self.payloads.zeros(n)?;
                for i in 0..n {
                    let x = if a {
                        input.first[i]
                            .wrapping_add(input.second[i])
                            .wrapping_add(1 << 62)
                    } else {
                        input.second[i]
                    };
                    opening[i] = x.wrapping_add(mask[i]);
                }
                let message = Message {
                    header: header(operation, self.party, peer, MessagePhase::MaskedInput),
                    words: self.payloads.copy(&opening, n)?,
                };
                self.state = State::Open {
                    operation,
                    own_open: opening,
                    output_component: output,
                };
                Ok([Some(message), None])
            }
            MaterialData::Helper {
                mask_a,
                mask_b,
                split,
                output_a,
                output_b,
            } => {
                if self.party != operation.helper() {
                    return Err(Error::Role);
                }
                // Source fillPriv is ONE contiguous 2*n-word call: rb first,
                // rc second. A source PRG adapter must not fill lane by lane.
                // Source PRSS already issued this before its output-mask fills.
                let mut ca = match split {
                    Some(words) => words,
                    None => {
                        let mut words = self.payloads.zeros(2 * n)?;
                        entropy.fill(&mut words)?;
                        words
                    }
                };
                let mut cb = self.payloads.zeros(2 * n)?;
                for i in 0..n {
                    let r = mask_a[i].wrapping_add(mask_b[i]);
                    cb[i] = 0_u64.wrapping_sub(ca[i]);
                    cb[n + i] = 0_u64.wrapping_sub(ca[n + i]);
                    ca[i] = ca[i].wrapping_add(r >> 63);
                    ca[n + i] = ca[n + i].wrapping_add((r & (u64::MAX >> 1)) >> operation.shift);
                }
                self.state = State::Complete(ReplicatedShare {
                    operation,
                    party: self.party,
                    first: output_b,
                    second: output_a,
                });
                Ok([
                    Some(Message {
                        header: header(
                            operation,
                            self.party,
                            operation.evaluator_a(),
                            MessagePhase::Correlation,
                        ),
                        words: ca,
                    }),
                    Some(Message {
                        header: header(
                            operation,
                            self.party,
                            operation.evaluator_b(),
                            MessagePhase::Correlation,
                        ),
                        words: cb,
                    }),
                ])
            }
        }
    }

    /// Round barrier: supply exactly the peer opening and helper correction.
    /// H cannot enter this phase; it never receives/reconstructs C.
    pub fn open(&mut self, peer_open: Message, correlation: Message) -> Result<Message, Error> {
        let result = self.open_inner(peer_open, correlation);
        self.checked(result)
    }

    fn open_inner(
        &mut self,
        mut peer_open: Message,
        mut correlation: Message,
    ) -> Result<Message, Error> {
        let state = std::mem::replace(&mut self.state, State::Retired);
        let State::Open {
            operation,
            own_open,
            output_component,
        } = state
        else {
            return Err(Error::Phase);
        };
        let a = self.party == operation.evaluator_a();
        let peer = if a {
            operation.evaluator_b()
        } else {
            operation.evaluator_a()
        };
        peer_open.validate(operation, peer, self.party, MessagePhase::MaskedInput)?;
        correlation.validate(
            operation,
            operation.helper(),
            self.party,
            MessagePhase::Correlation,
        )?;
        peer_open.words.charge_to(&self.payloads)?;
        correlation.words.charge_to(&self.payloads)?;
        let n = operation.lanes;
        let mut share = self.payloads.zeros(n)?;
        for i in 0..n {
            let c = own_open[i].wrapping_add(peer_open.words[i]);
            let msb = c >> 63;
            let rb = correlation.words[i];
            let rc = correlation.words[n + i];
            let b = rb
                .wrapping_add(if a { msb } else { 0 })
                .wrapping_sub(msb.wrapping_mul(rb).wrapping_mul(2));
            let mut y = b
                .wrapping_shl(u32::from(63 - operation.shift))
                .wrapping_sub(rc);
            if a {
                y = y
                    .wrapping_add((c & (u64::MAX >> 1)) >> operation.shift)
                    .wrapping_sub(1_u64 << (62 - operation.shift));
            }
            share[i] = y.wrapping_sub(output_component[i]);
        }
        let message = Message {
            header: header(operation, self.party, peer, MessagePhase::Reshare),
            words: self.payloads.copy(&share, n)?,
        };
        self.state = State::Reshare {
            operation,
            own_reshare: share,
            output_component,
        };
        Ok(message)
    }

    pub fn reshare(&mut self, peer_share: Message) -> Result<(), Error> {
        let result = self.reshare_inner(peer_share);
        self.checked(result)
    }

    fn reshare_inner(&mut self, mut peer_share: Message) -> Result<(), Error> {
        let state = std::mem::replace(&mut self.state, State::Retired);
        let State::Reshare {
            operation,
            mut own_reshare,
            output_component,
        } = state
        else {
            return Err(Error::Phase);
        };
        let a = self.party == operation.evaluator_a();
        let peer = if a {
            operation.evaluator_b()
        } else {
            operation.evaluator_a()
        };
        peer_share.validate(operation, peer, self.party, MessagePhase::Reshare)?;
        peer_share.words.charge_to(&self.payloads)?;
        for i in 0..operation.lanes {
            own_reshare[i] = own_reshare[i].wrapping_add(peer_share.words[i]);
        }
        let (first, second) = if a {
            (output_component, own_reshare)
        } else {
            (own_reshare, output_component)
        };
        self.state = State::Complete(ReplicatedShare {
            operation,
            party: self.party,
            first,
            second,
        });
        Ok(())
    }

    pub fn take_output(&mut self) -> Result<ReplicatedShare, Error> {
        let state = std::mem::replace(&mut self.state, State::Retired);
        if let State::Complete(output) = state {
            self.state = State::Ready;
            Ok(output)
        } else {
            Err(Error::Phase)
        }
    }
}

impl Drop for PartyRuntime {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn header(operation: Operation, from: Party, to: Party, phase: MessagePhase) -> Header {
    Header {
        binding: operation.digest(),
        from,
        to,
        phase,
    }
}

pub(super) trait Entropy {
    fn fill(&mut self, words: &mut [u64]) -> Result<(), Error>;
}
struct OsEntropy;
impl Entropy for OsEntropy {
    fn fill(&mut self, words: &mut [u64]) -> Result<(), Error> {
        let mut bytes = Zeroizing::new([0u8; 8]);
        for word in words {
            getrandom::fill(&mut *bytes).map_err(|_| Error::Entropy)?;
            *word = u64::from_le_bytes(*bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod entropy_tests {
    use super::*;

    #[test]
    fn helper_entropy_failure_burns_pending_material() {
        struct Fails {
            remaining: usize,
        }
        impl Entropy for Fails {
            fn fill(&mut self, words: &mut [u64]) -> Result<(), Error> {
                for word in words {
                    if self.remaining == 0 {
                        return Err(Error::Entropy);
                    }
                    self.remaining -= 1;
                    *word = 0x123456789abcdef0;
                }
                Ok(())
            }
        }
        let limits = Limits::default();
        let context = Context::new([1; 32], [2; 32], 3).unwrap();
        let op = Operation::new(context, 0, Party::P0, 1, 18, limits).unwrap();
        let input = ReplicatedShare::from_local(op, Party::P2, &[0], &[0], limits).unwrap();
        let material = LocalMaterial::helper(op, &[3], &[4], &[5], &[6], limits).unwrap();
        let mut runtime = PartyRuntime::new(context, Party::P2, limits);
        // Failure after one private word was drawn: no partial material escapes.
        let result = runtime.begin_inner(op, input, material, &mut Fails { remaining: 1 });
        assert!(matches!(runtime.checked(result), Err(Error::Entropy)));
        assert_eq!(runtime.status(), Status::Retired);
        assert_eq!(runtime.next_sequence(), 1);
    }
}
