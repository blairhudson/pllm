//! Fixed signed39 comparison -> Select50 maximum. No clear maximum/winner,
//! no generic fanout, and no widened view of a masked signed39 result.
use super::*;
use ledger::Read;

const MASK50: u64 = (1u64 << 50) - 1;
const MASK39: u64 = (1u64 << 39) - 1;

pub(super) struct Bindings {
    pub(super) reads: [[Binding; 8]; 2],
    pub(super) maxima: [Binding; 2],
    pub(super) scores: [Binding; 3],
}
pub(super) struct DealerFor<C: AllocationContext> {
    root: DealerMaskFor<C, Ring>,
    #[allow(dead_code)] // Retain actual dealer material until outer mask retirement.
    comparison: [Option<arithmetic::OutputMasksFor<C>>; 2],
    #[allow(dead_code)] // Retain actual dealer material until outer mask retirement.
    selection: [Option<selection::OutputMasksFor<C>>; 2],
    pub(super) maximum_mask: u64,
}
impl<C: AllocationContext> Drop for DealerFor<C> {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.maximum_mask.zeroize();
    }
}
pub(super) struct PartyFor<C: AllocationContext> {
    root: CommonMaskedFor<C, Ring>,
    comparison: [Option<arithmetic::PartyFor<C>>; 2],
    selection: [Option<selection::PartyFor<C>>; 2],
    predicate: Option<arithmetic::MaskedOutputFor<C>>,
    selected: Option<selection::MaskedOutputFor<C>>,
    maximum: u64,
    difference: u64,
    projected: u64,
    update: usize,
    phase: u8,
    count: usize,
    active: [u8; 3],
    bindings: Bindings,
}
impl Clone for Bindings {
    fn clone(&self) -> Self {
        Self {
            reads: self.reads,
            maxima: self.maxima,
            scores: self.scores,
        }
    }
}
#[allow(clippy::large_enum_variant)]
pub(super) enum Frame<C: AllocationContext> {
    Compare(arithmetic::FrameFor<C>),
    Select(selection::FrameFor<C>),
}
impl<C: AllocationContext> Frame<C> {
    pub(super) fn bytes(&self) -> &[u8] {
        match self {
            Self::Compare(f) => f.as_bytes(),
            Self::Select(f) => f.as_bytes(),
        }
    }
}

pub(super) fn costs<C: AllocationContext>(
) -> Result<(arithmetic::Resources, selection::Resources), Error> {
    Ok((
        arithmetic::resources_for_context::<C>(arithmetic::Spec {
            bits: 39,
            lanes: 1,
            gate: arithmetic::Gate::Drelu,
        })?,
        selection::resources_reconstructed_for_context::<C>(selection::Spec {
            bits: 50,
            lanes: 1,
        })?,
    ))
}

pub(super) fn declare<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: &CausalSoftmaxSpec,
    root: Binding,
) -> Result<Bindings, Error> {
    let mut reads = [[root; 8]; 2];
    let mut maxima = [root; 2];
    let mut scores = [root; 3];
    for (lane, score) in scores.iter_mut().enumerate().take(usize::from(s.count)) {
        *score = dealer.binding(
            s.wire(1 + lane as u16, 1, 50, 12, Interpretation::Signed)?,
            s.operation,
            Some(Interval {
                lower: -spec::SCORE_LIMIT,
                upper: spec::SCORE_LIMIT,
            }),
        )?;
    }
    let mut previous = scores[0];
    for update in 0..usize::from(s.count) - 1 {
        let base = 10 + update as u16 * 8;
        let mut bind = |slot, bits, scale, kind, interval| -> Result<Binding, Error> {
            dealer.binding(
                s.wire(base + slot, 1, bits, scale, kind)?,
                s.operation,
                interval,
            )
        };
        let x = scores[update + 1];
        let d50 = bind(
            1,
            50,
            12,
            Interpretation::Signed,
            Some(Interval {
                lower: -(1i128 << 38) + 1,
                upper: 1i128 << 38,
            }),
        )?;
        let d39 = bind(2, 39, 12, Interpretation::Signed, d50.interval)?;
        let predicate = bind(
            3,
            1,
            0,
            Interpretation::Predicate,
            Some(Interval { lower: 0, upper: 2 }),
        )?;
        let selected = bind(
            4,
            50,
            12,
            Interpretation::Signed,
            Some(Interval {
                lower: 0,
                upper: 1i128 << 38,
            }),
        )?;
        let maximum = bind(
            5,
            50,
            12,
            Interpretation::Signed,
            Some(Interval {
                lower: -spec::SCORE_LIMIT,
                upper: spec::SCORE_LIMIT,
            }),
        )?;
        reads[update] = [previous, x, d50, d39, predicate, d50, x, selected];
        maxima[update] = maximum;
        previous = maximum;
    }
    Ok(Bindings {
        reads,
        maxima,
        scores,
    })
}

#[allow(clippy::too_many_arguments)] // Separate original owned inputs and exact leases.
pub(super) fn issue<C: AllocationContext>(
    s: &CausalSoftmaxSpec,
    region: [u8; 32],
    bindings: Bindings,
    root: (DealerMaskFor<C, Ring>, [CommonMaskedFor<C, Ring>; 2]),
    comparisons: [Option<LeaseFor<C>>; 2],
    selections: [Option<LeaseFor<C>>; 2],
    ledger: &mut ledger::Ledger,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(DealerFor<C>, [PartyFor<C>; 2]), Error> {
    let (root, [left, right]) = root;
    let mut comparison_masks = [None, None];
    let mut select_masks = [None, None];
    let mut comparison_roles = [[None, None], [None, None]];
    let mut select_roles = [[None, None], [None, None]];
    let mut maximum = 0;
    for (update, (comparison, selection)) in comparisons
        .into_iter()
        .zip(selections)
        .enumerate()
        .take(usize::from(s.count) - 1)
    {
        let reads = bindings.reads[update];
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 0,
            },
            reads[0],
            2,
        )?;
        if update == 0 {
            maximum = root.backing.values()[usize::from(s.active[0])];
        }
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 1,
            },
            reads[1],
            2,
        )?;
        let x = root.backing.values()[usize::from(s.active[update + 1])];
        let d50 = maximum.wrapping_sub(x) & MASK50;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 2,
            },
            reads[2],
            2,
        )?;
        let d39 = d50 & MASK39;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 3,
            },
            reads[3],
            2,
        )?;
        let (mask, [a, b]) = arithmetic::issue_reserved_with(
            arithmetic::Spec {
                bits: 39,
                lanes: 1,
                gate: arithmetic::Gate::Drelu,
            },
            arithmetic::Context {
                plan: reads[3].descriptor.identity().plan(),
                session: reads[3].session,
                operation: records::stage(region, 1 + 2 * update as u16),
                input_wire: reads[3].descriptor.identity().wire(),
                output_wire: reads[4].descriptor.identity().wire(),
                first_tensor_index: reads[3].descriptor.identity().first_tensor_index(),
            },
            &[d39],
            comparison.ok_or(Error::Resource)?,
            &mut fill,
        )?;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 4,
            },
            reads[4],
            2,
        )?;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 5,
            },
            reads[5],
            2,
        )?;
        let (select, [c, d]) = selection::issue_reserved_with(
            selection::Spec { bits: 50, lanes: 1 },
            selection::Context {
                plan: reads[5].descriptor.identity().plan(),
                session: reads[5].session,
                operation: records::stage(region, 2 + 2 * update as u16),
                predicate_wire: reads[4].descriptor.identity().wire(),
                input_wire: reads[5].descriptor.identity().wire(),
                output_wire: reads[7].descriptor.identity().wire(),
                first_tensor_index: reads[5].descriptor.identity().first_tensor_index(),
            },
            mask.values(),
            &[d50],
            selection::Boundary::Reconstructed,
            selection.ok_or(Error::Resource)?,
            &mut fill,
        )?;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 6,
            },
            reads[6],
            2,
        )?;
        ledger.burn(
            Read::Max {
                update: update as u8,
                operand: 7,
            },
            reads[7],
            2,
        )?;
        maximum = x.wrapping_add(select.values()[0]) & MASK50;
        comparison_masks[update] = Some(mask);
        select_masks[update] = Some(select);
        comparison_roles[update] = [Some(a), Some(b)];
        select_roles[update] = [Some(c), Some(d)];
    }
    let [mut ca, mut cb] = comparison_roles;
    let [mut sa, mut sb] = select_roles;
    let mut make = |root, role: usize| PartyFor {
        root,
        comparison: [ca[role].take(), cb[role].take()],
        selection: [sa[role].take(), sb[role].take()],
        predicate: None,
        selected: None,
        maximum: 0,
        difference: 0,
        projected: 0,
        update: 0,
        phase: 0,
        count: usize::from(s.count),
        active: s.active,
        bindings: bindings.clone(),
    };
    let parties = [make(left, 0), make(right, 1)];
    Ok((
        DealerFor {
            root,
            comparison: comparison_masks,
            selection: select_masks,
            maximum_mask: maximum,
        },
        parties,
    ))
}

impl<C: AllocationContext> DealerFor<C> {
    pub(super) fn original_mask(&self, position: u8) -> u64 {
        self.root.backing.values()[usize::from(position)]
    }
}
impl<C: AllocationContext> PartyFor<C> {
    pub(super) fn original(&self, position: u8) -> u64 {
        self.root.backing.values()[usize::from(position)]
    }
    pub(super) fn maximum(&self) -> u64 {
        self.maximum
    }
    fn start(&mut self, ledger: &mut ledger::Ledger) -> Result<Frame<C>, Error> {
        let reads = self.bindings.reads[self.update];
        let role = self.root.role;
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 0,
            },
            reads[0],
            role,
        )?;
        if self.update == 0 {
            self.maximum = self.original(self.active[0]);
        }
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 1,
            },
            reads[1],
            role,
        )?;
        self.difference = self
            .maximum
            .wrapping_sub(self.original(self.active[self.update + 1]))
            & MASK50;
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 2,
            },
            reads[2],
            role,
        )?;
        self.projected = self.difference & MASK39;
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 3,
            },
            reads[3],
            role,
        )?;
        self.phase = 1;
        Ok(Frame::Compare(
            self.comparison[self.update]
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(&[self.projected])?,
        ))
    }
    pub(super) fn begin(&mut self, ledger: &mut ledger::Ledger) -> Result<Frame<C>, Error> {
        if self.phase != 0 {
            return Err(Error::Consumed);
        }
        self.start(ledger)
    }
    pub(super) fn advance(
        &mut self,
        peer: &[u8],
        ledger: &mut ledger::Ledger,
    ) -> Result<Option<Frame<C>>, Error> {
        let reads = self.bindings.reads[self.update];
        let role = self.root.role;
        if self.phase == 1 {
            let arithmetic::StepFor::Output(predicate) = self.comparison[self.update]
                .as_mut()
                .ok_or(Error::Consumed)?
                .advance(peer)?
            else {
                return Err(Error::Context);
            };
            self.predicate = Some(predicate);
            ledger.burn(
                Read::Max {
                    update: self.update as u8,
                    operand: 4,
                },
                reads[4],
                role,
            )?;
            ledger.burn(
                Read::Max {
                    update: self.update as u8,
                    operand: 5,
                },
                reads[5],
                role,
            )?;
            self.phase = 2;
            return Ok(Some(Frame::Select(
                self.selection[self.update]
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(
                        self.predicate.as_ref().ok_or(Error::Consumed)?.values(),
                        &[self.difference],
                    )?,
            )));
        }
        if self.phase != 2 {
            return Err(Error::Consumed);
        }
        self.selected = Some(
            self.selection[self.update]
                .as_mut()
                .ok_or(Error::Consumed)?
                .finish(peer)?,
        );
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 6,
            },
            reads[6],
            role,
        )?;
        ledger.burn(
            Read::Max {
                update: self.update as u8,
                operand: 7,
            },
            reads[7],
            role,
        )?;
        self.maximum = self
            .original(self.active[self.update + 1])
            .wrapping_add(self.selected.as_ref().ok_or(Error::Consumed)?.values()[0])
            & MASK50;
        self.update += 1;
        if self.update == self.count - 1 {
            self.phase = 3;
            Ok(None)
        } else {
            self.start(ledger).map(Some)
        }
    }
    pub(super) fn cancel(&mut self) {
        for p in self.comparison.iter_mut().flatten() {
            p.cancel();
        }
        for p in self.selection.iter_mut().flatten() {
            p.cancel();
        }
        self.predicate = None;
        self.selected = None;
        use zeroize::Zeroize;
        self.maximum.zeroize();
        self.difference.zeroize();
        self.projected.zeroize();
    }
}
impl<C: AllocationContext> Drop for PartyFor<C> {
    fn drop(&mut self) {
        self.cancel();
    }
}
