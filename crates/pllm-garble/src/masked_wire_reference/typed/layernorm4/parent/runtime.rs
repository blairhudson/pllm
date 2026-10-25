//! Closed local-row dispatcher. Native masked results are never reconstructed.
use super::*;
use crate::compact_dpf::gpu127;
const READY: u8 = 16;
pub(super) struct Role<C: AllocationContext> {
    coordination: Arc<CoordinationFor<C, InitializerState<C>>>,
    role: u8,
    handed_off: bool,
}
impl<C: AllocationContext> Drop for Role<C> {
    fn drop(&mut self) {
        if !self.handed_off {
            self.coordination.abort()
        }
    }
}
#[allow(dead_code)] // Private consuming native ingress; no SDK/default activation.
impl<C: AllocationContext> PartyInitializer<C> {
    pub(super) fn into_role(mut self) -> Role<C> {
        self.handed_off = true;
        Role {
            coordination: Arc::clone(&self.coordination),
            role: self.role,
            handed_off: false,
        }
    }
}
// Borrowed guards, not user callbacks. Drop a loan before any role operation.
pub(super) struct FrameLoan<'a, C: AllocationContext> {
    state: std::sync::MutexGuard<'a, InitializerState<C>>,
    _control: std::sync::MutexGuard<'a, SplitControl>,
    slot: usize,
    part: usize,
}
#[allow(dead_code)] // Guarded original frame read for private component users.
impl<C: AllocationContext> FrameLoan<'_, C> {
    pub(super) fn bytes(&self) -> Result<&[u8], Error> {
        match self.state.frames[self.slot]
            .as_ref()
            .ok_or(Error::Consumed)?
        {
            FrameOwner::Arithmetic(v) => Ok(v.as_bytes()),
            FrameOwner::Product(v) => Ok(v.as_bytes()),
            FrameOwner::Interval(v) => v.frame(self.part),
            FrameOwner::Inverse(v) => Ok(v.as_bytes()),
            FrameOwner::Modular(v) => Ok(v.as_bytes()),
        }
    }
}
#[allow(dead_code)] // Borrow-only native frame API, no transport registration.
impl<C: AllocationContext> FrameHandle<C> {
    pub(super) fn loan(&self, part: usize) -> Result<FrameLoan<'_, C>, Error> {
        let control = self.coordination.control();
        if control.aborted {
            return Err(Error::Consumed);
        }
        let state = self.coordination.slots[usize::from(self.role)]
            .state
            .lock()
            .map_err(|_| Error::Consumed)?;
        let f = state.frames[usize::from(self.slot)]
            .as_ref()
            .ok_or(Error::Consumed)?;
        if part
            >= if matches!(f, FrameOwner::Interval(_)) {
                2
            } else {
                1
            }
        {
            return Err(Error::Domain);
        }
        Ok(FrameLoan {
            state,
            _control: control,
            slot: usize::from(self.slot),
            part,
        })
    }
    pub(super) fn parts(&self) -> Result<usize, Error> {
        let loan = self.loan(0)?;
        Ok(
            if matches!(
                loan.state.frames[usize::from(self.slot)],
                Some(FrameOwner::Interval(_))
            ) {
                2
            } else {
                1
            },
        )
    }
}
pub(super) struct ShareLoan<'a, C: AllocationContext> {
    state: std::sync::MutexGuard<'a, InitializerState<C>>,
    _control: std::sync::MutexGuard<'a, SplitControl>,
}
#[allow(dead_code)] // Actual secret-share borrow, not a public API.
impl<C: AllocationContext> ShareLoan<'_, C> {
    pub(super) fn values(&self) -> Result<&[u64], Error> {
        Ok(self.state.share.as_ref().ok_or(Error::Consumed)?.values())
    }
}
#[allow(dead_code)] // Both-ready share loan, no output-source issuer.
impl<C: AllocationContext> ShareHandle<C> {
    pub(super) fn loan(&self) -> Result<ShareLoan<'_, C>, Error> {
        let control = self.coordination.control();
        if control.aborted {
            return Err(Error::Consumed);
        }
        control.require_both_ready()?;
        let state = self.coordination.slots[usize::from(self.role)]
            .state
            .lock()
            .map_err(|_| Error::Consumed)?;
        if state.phase != READY || state.share.is_none() {
            return Err(Error::Consumed);
        }
        Ok(ShareLoan {
            state,
            _control: control,
        })
    }
}
impl<C: AllocationContext> InitializerState<C> {
    fn words(&self, i: usize) -> Result<&[u64], Error> {
        Ok(self.values[i]
            .as_ref()
            .ok_or(Error::Consumed)?
            .backing
            .values())
    }
    fn burn(&mut self, use_index: usize, slot: usize, role: u8) -> Result<(), Error> {
        self.ledger
            .burn(use_index as u8, self.keeper.bindings[slot], role)
    }
    fn install(&mut self, i: usize, role: u8, backing: ValueBacking<C>) -> Result<(), Error> {
        if self.values[i].is_some() {
            return Err(Error::Consumed);
        }
        self.values[i] = Some(ProducedValueFor {
            backing,
            binding: self.keeper.bindings[i],
            role,
            source: Arc::clone(&self.keeper.source),
            lease: self.runtime_lease.split(size_of::<ProducedValueFor<C>>())?,
        });
        Ok(())
    }
    fn product_result(
        &mut self,
        i: usize,
        role: u8,
        native: product::OutputFor<C>,
    ) -> Result<(), Error> {
        let common = CommonMaskedFor {
            backing: CommonBackingFor::Elementwise(native),
            binding: self.keeper.bindings[i],
            role,
            _kind: PhantomData,
            _lease: self.runtime_lease.split(common_charge_for::<C>())?,
        };
        self.install(i, role, ValueBacking::Original(common))
    }
    fn linear(
        &mut self,
        i: usize,
        buffer: usize,
        map: LocalMap,
        left: usize,
        right: Option<usize>,
        role: u8,
    ) -> Result<(), Error> {
        let mut out =
            std::mem::replace(&mut self.local_buffers[buffer], Zeroizing::new(Vec::new()));
        map.apply(
            self.words(left)?,
            match right {
                Some(r) => self.words(r)?,
                None => &[],
            },
            &mut out,
            false,
        )?;
        // Cover preissued buffer payload until the real local result retires.
        let payload = self.runtime_lease.split(out.len() * 8 + 64)?;
        self.install(i, role, ValueBacking::Local(out))?;
        // Merge is forbidden: retain separate covering guard in existing slot.
        self.local_payloads[buffer] = Some(payload);
        Ok(())
    }
    fn native_begin(&mut self, role: u8) -> Result<FrameOwner<C>, Error> {
        if self.phase != 0 {
            return Err(Error::Consumed);
        }
        self.burn(0, 0, role)?;
        self.linear(3, 0, LocalMap::ScaledMean, 0, None, role)?;
        self.burn(1, 3, role)?;
        let frame = self.mean.begin(
            self.values[3]
                .as_ref()
                .ok_or(Error::Consumed)?
                .backing
                .values(),
        )?;
        self.phase = 1;
        Ok(FrameOwner::Arithmetic(frame))
    }
    fn native_advance(&mut self, role: u8, peer: &[&[u8]]) -> Result<Option<FrameOwner<C>>, Error> {
        let n = if self.phase == 4 || self.phase == 5 {
            2
        } else {
            1
        };
        if peer.len() != n {
            return Err(Error::Context);
        }
        let frame = match self.phase {
            1 => {
                let arithmetic::StepFor::Frame(f) = self.mean.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.phase = 2;
                FrameOwner::Arithmetic(f)
            }
            2 => {
                let arithmetic::StepFor::Output(v) = self.mean.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.install(4, role, ValueBacking::Arithmetic(v))?;
                self.burn(2, 0, role)?;
                self.burn(3, 4, role)?;
                self.linear(5, 1, LocalMap::Center, 0, Some(4), role)?;
                self.burn(4, 5, role)?;
                self.burn(5, 5, role)?;
                let f = self.square.begin(
                    self.values[5]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                    self.values[5]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                )?;
                self.phase = 3;
                FrameOwner::Product(f)
            }
            3 => {
                let v = self.square.finish(peer[0])?;
                self.product_result(6, role, v)?;
                self.burn(6, 6, role)?;
                self.linear(7, 2, LocalMap::Energy, 6, None, role)?;
                let loan = EnergyValueLoan::take(
                    &self.keeper.source,
                    &mut self.ledger,
                    &self.values,
                    role,
                )?;
                let f = self.interval.begin(
                    loan,
                    gpu127::interval50::resources()
                        .map_err(|_| Error::Resource)?
                        .work,
                )?;
                self.phase = 4;
                FrameOwner::Interval(f)
            }
            4 => {
                let paired::Step::Frames(f) = self.interval.advance([peer[0], peer[1]])? else {
                    return Err(Error::Context);
                };
                self.phase = 5;
                FrameOwner::Interval(f)
            }
            5 => {
                let paired::Step::Values(v) = self.interval.advance([peer[0], peer[1]])? else {
                    return Err(Error::Context);
                };
                let energy = self.values[7].take().ok_or(Error::Consumed)?;
                let f = self.modular.begin(&mut self.ledger, energy, v)?;
                self.phase = 6;
                FrameOwner::Modular(f)
            }
            6 => {
                let v = self.modular.finish(peer[0])?;
                let f = self.coordinate.begin(&mut self.ledger, v)?;
                self.phase = 7;
                FrameOwner::Arithmetic(f)
            }
            7 => {
                let coordinate::CoordinateStep::Frame(f) =
                    self.coordinate.advance(&mut self.ledger, peer[0])?
                else {
                    return Err(Error::Context);
                };
                self.phase = 8;
                FrameOwner::Arithmetic(f)
            }
            8 => {
                let coordinate::CoordinateStep::Index(v) =
                    self.coordinate.advance(&mut self.ledger, peer[0])?
                else {
                    return Err(Error::Context);
                };
                self.values[13] = Some(v);
                let loan = Norm13ValueLoan::take(
                    &self.keeper.source,
                    &mut self.ledger,
                    &self.values,
                    role,
                )?;
                let f = self.inverse.begin(loan)?;
                self.phase = 9;
                FrameOwner::Inverse(f)
            }
            9 => {
                let norm13::Step::Frame(f) = self.inverse.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.phase = 10;
                FrameOwner::Inverse(f)
            }
            10 => {
                let norm13::Step::Output(v) = self.inverse.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.install(14, role, ValueBacking::Inverse(v))?;
                self.burn(17, 5, role)?;
                self.burn(18, 14, role)?;
                let scalar = Zeroizing::new(self.words(14)?[0]);
                self.broadcast.fill(*scalar);
                let f = self.normalize.begin(
                    self.values[5]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                    &self.broadcast,
                )?;
                self.phase = 11;
                FrameOwner::Product(f)
            }
            11 => {
                let v = self.normalize.finish(peer[0])?;
                self.product_result(15, role, v)?;
                self.burn(19, 15, role)?;
                let f = self.floor.begin(
                    self.values[15]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                )?;
                self.phase = 12;
                FrameOwner::Arithmetic(f)
            }
            12 => {
                let arithmetic::StepFor::Frame(f) = self.floor.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.phase = 13;
                FrameOwner::Arithmetic(f)
            }
            13 => {
                let arithmetic::StepFor::Output(v) = self.floor.advance(peer[0])? else {
                    return Err(Error::Context);
                };
                self.install(16, role, ValueBacking::Arithmetic(v))?;
                self.burn(20, 16, role)?;
                self.burn(21, 1, role)?;
                let f = self.affine.begin(
                    self.values[16]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                    self.values[1]
                        .as_ref()
                        .ok_or(Error::Consumed)?
                        .backing
                        .values(),
                )?;
                self.phase = 14;
                FrameOwner::Product(f)
            }
            14 => {
                let v = self.affine.finish(peer[0])?;
                self.product_result(17, role, v)?;
                self.burn(22, 17, role)?;
                let ValueBacking::Original(input) =
                    &self.values[17].as_ref().ok_or(Error::Consumed)?.backing
                else {
                    return Err(Error::Context);
                };
                let f = self
                    .terminal
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin_borrowed(input)?;
                self.phase = 15;
                FrameOwner::Arithmetic(f)
            }
            15 => {
                self.burn(23, 2, role)?;
                let v = self.values[2].take().ok_or(Error::Consumed)?;
                let ValueBacking::Original(beta) = v.backing else {
                    return Err(Error::Context);
                };
                self.share = Some(
                    self.terminal
                        .take()
                        .ok_or(Error::Consumed)?
                        .finish(beta, peer[0])?,
                );
                if self.ledger.next != USES as u8 {
                    return Err(Error::Context);
                }
                self.phase = READY;
                return Ok(None);
            }
            _ => return Err(Error::Consumed),
        };
        Ok(Some(frame))
    }
}
#[allow(dead_code)] // Full private protocol dispatch awaiting component review.
impl<C: AllocationContext> Role<C> {
    fn dispatch(&mut self, peer: Option<&[&[u8]]>) -> Result<RowStep<C>, Error> {
        self.coordination.operation(|control| {
            let mut state = self.coordination.slots[usize::from(self.role)]
                .state
                .lock()
                .map_err(|_| Error::Consumed)?;
            let frame = match peer {
                None => Some(state.native_begin(self.role)?),
                Some(p) => state.native_advance(self.role, p)?,
            };
            if let Some(f) = frame {
                let slot = state
                    .frames
                    .iter()
                    .position(Option::is_none)
                    .ok_or(Error::Resource)?;
                state.frames[slot] = Some(f);
                Ok(RowStep::Frame(FrameHandle {
                    coordination: Arc::clone(&self.coordination),
                    role: self.role,
                    slot: slot as u8,
                }))
            } else {
                control.mark_ready(self.role);
                Ok(RowStep::Ready)
            }
        })
    }
    pub(super) fn begin(&mut self) -> Result<RowStep<C>, Error> {
        self.dispatch(None)
    }
    pub(super) fn advance(&mut self, peer: &[&[u8]]) -> Result<RowStep<C>, Error> {
        self.dispatch(Some(peer))
    }
    pub(super) fn finish(mut self) -> Result<ShareHandle<C>, Error> {
        let out = self.coordination.operation(|control| {
            control.require_both_ready()?;
            let state = self.coordination.slots[usize::from(self.role)]
                .state
                .lock()
                .map_err(|_| Error::Consumed)?;
            if state.phase != READY || state.share.is_none() {
                return Err(Error::Consumed);
            }
            Ok(ShareHandle {
                coordination: Arc::clone(&self.coordination),
                role: self.role,
            })
        })?;
        self.handed_off = true;
        Ok(out)
    }
    pub(super) fn cancel(&self) {
        self.coordination.abort()
    }
    pub(super) fn poll_cancelled(&self, cancelled: &mut impl FnMut() -> bool) -> Result<(), Error> {
        self.coordination.poll_cancelled(cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn issued() -> (OwnedBudget, IssuedRow<OwnedBudgetContext>) {
        let b = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
        let q = resources::<OwnedBudgetContext>().unwrap();
        let s =
            LocalClearRootSpec4::new(&[-1, 0, 0, 0], &[-4096, -1, 1, 4095], &[4095, -4096, 1, -1])
                .unwrap();
        let row = issue_reserved_with(s, [97; 32], b.reserve_native(q.total_bytes).unwrap(), |v| {
            getrandom::fill(v).map_err(|_| Error::Randomness)
        })
        .unwrap();
        (b, row)
    }
    fn wire<C: AllocationContext>(f: &FrameHandle<C>) -> Vec<Vec<u8>> {
        (0..f.parts().unwrap())
            .map(|i| f.loan(i).unwrap().bytes().unwrap().to_vec())
            .collect()
    }
    fn drive(
        a: &mut Role<OwnedBudgetContext>,
        b: &mut Role<OwnedBudgetContext>,
    ) -> (
        FrameHandle<OwnedBudgetContext>,
        FrameHandle<OwnedBudgetContext>,
    ) {
        let mut x = a.begin().unwrap();
        let mut y = b.begin().unwrap();
        let mut first = None;
        loop {
            match (x, y) {
                (RowStep::Frame(fa), RowStep::Frame(fb)) => {
                    let wa = wire(&fa);
                    let wb = wire(&fb);
                    let pa = wa.iter().map(Vec::as_slice).collect::<Vec<_>>();
                    let pb = wb.iter().map(Vec::as_slice).collect::<Vec<_>>();
                    if first.is_none() {
                        first = Some((fa, fb))
                    }
                    x = a.advance(&pb).unwrap();
                    y = b.advance(&pa).unwrap();
                }
                (RowStep::Ready, RowStep::Ready) => return first.unwrap(),
                _ => panic!("asymmetric stages"),
            }
        }
    }
    #[test]
    fn finish_before_ready_burns_peer_and_retained_frame() {
        let (
            _,
            IssuedRow {
                material,
                parties: [a, b],
            },
        ) = issued();
        let mut a = a.into_role();
        let mut b = b.into_role();
        drop(material);
        let RowStep::Frame(f) = a.begin().unwrap() else {
            panic!()
        };
        assert!(a.finish().is_err());
        assert!(b.begin().is_err());
        assert!(f.loan(0).is_err());
    }
    #[test]
    fn cancel_after_both_ready_revokes_share_and_old_frames() {
        let (
            budget,
            IssuedRow {
                material,
                parties: [a, b],
            },
        ) = issued();
        let mut a = a.into_role();
        let mut b = b.into_role();
        drop(material);
        let (fa, fb) = drive(&mut a, &mut b);
        let out = a.finish().unwrap();
        assert_eq!(out.loan().unwrap().values().unwrap().len(), 4);
        b.cancel();
        assert!(out.loan().is_err());
        assert!(fa.loan(0).is_err());
        assert!(fb.loan(0).is_err());
        assert!(budget.budget().used_bytes() > 65536);
        drop(b);
        drop(fb);
        drop(out);
        drop(fa);
        assert!(budget.budget().used_bytes() < 65536);
    }
    #[test]
    fn role_drop_after_provisional_handoffs_burns_without_manual_abort() {
        let (
            _,
            IssuedRow {
                material,
                parties: [a, b],
            },
        ) = issued();
        let mut a = a.into_role();
        let mut b = b.into_role();
        drop(material);
        let (fa, fb) = drive(&mut a, &mut b);
        drop(a);
        assert!(b.finish().is_err());
        assert!(fa.loan(0).is_err());
        assert!(fb.loan(0).is_err());
    }
    #[test]
    fn unlocked_callback_unwind_retires_original_roles() {
        let (
            _,
            IssuedRow {
                material,
                parties: [a, b],
            },
        ) = issued();
        let mut a = a.into_role();
        let mut b = b.into_role();
        drop(material);
        let RowStep::Frame(f) = a.begin().unwrap() else {
            panic!()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = a.poll_cancelled(&mut || panic!("original cancellation callback"));
        }));
        assert!(result.is_err());
        assert!(f.loan(0).is_err());
        assert!(b.begin().is_err());
    }
    #[test]
    fn malformed_final_peer_revokes_provisional_share() {
        let (
            _,
            IssuedRow {
                material,
                parties: [a, b],
            },
        ) = issued();
        let mut a = a.into_role();
        let mut b = b.into_role();
        drop(material);
        let mut x = a.begin().unwrap();
        let mut y = b.begin().unwrap();
        loop {
            let (RowStep::Frame(fa), RowStep::Frame(fb)) = (x, y) else {
                panic!("missing terminal frame")
            };
            let wa = wire(&fa);
            let wb = wire(&fb);
            let pa = wa.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let pb = wb.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let final_stage = a.coordination.slots[0].state.lock().unwrap().phase == 15;
            x = a.advance(&pb).unwrap();
            if final_stage {
                assert!(matches!(x, RowStep::Ready));
                assert!(b.advance(&[&[]]).is_err());
                assert!(fa.loan(0).is_err());
                assert!(fb.loan(0).is_err());
                assert!(a.finish().is_err());
                break;
            }
            y = b.advance(&pa).unwrap();
        }
    }
    #[test]
    fn literal_first_negative_mean_full_private_protocol() {
        let (
            b,
            IssuedRow {
                material,
                parties: [a, bp],
            },
        ) = issued();
        let before = b.budget().used_bytes();
        let mut a = a.into_role();
        let mut p = bp.into_role();
        drop(material);
        let mut sa = a.begin().unwrap();
        let mut sb = p.begin().unwrap();
        let mut batches = 0;
        loop {
            match (sa, sb) {
                (RowStep::Frame(fa), RowStep::Frame(fb)) => {
                    let wa = wire(&fa);
                    let wb = wire(&fb);
                    let pa = wa.iter().map(Vec::as_slice).collect::<Vec<_>>();
                    let pb = wb.iter().map(Vec::as_slice).collect::<Vec<_>>();
                    sa = a.advance(&pb).unwrap();
                    sb = p.advance(&pa).unwrap();
                    batches += wa.len();
                }
                (RowStep::Ready, RowStep::Ready) => break,
                _ => panic!("asymmetric native phases"),
            }
        }
        assert_eq!(batches, 17);
        let x = a.finish().unwrap();
        let y = p.finish().unwrap();
        let ax = x.loan().unwrap().values().unwrap().to_vec();
        let by = y.loan().unwrap().values().unwrap().to_vec();
        let mut result = [0i64; 4];
        for i in 0..4 {
            let u = ax[i].wrapping_add(by[i]) & ((1u64 << 50) - 1);
            result[i] = if u & (1 << 49) != 0 {
                (u as i64) - (1i64 << 50)
            } else {
                u as i64
            };
        }
        assert_eq!(result, [4095, -4097, 2, 4094]);
        println!("D43 actual reconstructed party shares={result:?}; original native frames/party={batches}");
        assert!(b.budget().used_bytes() <= before);
        drop(x);
        drop(y);
    }
    #[test]
    fn independent_twenty_two_records_through_fresh_native_rows() {
        for (index, case) in
            crate::masked_wire_reference::typed::layernorm4::native_oracle_cases::CASES
                .iter()
                .enumerate()
        {
            let x = case.x.map(i64::from);
            let gamma = case.gamma.map(i64::from);
            let beta = case.beta.map(i64::from);
            let budget = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
            let baseline = budget.budget().used_bytes();
            let q = resources::<OwnedBudgetContext>().unwrap();
            let spec = LocalClearRootSpec4::new(&x, &gamma, &beta).unwrap();
            let IssuedRow {
                material,
                parties: [a, b],
            } = issue_reserved_with(
                spec,
                [index as u8; 32],
                budget.reserve_native(q.total_bytes).unwrap(),
                |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
            )
            .unwrap();
            let mut a = a.into_role();
            let mut b = b.into_role();
            drop(material);
            let (fa, fb) = drive(&mut a, &mut b);
            let oa = a.finish().unwrap();
            let ob = b.finish().unwrap();
            let va = oa.loan().unwrap().values().unwrap().to_vec();
            let vb = ob.loan().unwrap().values().unwrap().to_vec();
            let result = std::array::from_fn::<_, 4, _>(|i| {
                let v = va[i].wrapping_add(vb[i]) & ((1u64 << 50) - 1);
                if v & (1 << 49) != 0 {
                    v as i64 - (1i64 << 50)
                } else {
                    v as i64
                }
            });
            assert_eq!(result, case.expected, "independent record {index}");
            println!("D43 native record {index:02} actual={result:?}");
            drop(oa);
            drop(ob);
            drop(fa);
            drop(fb);
            assert_eq!(
                budget.budget().used_bytes(),
                baseline,
                "record {index} source still retained"
            );
        }
    }
}
