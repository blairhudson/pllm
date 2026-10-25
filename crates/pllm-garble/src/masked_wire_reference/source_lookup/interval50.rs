//! Actual private D43 paired interval issuer over one ORIGINAL GPU50 key pair.
//! Pure scanning belongs exclusively to gpu127::interval50. No DPF or correction
//! evaluator is copied here. Source/scope can enter only via the owning parent.
use super::super::typed::layernorm4::{
    parent::{EnergyBinding, EnergyMaskLoan, EnergyValueLoan},
    SourceFor,
};
use super::*;
use std::alloc::Layout;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

fn spec(bits: u8) -> Spec {
    Spec {
        index_bits: 50,
        output_bits: bits,
        lanes: 1,
        boundary: Boundary::Reconstructed,
    }
}

#[cfg(test)]
pub(in crate::masked_wire_reference) fn panic_next_transition(begin: bool) {
    if begin {
        correction::PANIC_BEGIN.with(|p| p.set(true));
    } else {
        correction::PANIC_ADVANCE.with(|p| p.set(true));
    }
}

// Constructors/fields are private to this actual key-creation module. The two
// corrections cannot reuse the other output's nonce/profile ticket.
pub(super) struct IntervalKeyLane {
    output_bits: u8,
}
impl IntervalKeyLane {
    pub(super) fn output_bits(&self) -> u8 {
        self.output_bits
    }
}
pub(super) struct IntervalIssueSeal {
    spec: Spec,
    binding: [u8; 32],
    select: selection::Context,
}
impl IntervalIssueSeal {
    pub(super) fn consume(self) -> (Spec, [u8; 32], selection::Context) {
        (self.spec, self.binding, self.select)
    }
}

// Created INSIDE the issuer before the actual key pair, not attached to a
// completed scan. Original root source, canonical payloads and counter remain
// resident through the last native frame/output. No coordinator backlink.
#[allow(dead_code)] // Canonical payload/source/lease are actual retained owners, not observer fields.
struct DeclarationFor<C: AllocationContext> {
    entries: [[u64; 4]; 51],
    source: Arc<SourceFor<C>>,
    lease: LeaseFor<C>,
}
fn declaration_bytes<C: AllocationContext>() -> Result<usize, Error> {
    let layout = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<DeclarationFor<C>>())
        .map_err(|_| Error::Resource)?
        .0
        .pad_to_align();
    add(&[layout.size(), ALLOCATOR])
}

#[allow(dead_code)] // Dealer source/control leases remain owned through retirement.
pub(in crate::masked_wire_reference) struct MasksFor<C: AllocationContext> {
    children: [correction::CorrectionMasksFor<C>; 2],
    source: Arc<DeclarationFor<C>>,
    lease: LeaseFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Private native mask loan, no parent activation.
impl<C: AllocationContext> MasksFor<C> {
    pub(in crate::masked_wire_reference) fn source_is(&self, source: &Arc<SourceFor<C>>) -> bool {
        Arc::ptr_eq(source, &self.source.source)
    }
    pub(in crate::masked_wire_reference) fn output(&self, index: usize) -> Result<&[u64], Error> {
        Ok(&self.children.get(index).ok_or(Error::Domain)?.values)
    }
}
#[allow(dead_code)] // Original evaluator controls/leases stay resident, including after dealer Drop.
pub(in crate::masked_wire_reference) struct PartyFor<C: AllocationContext> {
    key: Option<GpuKey>,
    children: [correction::CorrectionFor<C>; 2],
    input: EnergyBinding,
    bindings: [[u8; 32]; 2],
    role: u8,
    phase: u8,
    source: Arc<DeclarationFor<C>>,
    scans: [Option<LeaseFor<C>>; 2],
    controls: LeaseFor<C>,
    lease: LeaseFor<C>,
}
#[allow(dead_code)] // Actual source/lease retention through last escaped frame.
pub(in crate::masked_wire_reference) struct FramesFor<C: AllocationContext> {
    bodies: [FrameFor<C>; 2],
    source: Arc<DeclarationFor<C>>,
    lease: LeaseFor<C>,
}
#[cfg_attr(not(test), allow(dead_code))] // Original native frame borrow; private qualification only.
impl<C: AllocationContext> FramesFor<C> {
    pub(in crate::masked_wire_reference) fn frame(&self, index: usize) -> Result<&[u8], Error> {
        Ok(self.bodies.get(index).ok_or(Error::Domain)?.as_bytes())
    }
}
#[allow(dead_code)] // Actual source/lease retention through last escaped native result.
pub(in crate::masked_wire_reference) struct ValuesFor<C: AllocationContext> {
    bodies: [MaskedOutputFor<C>; 2],
    source: Arc<DeclarationFor<C>>,
    lease: LeaseFor<C>,
    role: u8,
}
#[cfg_attr(not(test), allow(dead_code))] // Native component result loan, not whole-parent publication.
impl<C: AllocationContext> ValuesFor<C> {
    pub(in crate::masked_wire_reference) fn is_for(&self, masks: &MasksFor<C>, role: u8) -> bool {
        self.role == role && Arc::ptr_eq(&self.source, &masks.source)
    }
    pub(in crate::masked_wire_reference) fn output<'a>(
        &'a self,
        source: &Arc<SourceFor<C>>,
        index: usize,
    ) -> Result<&'a MaskedOutputFor<C>, Error> {
        if !Arc::ptr_eq(source, &self.source.source) {
            return Err(Error::Context);
        }
        self.bodies.get(index).ok_or(Error::Domain)
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Native-only continuation until whole parent issuance.
pub(in crate::masked_wire_reference) enum Step<C: AllocationContext> {
    Frames(FramesFor<C>),
    Values(ValuesFor<C>),
}

#[allow(dead_code)] // Named resource fields consumed by future aggregate; no aggregate authority implied.
pub(in crate::masked_wire_reference) struct Resources {
    pub dealer: usize,
    pub party: usize,
    pub combined: usize,
    pub scan: gpu127::interval50::ScanResources,
    pub paired_gpu_aes: usize,
    child: [correction::CorrectionResources; 2],
    scan_bytes: [usize; 2],
    evaluator_controls: usize,
}
impl Resources {
    pub(in crate::masked_wire_reference) fn frames_per_party(&self) -> Result<usize, Error> {
        add(&[
            1, // Each correction first opens its native scan share.
            self.child[0].select.frames_per_party,
            1,
            self.child[1].select.frames_per_party,
        ])
    }
    pub(in crate::masked_wire_reference) fn original_native_workspace_bytes(
        &self,
    ) -> Result<usize, Error> {
        mul(3, WORKSPACE)
    }
    // Same original nonce/pair/four-draw/Select issuance; expose counters rather
    // than duplicate the correction or DPF algorithm in the containing parent.
    pub(in crate::masked_wire_reference) fn work_counts(&self) -> Result<[usize; 3], Error> {
        Ok([
            add(&[
                32,
                32,
                2 * 4 * 8,
                self.child[0].select.random_bytes,
                self.child[1].select.random_bytes,
            ])?,
            add(&[
                self.paired_gpu_aes,
                self.child[0].select.paired_issue_aes_blocks,
                self.child[1].select.paired_issue_aes_blocks,
            ])?,
            add(&[
                self.scan.work.aes_blocks,
                self.child[0].select.evaluation_aes_blocks_per_party,
                self.child[1].select.evaluation_aes_blocks_per_party,
            ])?,
        ])
    }
}
#[allow(dead_code)] // Concrete simultaneous issuance/move slots included in original reservation.
struct IssueControls<'a, C: AllocationContext> {
    loan: EnergyMaskLoan<'a, C>,
    nonce: Zeroizing<[u8; 32]>,
    pair: Result<[GpuKey; 2], gpu127::Error>,
    builds: [Option<correction::CorrectionBuildFor<C>>; 2],
    keys: [Option<GpuKey>; 2],
    tickets: [IntervalKeyLane; 2],
    seals: [IntervalIssueSeal; 2],
    finished: [correction::CorrectionResult<C>; 2],
    result: Result<(MasksFor<C>, [PartyFor<C>; 2]), Error>,
    split_results: [Result<LeaseFor<C>, Error>; 8],
    query: Resources,
    original_intervals: [[u64; 4]; 51],
    converted_intervals: [[u64; 4]; 51],
    declaration: Option<DeclarationFor<C>>,
    arc: Option<Arc<DeclarationFor<C>>>,
    hash: Sha256,
    digest: sha2::digest::Output<Sha256>,
}
#[allow(dead_code)] // Concrete simultaneous guard/begin/advance/move slots, retained independently per party.
struct EvaluatorControls<'a, C: AllocationContext> {
    abort_guard: AbortOnDrop<'a, C>,
    loan: EnergyValueLoan<'a, C>,
    raw: Result<gpu127::interval50::ScanResult, gpu127::Error>,
    scans: [Option<correction::ScanResultFor<C>>; 2],
    native_frames: [Result<FrameFor<C>, Error>; 2],
    native_steps: [Result<StepFor<C>, Error>; 2],
    returned: Result<Step<C>, Error>,
    begin_returned: Result<FramesFor<C>, Error>,
    moving_frame: Option<FramesFor<C>>,
    moving_values: Option<ValuesFor<C>>,
    split: Result<LeaseFor<C>, Error>,
    hash: Sha256,
    source: Arc<DeclarationFor<C>>,
}
#[cfg(test)]
pub(in crate::masked_wire_reference) fn transition_layout<C: AllocationContext>() -> [usize; 3] {
    [
        size_of::<EvaluatorControls<'_, C>>(),
        size_of::<AbortOnDrop<'_, C>>(),
        size_of::<Result<FramesFor<C>, Error>>(),
    ]
}
fn arc_extra<C: AllocationContext>() -> Result<usize, Error> {
    add(&[
        size_of::<Arc<DeclarationFor<C>>>(),
        size_of::<LeaseFor<C>>(),
    ])
}

#[cfg_attr(not(test), allow(dead_code))] // Private native query; whole parent still must price itself.
pub(in crate::masked_wire_reference) fn resources<C: AllocationContext>() -> Result<Resources, Error>
{
    let a = correction::resources_validated::<C>(correction::ValidatedDescriptor::interval50(13)?)?;
    let b = correction::resources_validated::<C>(correction::ValidatedDescriptor::interval50(50)?)?;
    let scans = [
        correction::scan_bytes::<C>(spec(13))?,
        correction::scan_bytes::<C>(spec(50))?,
    ];
    let scan = gpu127::interval50::resources().map_err(gpu_error)?;
    let evaluator_controls = size_of::<EvaluatorControls<'_, C>>();
    // Native GpuKey is INLINE in PartyFor and includes all correction words.
    // Codec744 is not resident size. Original LESS/AES workspace remains the
    // existing native WORKSPACE term, separate from the helper's logical peak.
    let party = add(&[
        a.party,
        b.party,
        size_of::<PartyFor<C>>(),
        scans[0],
        scans[1],
        evaluator_controls,
        scan.known_peak_bytes,
        WORKSPACE,
        mul(3, arc_extra::<C>()?)?,
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let dealer = add(&[
        a.dealer,
        b.dealer,
        size_of::<MasksFor<C>>(),
        size_of::<IssueControls<'_, C>>(),
        declaration_bytes::<C>()?,
        WORKSPACE,
        C::TRANSIENT_LINK_BYTES,
    ])?;
    let combined = add(&[dealer, mul(2, party)?])?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        dealer,
        party,
        combined,
        scan,
        paired_gpu_aes: Profile::new(50)
            .map_err(gpu_error)?
            .resources()
            .paired_issue_aes_blocks,
        child: [a, b],
        scan_bytes: scans,
        evaluator_controls,
    })
}

#[cfg_attr(not(test), allow(dead_code))] // Real closed issuer; no SDK or whole-row activation.
pub(in crate::masked_wire_reference) fn issue_reserved_with<C: AllocationContext>(
    loan: EnergyMaskLoan<'_, C>,
    mut reservation: LeaseFor<C>,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(MasksFor<C>, [PartyFor<C>; 2]), Error> {
    let q = resources::<C>()?;
    if reservation.bytes != q.combined || !std::ptr::eq(reservation.budget.budget(), loan.budget())
    {
        return Err(Error::Resource);
    }
    let input = loan.binding();
    let context = [loan.context(8), loan.context(9)];
    context[0].validate(1)?;
    context[1].validate(1)?;
    let entries = loan.interval_entries();
    let mut hash = Sha256::new();
    hash.update(b"pllm.sigma.interval51.gpu127.r50.e13-u50.v1");
    for entry in &entries {
        for word in entry {
            hash.update(word.to_le_bytes());
        }
    }
    let payload_hash: [u8; 32] = hash.finalize().into();
    let declaration_lease = reservation.split(declaration_bytes::<C>()?)?;
    reservation.budget.budget().allocation()?;
    let source = Arc::new(DeclarationFor {
        entries,
        source: Arc::clone(loan.source()),
        lease: declaration_lease,
    });
    let child_party = add(&[q.child[0].party, q.child[1].party])?;
    let scanner_party = q.party.checked_sub(child_party).ok_or(Error::Resource)?;
    let left = reservation.split(scanner_party)?;
    let right = reservation.split(scanner_party)?;
    let ca = reservation.split(q.child[0].combined)?;
    let cb = reservation.split(q.child[1].combined)?;
    // Split entire child aggregates directly. No guard extraction, merge,
    // refund or reserve-again is necessary to obtain their ORIGINAL leases.
    let mut a =
        correction::prepare_validated(correction::ValidatedDescriptor::interval50(13)?, ca)?;
    let mut b =
        correction::prepare_validated(correction::ValidatedDescriptor::interval50(50)?, cb)?;
    // ALL resources/splits are prepaid before the first entropy byte. Exactly
    // nonce -> ONE GPU pair -> exp four draws -> u four draws -> both Selects.
    let mut nonce = Zeroizing::new([0; 32]);
    fill(&mut nonce[..])?;
    let [k0, k1] = GpuKey::issue_with(
        Profile::new(50).map_err(gpu_error)?,
        0u64.wrapping_sub(loan.mask()) & mask(50),
        |bytes| fill(bytes).map_err(|_| gpu127::Error::Randomness),
    )
    .map_err(gpu_error)?;
    let bindings = std::array::from_fn(|i| {
        invocation_binding(
            spec(if i == 0 { 13 } else { 50 }),
            context[i],
            payload_hash,
            *nonce,
        )
    });
    a.push_interval_keypair(IntervalKeyLane { output_bits: 13 }, &mut fill)?;
    b.push_interval_keypair(IntervalKeyLane { output_bits: 50 }, &mut fill)?;
    let (ma, [a0, a1]) = a.finish_interval(
        IntervalIssueSeal {
            spec: spec(13),
            binding: bindings[0],
            select: select_context(context[0], bindings[0]),
        },
        &mut fill,
    )?;
    let (mb, [b0, b1]) = b.finish_interval(
        IntervalIssueSeal {
            spec: spec(50),
            binding: bindings[1],
            select: select_context(context[1], bindings[1]),
        },
        &mut fill,
    )?;
    let make = |role, key, children, mut lease: LeaseFor<C>| -> Result<PartyFor<C>, Error> {
        let scans = [
            Some(lease.split(q.scan_bytes[0])?),
            Some(lease.split(q.scan_bytes[1])?),
        ];
        let controls = lease.split(q.evaluator_controls)?;
        Ok(PartyFor {
            key: Some(key),
            children,
            input,
            bindings,
            role,
            phase: 0,
            source: Arc::clone(&source),
            scans,
            controls,
            lease,
        })
    };
    let p0 = make(0, k0, [a0, b0], left)?;
    let p1 = make(1, k1, [a1, b1], right)?;
    Ok((
        MasksFor {
            children: [ma, mb],
            source,
            lease: reservation,
        },
        [p0, p1],
    ))
}

struct AbortOnDrop<'a, C: AllocationContext> {
    party: &'a mut PartyFor<C>,
    armed: bool,
}
impl<C: AllocationContext> Drop for AbortOnDrop<'_, C> {
    fn drop(&mut self) {
        if self.armed {
            self.party.cancel()
        }
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Real consuming transitions exercised only by native qualification so far.
impl<C: AllocationContext> PartyFor<C> {
    #[cfg(test)]
    pub(in crate::masked_wire_reference) fn retained_evaluator_bytes(&self) -> usize {
        self.controls.bytes
    }
    pub(in crate::masked_wire_reference) fn cancel(&mut self) {
        self.phase = u8::MAX;
        self.key = None;
        for child in &mut self.children {
            child.cancel()
        }
    }
    pub(in crate::masked_wire_reference) fn begin(
        &mut self,
        loan: EnergyValueLoan<'_, C>,
        allowance: gpu127::interval50::ScanWork,
    ) -> Result<FramesFor<C>, Error> {
        let mut guard = AbortOnDrop {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        if p.phase != 0
            || !Arc::ptr_eq(&p.source.source, loan.source())
            || p.input != loan.binding()
            || p.role != loan.role()
        {
            return Err(Error::Context);
        }
        p.phase = 1;
        let raw = gpu127::interval50::scan(
            p.key.as_ref().ok_or(Error::Consumed)?,
            loan.masked(),
            allowance,
        )
        .map_err(gpu_error)?;
        let mut scans = [None, None];
        for (i, slot) in scans.iter_mut().enumerate() {
            let lease = p.scans[i].take().ok_or(Error::Consumed)?;
            let mut counts = Zeroizing::new(allocate(1, lease.budget.budget())?);
            counts.push(*raw.count);
            let mut sums = Zeroizing::new(allocate(1, lease.budget.budget())?);
            sums.push(raw.sums[i]);
            let mut hash = Sha256::new();
            hash.update(b"pllm.sigma.interval51.masked-energy.v1");
            hash.update(p.bindings[i]);
            hash.update(loan.masked().to_le_bytes());
            *slot = Some(correction::ScanResultFor {
                sums,
                counts,
                role: p.role,
                spec: spec(if i == 0 { 13 } else { 50 }),
                binding: p.bindings[i],
                digest: hash.finalize().into(),
                lease,
            });
        }
        let f0 = p.children[0].begin(scans[0].take().ok_or(Error::Consumed)?)?;
        let f1 = p.children[1].begin(scans[1].take().ok_or(Error::Consumed)?)?;
        let lease = p.lease.split(arc_extra::<C>()?)?;
        let result = FramesFor {
            bodies: [f0, f1],
            source: Arc::clone(&p.source),
            lease,
        };
        guard.armed = false;
        Ok(result)
    }
    pub(in crate::masked_wire_reference) fn advance(
        &mut self,
        peer: [&[u8]; 2],
    ) -> Result<Step<C>, Error> {
        let mut guard = AbortOnDrop {
            party: self,
            armed: true,
        };
        let p = &mut *guard.party;
        if !matches!(p.phase, 1 | 2) {
            return Err(Error::Consumed);
        }
        let a = p.children[0].advance(peer[0])?;
        let b = p.children[1].advance(peer[1])?;
        let lease = p.lease.split(arc_extra::<C>()?)?;
        let result = match (a, b, p.phase) {
            (StepFor::Frame(a), StepFor::Frame(b), 1) => {
                p.phase = 2;
                Step::Frames(FramesFor {
                    bodies: [a, b],
                    source: Arc::clone(&p.source),
                    lease,
                })
            }
            (StepFor::Reconstructed(a), StepFor::Reconstructed(b), 2) => {
                p.phase = 3;
                Step::Values(ValuesFor {
                    bodies: [a, b],
                    source: Arc::clone(&p.source),
                    lease,
                    role: p.role,
                })
            }
            _ => return Err(Error::Context),
        };
        guard.armed = false;
        Ok(result)
    }
}
