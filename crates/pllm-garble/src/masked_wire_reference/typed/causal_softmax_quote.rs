//! Aggregate preconstruction census. Concrete owned buffers/controls and native
//! owner contracts only; no private-stack, allocator/RSS or physical-peak claim.
use super::super::activation::{certified_exp_lookup as exp, certified_softmax_inverse as inverse};
use super::super::nexp::region::inherited;
use super::super::{role_coordination as custody, softmax_coordinate as coordinate};
use super::*;
use crate::masked_wire_reference::elementwise_product as product;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalSoftmaxWork {
    /// Fixed-arity original-owner quote passes. A pass includes its bounded
    /// metadata helpers, not a CPU-instruction/runtime measurement.
    pub resource_queries: usize,
    pub original_validation: usize,
    pub original_retirement: usize,
    pub outer_uses: usize,
    /// Full fixed arrays, including non-executable padding. Four templates
    /// (uses + three ledgers), use initialization, and source-level array copies.
    pub fixed_record_initializations: usize,
    pub local_lane_visits: usize,
    pub initialized_words: usize,
    pub erased_words: usize,
    pub native_issue_aes: usize,
    pub native_eval_aes: usize,
    pub entropy_bytes: usize,
    pub public_hash_bytes: usize,
    pub inverse_verify: usize,
    pub coordinate_bridge: usize,
    /// SECOND public 512-floor pass, in addition to the complete child quote.
    pub outer_exp_floor: PublicFloorWork,
    pub child: CausalChildWork,
    pub total: u128,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicFloorWork {
    pub entries: usize,
    pub taylor: usize,
    pub squarings: usize,
    pub multiply: usize,
    pub divide: usize,
    pub remainder: usize,
    pub add: usize,
    pub shift: usize,
}
impl PublicFloorWork {
    fn total(self) -> Result<usize, Error> {
        sum(&[
            self.entries,
            self.taylor,
            self.squarings,
            self.multiply,
            self.divide,
            self.remainder,
            self.add,
            self.shift,
        ])
    }
}
pub type CausalChildWork = super::super::nexp::region::WorkQuote;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalSoftmaxResources {
    pub incoming_owner_bytes: usize,
    pub original_root_bytes: usize,
    pub maximum_bytes: usize,
    pub child_bytes: usize,
    pub coordinate_bytes: usize,
    pub inverse_bytes: usize,
    pub product_bytes: usize,
    pub gap_bytes: usize,
    pub source_bytes: usize,
    pub state_bytes: usize,
    pub slot_bytes: usize,
    pub coordination_bytes: usize,
    pub transition_bytes: usize,
    pub total_reserved_bytes: usize,
    pub peer_bytes: usize,
    pub work: CausalSoftmaxWork,
}

/// Existing child work categories are cumulative; none is reset/loaned merely
/// because a prior phase stopped using its workspace. Count the full bound.
fn child_work(w: CausalChildWork) -> Result<u128, Error> {
    [
        w.public_declaration_descriptor_checks,
        w.public_resource_query_passes,
        w.public_resource_descriptor_checks,
        w.input_validation_lanes,
        w.input_retirement_words,
        w.root_validation_lanes,
        w.root_initialization_words,
        w.root_encoding_word_writes,
        w.root_mask_generation_words,
        w.root_retirement_words,
        w.ordered_use_checks,
        w.public_table_range_checks,
        w.stage_bound_descriptor_checks,
        w.stage_bound_interval_writes,
        w.stage_bound_metadata_queries,
        w.stage_bound_table_metadata_passes,
        w.buffer_initialization_words,
        w.buffer_erasure_words,
        w.projection_lane_visits,
        w.floor_taylor,
        w.floor_squarings,
        w.floor_multiply,
        w.floor_divide,
        w.floor_remainder,
        w.floor_add,
        w.floor_shift,
        w.paired_native_issue_aes,
        w.random_byte_requests,
    ]
    .into_iter()
    .try_fold(
        (w.native_evaluate_aes_per_role as u128)
            .checked_mul(2)
            .ok_or(Error::Resource)?,
        |a, b| a.checked_add(b as u128).ok_or(Error::Resource),
    )
}

pub(super) fn resources<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    s: &CausalSoftmaxSpec,
    input: &CausalSoftmaxInput,
) -> Result<CausalSoftmaxResources, Error> {
    s.check_original(dealer, input)?;
    let k = usize::from(s.count);
    let width = s.input.shape().elements();
    // One root; k original-lane views; five bindings/update; difference;
    // thirteen original inherited child outputs; D50, D14, TR, R, P, Gap, row.
    let issued_bindings = if k == 1 { 1 } else { 17 + 6 * k };
    dealer
        .counter
        .checked_add(issued_bindings as u64)
        .ok_or(Error::Resource)?;
    let root = if k == 1 {
        0
    } else {
        dealer.root_reservation_bytes(
            s.input,
            &BoundRange::new(-spec::SCORE_LIMIT, spec::SCORE_LIMIT)?,
        )?
    };
    let (comparison, select) = maximum::costs::<C>()?;
    let child_spec = super::region::child_spec(s)?;
    let child = if k == 1 {
        None
    } else {
        Some(inherited::resources(dealer, &child_spec, input)?)
    };
    let coordinate = coordinate::resources_for::<C>()?;
    let inverse = inverse::quote::<C>(1).map_err(super::region::native_inverse_error)?;
    let exp = exp::quote::<C>(k).map_err(super::region::native_exp_error)?;
    let product = product::resources_for::<C>(k)?;
    let gap = arithmetic::resources_for_context::<C>(arithmetic::Spec {
        bits: 50,
        lanes: k,
        gate: arithmetic::Gate::GapArs { shift: 12 },
    })?;
    let original_root_bytes = root;
    let maximum_bytes = if k == 1 {
        0
    } else {
        mul(
            k - 1,
            sum(&[
                comparison.combined_reservation_bytes,
                select.combined_reservation_bytes,
            ])?,
        )?
    };
    let child_bytes = child.as_ref().map_or(0, |q| q.total_reserved_bytes());
    let coordinate_bytes = if k == 1 { 0 } else { coordinate.total_bytes };
    let inverse_bytes = if k == 1 {
        0
    } else {
        inverse.native_reservation_bytes
    };
    let product_bytes = if k == 1 { 0 } else { product.combined_bytes };
    let gap_bytes = if k == 1 {
        0
    } else {
        gap.combined_reservation_bytes
    };
    let source_bytes = sum(&[
        custody::arc_bytes::<SourceFor<C>>()?,
        size_of::<super::region::KeepaliveFor<C>>(),
        custody::arc_bytes::<super::region::KeepaliveFor<C>>()?,
        exp.preparation_peak_bytes,
        inverse.preparation_peak_bytes,
        inverse.native_query_phase_bytes,
        inverse.issuance_adapter_bytes,
    ])?;
    let state_bytes = sum(&[
        size_of::<super::region::State<C>>(),
        // Initial broadcast k + original difference k + original D14 one-word
        // buffers. The last two move intact into child/bridge owners.
        mul(if k == 1 { 0 } else { 2 * k + 1 }, size_of::<u64>())?,
        mul(s.exchanges(), size_of::<super::region::Frame<C>>())?,
        mul(s.exchanges(), size_of::<super::region::Step<C>>())?,
        size_of::<DenominatorValueFor<C>>(),
        size_of::<DifferenceValueFor<C>>(),
        size_of::<super::region::Output<C>>(),
    ])?;
    let slot_bytes = custody::arc_bytes::<super::region::Slot<C>>()?;
    let coordination_bytes = custody::arc_bytes::<super::region::Shared<C>>()?;
    let transition_bytes = super::region::transition_bytes::<C>()?;
    let total_reserved_bytes = sum(&[
        original_root_bytes,
        maximum_bytes,
        child_bytes,
        coordinate_bytes,
        inverse_bytes,
        product_bytes,
        gap_bytes,
        source_bytes,
        mul(2, state_bytes)?,
        mul(2, slot_bytes)?,
        coordination_bytes,
        transition_bytes,
        // Dealer difference k, denominator1 and reciprocal broadcast k.
        mul(if k == 1 { 0 } else { 2 * k + 1 }, size_of::<u64>())?,
    ])?
    .max(super::region::bootstrap_bytes::<C>()?);
    if add(total_reserved_bytes, input.charged_bytes())? > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    let peer_bytes = if k == 1 {
        0
    } else {
        sum(&[
            mul(
                k - 1,
                sum(&[
                    comparison.total_peer_frame_bytes,
                    select.total_peer_frame_bytes,
                ])?,
            )?,
            child.as_ref().ok_or(Error::Context)?.peer_bytes(),
            coordinate.native.total_peer_frame_bytes,
            inverse.lookup.total_peer_frame_bytes,
            product.total_peer_bytes,
            gap.total_peer_frame_bytes,
        ])?
    };
    let mut child_work_value = child
        .as_ref()
        .map(|q| q.work())
        .unwrap_or(super::region::zero_child_work());
    if k > 1 {
        // Two original-owner parent quote passes, then the inherited factory's
        // actual-mask quote; not three independently reusable full ceilings.
        child_work_value.public_resource_query_passes = 3;
        child_work_value.public_resource_descriptor_checks = 3 * 14;
    }
    let mut work = CausalSoftmaxWork {
        resource_queries: 2,
        original_validation: mul(2, width)?,
        original_retirement: add(input.payload().capacity(), input.payload().len())?,
        outer_uses: mul(3, s.outer_uses())?,
        fixed_record_initializations: if k == 1 {
            0
        } else {
            sum(&[
                mul(4, add(spec::MAX_USES, s.outer_uses())?)?,
                add(spec::MAX_USES, s.outer_uses())?,
                // Dealer bindings, frozen postimage, two role binding copies;
                // one dealer + two role ConsumerUse copies.
                mul(7, spec::MAX_USES)?,
            ])?
        },
        // Original max reads/projections/affine writes, all difference/sum and
        // broadcast/output lane passes, independently for dealer and two roles.
        local_lane_visits: mul(3, sum(&[mul(8, k - 1)?, mul(2, k)?, k, mul(2, k)?, width])?)?,
        initialized_words: if k == 1 {
            0
        } else {
            sum(&[mul(3, width)?, mul(3, 2 * k + 1)?])?
        },
        // Root backing is separately owned by original native leases. Count its
        // retirement too; parent Boxes retire once, no repeated suffix scans.
        erased_words: if k == 1 {
            0
        } else {
            sum(&[mul(6, width)?, mul(3, 2 * k + 1)?])?
        },
        native_issue_aes: if k == 1 {
            0
        } else {
            sum(&[
                mul(
                    k - 1,
                    add(
                        comparison.paired_issue_aes_blocks,
                        select.paired_issue_aes_blocks,
                    )?,
                )?,
                coordinate.native.paired_issue_aes_blocks,
                inverse.lookup.paired_issue_aes_blocks,
                gap.paired_issue_aes_blocks,
            ])?
        },
        native_eval_aes: if k == 1 {
            0
        } else {
            mul(
                2,
                sum(&[
                    mul(
                        k - 1,
                        add(
                            comparison.evaluation_aes_blocks_per_party,
                            select.evaluation_aes_blocks_per_party,
                        )?,
                    )?,
                    coordinate.native.evaluation_aes_blocks_per_party,
                    inverse.lookup.evaluation_aes_blocks_per_party,
                    gap.evaluation_aes_blocks_per_party,
                ])?,
            )?
        },
        entropy_bytes: if k == 1 {
            0
        } else {
            sum(&[
                mul(8, width)?,
                mul(k - 1, add(comparison.random_bytes, select.random_bytes)?)?,
                coordinate.native.random_bytes,
                inverse.lookup.random_bytes,
                product.random_bytes,
                gap.random_bytes,
            ])?
        },
        public_hash_bytes: sum(&[
            if k == 1 {
                0
            } else {
                exp.preparation_hash_bytes
            },
            // Inherited child's own preparation+pin hashes are NOT present in
            // WorkQuote's floor arithmetic/AES categories. SourceLookup also
            // binds actual complete table words again at native issuance.
            if k == 1 {
                0
            } else {
                sum(&[
                    exp.preparation_hash_bytes,
                    mul(2, exp.issuance_pin_hash_bytes_per_lookup)?,
                    exp.native_table_payload_bytes_per_pair / 2,
                ])?
            },
            inverse.preparation_hash_bytes,
            inverse.issuance_pin_hash_bytes_per_lookup,
            if k == 1 {
                0
            } else {
                inverse.native_table_payload_bytes / 2
            },
            super::region::record_bytes(s)?,
        ])?,
        outer_exp_floor: if k == 1 {
            PublicFloorWork {
                entries: 0,
                taylor: 0,
                squarings: 0,
                multiply: 0,
                divide: 0,
                remainder: 0,
                add: 0,
                shift: 0,
            }
        } else {
            let proof = &exp.floors.proof;
            PublicFloorWork {
                entries: proof.entries,
                taylor: proof.taylor_steps,
                squarings: proof.squarings,
                multiply: proof.checked_multiplications,
                divide: proof.checked_divisions,
                remainder: proof.checked_remainders,
                add: proof.checked_additions,
                shift: proof.checked_shifts,
            }
        },
        inverse_verify: sum(&[
            inverse.verification.entries,
            inverse.verification.sentinel_checks,
            inverse.verification.quotient_checks,
            inverse.verification.checked_u128_products,
            inverse.verification.checked_u128_additions,
        ])?,
        coordinate_bridge: if k == 1 {
            0
        } else {
            sum(&[
                // Two root quote calls, plus the original three bridge queries.
                1 + coordinate.work.resource_queries,
                coordinate.work.declaration_checks,
                coordinate.work.original_owner_checks,
                coordinate.work.bindings_issued,
                coordinate.work.binding_digests,
                coordinate.work.descriptor_digests,
                coordinate.work.context_hashes,
                coordinate.work.input_word_reads,
                coordinate.work.coordinate_loans,
            ])?
        },
        child: child_work_value,
        total: 0,
    };
    work.total = [
        work.resource_queries,
        work.original_validation,
        work.original_retirement,
        work.outer_uses,
        work.fixed_record_initializations,
        work.local_lane_visits,
        work.initialized_words,
        work.erased_words,
        work.native_issue_aes,
        work.native_eval_aes,
        work.entropy_bytes,
        work.public_hash_bytes,
        work.inverse_verify,
        work.coordinate_bridge,
        work.outer_exp_floor.total()?,
    ]
    .into_iter()
    .try_fold(child_work(work.child)?, |a, b| {
        a.checked_add(b as u128).ok_or(Error::Resource)
    })?;
    Ok(CausalSoftmaxResources {
        incoming_owner_bytes: input.charged_bytes(),
        original_root_bytes,
        maximum_bytes,
        child_bytes,
        coordinate_bytes,
        inverse_bytes,
        product_bytes,
        gap_bytes,
        source_bytes,
        state_bytes,
        slot_bytes,
        coordination_bytes,
        transition_bytes,
        total_reserved_bytes,
        peer_bytes,
        work,
    })
}
