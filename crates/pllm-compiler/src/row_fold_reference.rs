//! Clear-only ownership boundary for the reusable caller-buffered row program.
//! All acquisition is upstream: guards must bind complete original allocations,
//! including allocator/transition obligations, not merely the slices supplied here.
//! This checks caller declarations/continuity; it cannot manufacture provenance.

use crate::encrypted_level_schedule::construction::ControlLedger;
use pllm_core::row_fold::{Error, Geometry, Invocation, State};
use pllm_core::row_normalization_reference as normalization;
use pllm_garble::shared_allocation::{Budget, ExternalCharge};
use pllm_models::{ModelOperation, ModelOperator};

fn check_original_guards<const N: usize>(
    guards: [&ExternalCharge<'_>; N],
    requirements: [usize; N],
    budget: &Budget,
) -> Result<(), Error> {
    for (i, g) in guards.iter().enumerate() {
        if !g.is_for(budget)
            || g.payload_bytes() < requirements[i]
            || guards[..i].iter().any(|old| std::ptr::eq(*old, *g))
        {
            return Err(Error::LifetimeMismatch);
        }
    }
    Ok(())
}

pub struct BorrowedClearBuffers<'a> {
    pub input: &'a [f64],
    pub public: &'a mut [f64],
    pub workspace: &'a mut [f64],
    /// Three distinct already-admitted original allocations, in field order.
    pub original_guards: [&'a ExternalCharge<'a>; 3],
    /// Prepaid caller control region, retained through invocation cleanup.
    pub control_guard: &'a ExternalCharge<'a>,
}

pub struct ChargedRowFold<'a> {
    // Drop invocation (payload cleanup) before any guard binding can end.
    invocation: Invocation<'a>,
    _guards: [&'a ExternalCharge<'a>; 4],
    _budget: &'a Budget,
}
impl<'a> ChargedRowFold<'a> {
    pub fn new(
        geometry: Geometry,
        buffers: BorrowedClearBuffers<'a>,
        state: &'a mut State,
        control: &mut ControlLedger,
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        let quote = geometry.quote()?;
        let guards = [
            buffers.original_guards[0],
            buffers.original_guards[1],
            buffers.original_guards[2],
            buffers.control_guard,
        ];
        let requirements = [
            buffers.input.len() * 8,
            buffers.public.len() * 8,
            buffers.workspace.len() * 8,
            quote.control_bytes + std::mem::size_of::<Self>(),
        ];
        check_original_guards(guards, requirements, budget)?;
        let invocation = Invocation::new(
            geometry,
            buffers.input,
            buffers.public,
            buffers.workspace,
            state,
            control.core_work(),
        )?;
        Ok(Self {
            invocation,
            _guards: guards,
            _budget: budget,
        })
    }
    pub fn execute(&mut self) -> Result<&[f64], Error> {
        self.invocation.execute()
    }
}

/// Explicit CLEAR reference policy, never implicit reinterpretation of decoder
/// LayerNorm attrs or activation of any source/model/SDK/native HE path.
pub const CENTERED_ENERGY_REFERENCE_POLICY: &str =
    "row_head_ascending_tree_binary64_no_epsilon_reference_v1";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceStatus {
    ResearchReference,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalizationRefusal {
    IncompatibleOperatorPolicy,
    UnknownOriginalCapacity,
    OriginalOwnerRefused,
    ControlStorageRefused,
    Arithmetic(normalization::Error),
}
impl From<normalization::Error> for NormalizationRefusal {
    fn from(e: normalization::Error) -> Self {
        Self::Arithmetic(e)
    }
}
/// Immutable operation capture retained with the output. IDs/digests confer no
/// tensor authenticity. Gamma/beta are original immutable PUBLIC affine owners.
/// Positional bindings are operation.inputs[0]=clear.input, [1]=gamma, [2]=beta;
/// no source lookup, clone, import, or authenticated-tensor upgrade occurs here.
pub struct BorrowedNormalizationBuffers<'a> {
    pub clear: BorrowedClearBuffers<'a>,
    pub gamma: &'a [f64],
    pub beta: &'a [f64],
    pub affine_guards: [&'a ExternalCharge<'a>; 2],
    /// Separate upstream admission for the FULL original DTO allocation region,
    /// including nested capacities/staging; not just its visible field lengths.
    /// This declaration is not source authentication or retrospective acquisition.
    pub operation_guard: &'a ExternalCharge<'a>,
    pub operation_original_bytes: Option<usize>,
}
pub struct ChargedNormalization<'a> {
    reference: normalization::Reference<'a>,
    operation: &'a ModelOperation,
    guards: [&'a ExternalCharge<'a>; 7],
    budget: &'a Budget,
}
pub struct RetainedNormalization<'a> {
    output: normalization::Output<'a>,
    operation: &'a ModelOperation,
    _guards: [&'a ExternalCharge<'a>; 7],
    _budget: &'a Budget,
}
impl RetainedNormalization<'_> {
    pub fn values(&self) -> &[f64] {
        self.output.values()
    }
    pub fn operation(&self) -> &ModelOperation {
        self.operation
    }
    pub fn status(&self) -> ReferenceStatus {
        ReferenceStatus::ResearchReference
    }
}
fn check_normalization_policy(op: &ModelOperation) -> Result<(), NormalizationRefusal> {
    let bad = NormalizationRefusal::IncompatibleOperatorPolicy;
    if op.operator != ModelOperator::LayerNorm
        || op.output_shape.as_slice() != [32, 1024]
        || op.inputs.len() != 3
        || op.id.is_empty()
        || op.id.len() > 64
        || op.layer.is_some()
        || op.state_kind.is_some()
    {
        return Err(bad);
    }
    if op.inputs.iter().any(|s| s.is_empty() || s.len() > 64)
        || op.inputs[0] == op.inputs[1]
        || op.inputs[0] == op.inputs[2]
        || op.inputs[1] == op.inputs[2]
    {
        return Err(bad);
    }
    let attrs = op.attributes.as_object().ok_or(bad)?;
    // Exact bounded key set. Numeric epsilon (including zero), float32,
    // inside-sqrt epsilon, alternate variance or fusion cannot be elided.
    if attrs.len() != 11
        || attrs.get("axis").and_then(|v| v.as_i64()) != Some(-1)
        || attrs.get("active_width").and_then(|v| v.as_u64()) != Some(768)
    {
        return Err(bad);
    }
    for (key, expected) in [
        ("reference_policy", CENTERED_ENERGY_REFERENCE_POLICY),
        ("dtype", "float64"),
        ("layout", "rows32_block1024_active768"),
        ("epsilon_policy", "none_reject_zero_energy"),
        ("variance", "centered_energy"),
        ("affine_order", "normalize_multiply_add"),
        (
            "arithmetic",
            "ordered_nonfused_binary64_row_head_ascending_tree",
        ),
        ("sqrt", "clear_sqrt"),
        ("status", "research_reference"),
    ] {
        if attrs.get(key).and_then(|v| v.as_str()) != Some(expected) {
            return Err(bad);
        }
    }
    Ok(())
}
impl<'a> ChargedRowFold<'a> {
    /// Narrow whole-region constructor using this module's existing original-
    /// owner checks and the core consuming row-fold continuation, not another
    /// centered kernel. Complete policy+prefix+affine work prepaid before loops.
    pub fn normalization_reference(
        operation: &'a ModelOperation,
        buffers: BorrowedNormalizationBuffers<'a>,
        state: &'a mut State,
        control: &mut ControlLedger,
        budget: &'a Budget,
    ) -> Result<ChargedNormalization<'a>, NormalizationRefusal> {
        let geometry = Geometry::new(32, 1024, 768).map_err(normalization::Error::from)?;
        let payment = normalization::prepay(geometry, control.core_work(), 1024)?;
        check_normalization_policy(operation)?;
        let operation_bytes = buffers
            .operation_original_bytes
            .ok_or(NormalizationRefusal::UnknownOriginalCapacity)?;
        if operation_bytes < std::mem::size_of::<ModelOperation>()
            || operation_bytes > super::encrypted_level_schedule::construction::METADATA_BYTES
        {
            return Err(NormalizationRefusal::OriginalOwnerRefused);
        }
        let clear = buffers.clear;
        let guards = [
            clear.original_guards[0],
            clear.original_guards[1],
            clear.original_guards[2],
            clear.control_guard,
            buffers.affine_guards[0],
            buffers.affine_guards[1],
            buffers.operation_guard,
        ];
        let control_bytes = normalization::control_bytes(geometry)?
            + std::mem::size_of::<ChargedNormalization<'static>>()
            + std::mem::size_of::<RetainedNormalization<'static>>()
            + std::mem::size_of::<BorrowedNormalizationBuffers<'static>>();
        if control_bytes > pllm_core::row_fold::CONTROL_BYTES {
            return Err(NormalizationRefusal::ControlStorageRefused);
        }
        if budget.used_bytes() > super::encrypted_level_schedule::construction::OWNER_BYTES {
            return Err(NormalizationRefusal::OriginalOwnerRefused);
        }
        let requirements = [
            clear.input.len().checked_mul(8),
            clear.public.len().checked_mul(8),
            clear.workspace.len().checked_mul(8),
            Some(control_bytes),
            buffers.gamma.len().checked_mul(8),
            buffers.beta.len().checked_mul(8),
            Some(operation_bytes),
        ];
        let mut bytes = [0; 7];
        for (n, b) in requirements.into_iter().zip(&mut bytes) {
            *b = n.ok_or(NormalizationRefusal::OriginalOwnerRefused)?;
        }
        check_original_guards(guards, bytes, budget)
            .map_err(|_| NormalizationRefusal::OriginalOwnerRefused)?;
        let reference = normalization::Reference::new_prepaid(
            payment,
            clear.input,
            buffers.gamma,
            buffers.beta,
            clear.public,
            clear.workspace,
            state,
        )?;
        Ok(ChargedNormalization {
            reference,
            operation,
            guards,
            budget,
        })
    }
}
impl<'a> ChargedNormalization<'a> {
    pub fn execute(self) -> Result<RetainedNormalization<'a>, NormalizationRefusal> {
        let output = self.reference.execute()?;
        Ok(RetainedNormalization {
            output,
            operation: self.operation,
            _guards: self.guards,
            _budget: self.budget,
        })
    }
}

#[cfg(test)]
#[path = "row_normalization_reference_tests.rs"]
mod normalization_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypted_level_schedule::construction::{
        quote_public_ciphertext_construction, Backing, Element,
        PublicCiphertextConstructionManifest,
    };
    use pllm_core::row_fold::WorkLedger;
    #[test]
    fn row_fold_retains_full_original_pool_charge_and_counter_continuity() {
        let budget = Budget::new(1 << 20).unwrap();
        let input = [1.0; 16];
        let mut public = [0.0; 128];
        let mut pool = [0.0; 96];
        let a = budget
            .charge_external(std::mem::size_of_val(&input))
            .unwrap();
        let masks = budget
            .charge_external(std::mem::size_of_val(&public))
            .unwrap();
        // Full original backing obligation deliberately exceeds visible slice.
        let workspace = budget.charge_external(16384).unwrap();
        let control = budget.charge_external(65536).unwrap();
        let before = budget.used_bytes();
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(1 << 20, 1 << 20);
        let mut control_work = ControlLedger::new(&mut work);
        {
            let mut run = ChargedRowFold::new(
                Geometry::new(2, 8, 6).unwrap(),
                BorrowedClearBuffers {
                    input: &input,
                    public: &mut public,
                    workspace: &mut pool,
                    original_guards: [&a, &masks, &workspace],
                    control_guard: &control,
                },
                &mut state,
                &mut control_work,
                &budget,
            )
            .unwrap();
            assert!(run.execute().unwrap().iter().all(|&x| x == 0.0));
            assert_eq!(budget.used_bytes(), before);
        }
        assert_eq!(budget.used_bytes(), before); // Logical releases did not refund.
        assert!(pool.iter().chain(&public).all(|&x| x == 0.0));
    }
    #[test]
    fn row_fold_wrong_budget_and_duplicate_guard_refuse_before_writes() {
        let b = Budget::new(1 << 20).unwrap();
        let other = Budget::new(1 << 20).unwrap();
        let input = [1.0; 16];
        let mut public = [7.0; 128];
        let mut pool = [7.0; 96];
        let a = b.charge_external(16384).unwrap();
        let ctl = b.charge_external(65536).unwrap();
        let mut state = State::Fresh;
        let mut work = WorkLedger::new(1 << 20, 1 << 20);
        let mut control_work = ControlLedger::new(&mut work);
        for budget in [&b, &other] {
            assert!(matches!(
                ChargedRowFold::new(
                    Geometry::new(2, 8, 6).unwrap(),
                    BorrowedClearBuffers {
                        input: &input,
                        public: &mut public,
                        workspace: &mut pool,
                        original_guards: [&a, &a, &a],
                        control_guard: &ctl,
                    },
                    &mut state,
                    &mut control_work,
                    budget
                ),
                Err(Error::LifetimeMismatch)
            ));
        }
        assert!(pool.iter().chain(&public).all(|&x| x == 7.0));
        assert_eq!(state, State::Fresh);
        assert_eq!(work.remaining(), 1 << 20);
    }

    #[test]
    fn row_fold_compiler_control_and_core_share_exact_aggregate_before_entry() {
        // Independent ten-backing empty construction inventory: two IDs and the
        // eight outer arrays. No graph factory/validator or full slot fixture.
        let mut backings = [Backing {
            element: Element::Byte,
            length: 1,
            capacity: Some(1),
            first_phase: 0,
            end_phase: 1,
        }; 10];
        for (b, element) in backings[2..].iter_mut().zip([
            Element::Owner,
            Element::String,
            Element::Value,
            Element::Operation,
            Element::Capability,
            Element::String,
            Element::String,
            Element::Step,
        ]) {
            b.element = element;
            b.length = 0;
            b.capacity = Some(0);
        }
        let manifest = PublicCiphertextConstructionManifest {
            owners: 0,
            persistent_owners: 0,
            input_ranks: &[],
            operations: &[],
            capabilities: &[],
            outputs: 0,
            retained_inputs: 0,
            release_lengths: &[],
            backings: &backings,
            other_owners: &[],
            phases: 1,
            manifest_original_bytes: Some(4096),
        };
        // Quote control 256+32768+10*8 =33104; tiny core542+64528=65070.
        let budget = Budget::new(1 << 20).unwrap();
        let input = [1.0];
        let mut public = [7.0; 2];
        let mut pool = [7.0; 6];
        let a = budget.charge_external(8).unwrap();
        let masks = budget.charge_external(16).unwrap();
        let workspace = budget.charge_external(48).unwrap();
        let ctl = budget.charge_external(65536).unwrap();
        for limit in [98_173, 98_174] {
            let mut work = WorkLedger::new(limit, 1 << 20);
            let mut control = ControlLedger::new(&mut work);
            let _quote = quote_public_ciphertext_construction(&manifest, &mut control).unwrap();
            assert_eq!(control.core_work().remaining(), limit - 33_104);
            let mut state = State::Fresh;
            {
                let run = ChargedRowFold::new(
                    Geometry::new(1, 1, 1).unwrap(),
                    BorrowedClearBuffers {
                        input: &input,
                        public: &mut public,
                        workspace: &mut pool,
                        original_guards: [&a, &masks, &workspace],
                        control_guard: &ctl,
                    },
                    &mut state,
                    &mut control,
                    &budget,
                );
                if limit == 98_173 {
                    assert!(matches!(run, Err(Error::WorkRefused)));
                } else {
                    assert_eq!(run.unwrap().execute().unwrap(), &[0.0]);
                }
            }
            if limit == 98_173 {
                assert_eq!(state, State::Fresh);
                assert!(pool.iter().chain(&public).all(|&v| v == 7.0));
                assert_eq!(control.core_work().remaining(), 65_069);
            } else {
                assert_eq!(control.core_work().remaining(), 0);
                assert!(pool.iter().chain(&public).all(|&v| v == 0.0));
            }
        }
    }
}
