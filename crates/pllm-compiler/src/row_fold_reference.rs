//! Clear-only ownership boundary for the reusable caller-buffered row program.
//! All acquisition is upstream: guards must bind complete original allocations,
//! including allocator/transition obligations, not merely the slices supplied here.
//! This checks caller declarations/continuity; it cannot manufacture provenance.

use crate::encrypted_level_schedule::construction::ControlLedger;
use pllm_core::row_fold::{Error, Geometry, Invocation, State};
use pllm_garble::shared_allocation::{Budget, ExternalCharge};

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
        for (i, g) in guards.iter().enumerate() {
            if !g.is_for(budget)
                || g.payload_bytes() < requirements[i]
                || guards[..i].iter().any(|old| std::ptr::eq(*old, *g))
            {
                return Err(Error::LifetimeMismatch);
            }
        }
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
