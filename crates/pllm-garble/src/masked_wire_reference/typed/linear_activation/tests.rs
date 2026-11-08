use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use std::sync::atomic::Ordering;

#[path = "../../../../tests/sigma_linear_activation/fixture.rs"]
mod fixture;
#[path = "../../../../tests/sigma_linear_activation/oracle.rs"]
mod independent;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
pub(super) fn spec(
    rows: usize,
    inner: usize,
    columns: usize,
    kind: ActivationKind,
) -> LinearActivationSpec {
    let edges = std::array::from_fn(|i| {
        EdgeDescriptor::new(
            EdgeIdentity::new(source(), [1; 32], [20 + i as u8; 32], [3 + i as u8; 32], 0),
            Shape::new(&match i {
                0 => [rows, inner],
                1 => [inner, columns],
                _ => [rows, columns],
            })
            .unwrap(),
            Numeric::new(50, if i == 2 { 24 } else { 12 }, Interpretation::Signed).unwrap(),
            Representation::CommonAdditiveMasked,
            (i == 1).then_some(WeightVisibility::Private),
        )
        .unwrap()
    });
    LinearActivationSpec::new(
        matrix::Spec {
            bits: 50,
            rows,
            inner,
            columns,
            layout: matrix::MatrixLayout::InputOutput,
        },
        edges,
        [[31; 32], [32; 32], [33; 32], [34; 32]],
        kind,
        [(-8192, 8192), (-8192, 8192)],
    )
    .unwrap()
}
pub(super) fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
pub(super) fn oracle(spec: LinearActivationSpec, x: &[i64], w: &[i64]) -> Vec<i128> {
    let table = match spec.activation.kind() {
        ActivationKind::Gelu => independent::Activation::Gelu.table(),
        ActivationKind::Silu => independent::Activation::Silu.table(),
    };
    let mut values = Vec::new();
    for row in 0..spec.matrix.rows {
        for column in 0..spec.matrix.columns {
            let mut sum = 0i128;
            for i in 0..spec.matrix.inner {
                let wi = match spec.matrix.layout {
                    matrix::MatrixLayout::InputOutput => i * spec.matrix.columns + column,
                    matrix::MatrixLayout::OutputInput => column * spec.matrix.inner + i,
                };
                sum += i128::from(x[row * spec.matrix.inner + i]) * i128::from(w[wi]);
            }
            let z = independent::rescale(sum).unwrap();
            values.push(independent::activate(z, &table));
        }
    }
    values
}
pub(super) fn unmask<B: AllocationContext>(
    mask: &DealerMaskFor<B, Ring>,
    output: &CommonMaskedFor<B, Ring>,
) -> Vec<i128> {
    assert_eq!(mask.binding, output.binding);
    output
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| {
            let value = i128::from(x.wrapping_sub(r) & ((1u64 << 50) - 1));
            if value >= 1i128 << 49 {
                value - (1i128 << 50)
            } else {
                value
            }
        })
        .collect()
}
fn finish<'b>(
    run: &mut LinearActivationRun<'b>,
    mut frames: [LinearActivationFrame<'b>; 2],
) -> [CommonMasked<'b, Ring>; 2] {
    for round in 0..EXCHANGES {
        let result = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        match result.body {
            StepBody::Frames(next, _metadata) => {
                assert!(round < EXCHANGES - 1);
                frames = next;
            }
            StepBody::Output(output, _metadata) => {
                assert_eq!(round, EXCHANGES - 1);
                return output;
            }
        }
    }
    panic!("missing complete output")
}

#[test]
fn owned_d22_reserved_exact_short_foreign_and_borrowed_layout_compatibility() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    let owned = linear_activation_resources_for::<OwnedBudgetContext>(s).unwrap();
    let borrowed = linear_activation_resources(s).unwrap();
    assert_eq!(size_of::<OwnedLocalDealer>(), size_of::<LocalDealer<'_>>());
    assert_eq!(
        size_of::<LinearActivationRunFor<OwnedBudgetContext>>(),
        size_of::<LinearActivationRun<'_>>()
    );
    assert_eq!(
        size_of::<ActivationPartyFor<OwnedBudgetContext>>(),
        size_of::<ActivationParty<'_>>()
    );
    assert_eq!(
        size_of::<CommonMaskedFor<OwnedBudgetContext, Ring>>(),
        common_charge()
    );
    assert_eq!(owned.root_bytes, borrowed.root_bytes);
    assert!(owned.reservation_bytes >= borrowed.reservation_bytes);
    for case in 0..3 {
        let budget = OwnedBudget::new(64 << 20).unwrap();
        let foreign = OwnedBudget::new(64 << 20).unwrap();
        let mut d = OwnedLocalDealer::new(source(), [1; 32], [2; 32], &budget).unwrap();
        let baseline = budget.budget().used_bytes();
        let amount = owned.reservation_bytes - usize::from(case == 0);
        let owner = if case == 1 { &foreign } else { &budget };
        let mut charge = owner.charge_external(amount).unwrap();
        let admitted = owner.budget().used_bytes();
        // Fill remaining capacity on the exact-fit case. Native admission must
        // move the existing slice rather than reserving a second native peak.
        let blocker = if case == 2 {
            Some(
                budget
                    .charge_external(
                        (64 << 20)
                            - budget.budget().used_bytes()
                            - size_of::<OwnedExternalCharge>(),
                    )
                    .unwrap(),
            )
        } else {
            None
        };
        let mut draws = 0;
        let result = d.linear_activation_from_reserved_values_with(
            s,
            &[1],
            &[4096],
            &mut charge,
            &mut |bytes| {
                draws += bytes.len();
                bytes.fill(173);
                Ok(())
            },
        );
        if case < 2 {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(draws, 0);
            assert_eq!(charge.payload_bytes(), amount);
            assert_eq!(owner.budget().used_bytes(), admitted);
        } else {
            let (mask, run) = result.unwrap();
            assert!(draws > 0);
            assert_eq!(charge.payload_bytes(), 0);
            assert!(run.is_budget(budget.budget()));
            assert!(std::ptr::eq(mask._lease.budget.budget(), budget.budget()));
            drop((mask, run));
        }
        drop((blocker, charge));
        assert_eq!(budget.budget().used_bytes(), baseline);
    }
}

#[test]
fn owned_d22_original_run_frames_and_outputs_outlive_root_and_dealer() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    let (mask, mut run) = {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let mut d = OwnedLocalDealer::new(source(), [1; 32], [2; 32], &root).unwrap();
        let mut charge = root
            .charge_external(
                linear_activation_resources_for::<OwnedBudgetContext>(s)
                    .unwrap()
                    .reservation_bytes,
            )
            .unwrap();
        d.linear_activation_from_reserved_values_with(
            s,
            &[-1],
            &[4096],
            &mut charge,
            &mut |bytes| {
                bytes.fill(173);
                Ok(())
            },
        )
        .unwrap()
    };
    assert!(run.is_budget(mask._lease.budget.budget()));
    let initial = run.begin(&mut || false).unwrap();
    let mut result = run
        .advance([initial[1].as_bytes(), initial[0].as_bytes()], &mut || {
            false
        })
        .unwrap();
    let output = loop {
        match result.body {
            StepBodyFor::Frames(next, _metadata) => {
                result = run
                    .advance([next[1].as_bytes(), next[0].as_bytes()], &mut || false)
                    .unwrap();
            }
            StepBodyFor::Output(output, _metadata) => break output,
        }
    };
    drop(run);
    for value in &output {
        assert_eq!(unmask(&mask, value), oracle(s, &[-1], &[4096]));
        assert!(std::ptr::eq(
            mask._lease.budget.budget(),
            value._lease.budget.budget()
        ));
    }
    // Initial original matrix frames still retain the real singleton even after
    // completion and destruction of the issuing run, dealer and root handle.
    assert!(matches!(
        matrix::party::IssuanceTicket::acquire(),
        Err(Error::Resource)
    ));
    drop(initial);
    drop(matrix::party::IssuanceTicket::acquire().unwrap());
    drop(output);
    let quote = OwnedBudget::resources().unwrap();
    assert_eq!(
        mask._lease.budget.budget().used_bytes(),
        quote.initial_reserved_bytes - quote.owner_metadata_bytes + mask._lease.bytes
    );
}

#[test]
fn complete_dot_negative_floor_mask_endpoints_and_both_layouts_match_integer_oracle() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for endpoint in [0, 255] {
            for layout in [
                matrix::MatrixLayout::InputOutput,
                matrix::MatrixLayout::OutputInput,
            ] {
                let budget = Budget::new(64 << 20).unwrap();
                let mut d = dealer(&budget);
                let mut s = spec(1, 2, 2, kind);
                s.matrix.layout = layout;
                // Square weights permit changing layout without changing shape.
                let x = [-1, -1];
                let w = [1, 4095, -1, -4095];
                let expected = oracle(s, &x, &w);
                let (mask, mut run) = d
                    .linear_activation_from_values_with(s, &x, &w, &mut |bytes| {
                        bytes.fill(endpoint);
                        Ok(())
                    })
                    .unwrap();
                let frames = run.begin(&mut || false).unwrap();
                let output = finish(&mut run, frames);
                for party in &output {
                    assert_eq!(unmask(&mask, party), expected);
                    assert_eq!(party.binding.consumer, s.operations[3]);
                }
                drop((mask, output, run));
                assert_eq!(
                    budget.used_bytes(),
                    size_of::<LocalDealer<'static>>() + (4 << 10)
                );
            }
        }
    }
    // Explicitly distinguish complete floor from forbidden per-term floors.
    assert_eq!((-1i128 - 4095).div_euclid(4096), -1);
    assert_eq!((-1i128).div_euclid(4096) + (-4095i128).div_euclid(4096), -2);
}

#[test]
fn conditional_quarter_ring_endpoints_and_invalid_domains() {
    let s = spec(1, 1, 1, ActivationKind::Silu);
    let make = |domains| {
        LinearActivationSpec::new(
            s.matrix,
            s.edges,
            s.operations,
            ActivationKind::Silu,
            domains,
        )
    };
    let negative = make([(-(1 << 48), -(1 << 48) + 1), (1, 2)]).unwrap();
    assert_eq!(
        negative.sum,
        Interval {
            lower: -(1 << 48),
            upper: -(1 << 48) + 1
        }
    );
    assert_eq!(negative.rescaled.lower, -(1 << 36));
    let positive = make([((1 << 48) - 1, 1 << 48), (1, 2)]).unwrap();
    assert_eq!(positive.sum.upper, 1 << 48);
    assert_eq!(positive.rescaled.upper, 1 << 36);
    for domains in [
        [(-(1 << 48) - 1, -(1 << 48)), (1, 2)],
        [(1 << 48, (1 << 48) + 1), (1, 2)],
        [(0, 0), (1, 2)],
        [(-(1 << 49) - 1, 0), (1, 2)],
    ] {
        assert!(matches!(make(domains), Err(Error::Domain)));
    }
    let mut operations = s.operations;
    operations[2] = operations[1];
    assert!(matches!(
        LinearActivationSpec::new(
            s.matrix,
            s.edges,
            operations,
            ActivationKind::Silu,
            [(-1, 1); 2]
        ),
        Err(Error::Context)
    ));
}

#[test]
fn preconstruction_metadata_quote_matches_independent_phase_inventory() {
    // Query before constructing any declaration, validating tables or hashing.
    let q = LinearActivationSpec::checked_logical_resources().unwrap();
    #[allow(dead_code)]
    struct Input {
        matrix: matrix::Spec,
        edges: [EdgeDescriptor; 5],
        operations: [[u8; 32]; 4],
        kind: ActivationKind,
        domains: [(i128, i128); 2],
    }
    #[allow(dead_code)]
    struct Geometry {
        input: [usize; 2],
        weight: [usize; 2],
        output: [usize; 2],
    }
    #[allow(dead_code)]
    struct Intervals {
        operands: [Interval; 2],
        sum: Interval,
        rescaled: Interval,
    }
    let core = EdgeContractResourceQuote::checked().unwrap();
    let activation = ActivationSpec::checked_logical_resources().unwrap();
    // Shared native inventory has independent owner-level slot checks.
    let native_matrix = matrix::party::PartyFor::<&Budget>::NATIVE_CALC.unwrap();
    let matrix = size_of::<matrix::party::Resources>() + native_matrix.max(size_of::<[usize; 3]>());
    let numeric = core
        .numeric_construction_bytes
        .max(size_of::<Result<Numeric, Error>>());
    let implementation = size_of::<[&[u8]; 3]>()
        + size_of::<Sha256>()
        + size_of::<[u8; 8]>()
        + size_of::<[u8; 32]>();
    let interval = size_of::<Intervals>()
        + size_of::<[Interval; 2]>().max(
            size_of::<[Interval; 2]>()
                + 2 * size_of::<std::array::IntoIter<i128, 2>>()
                + size_of::<Result<Interval, Error>>(),
        );
    let activation_phase = size_of::<Intervals>() + activation.constructor_peak_bytes;
    let returned = size_of::<Intervals>()
        + size_of::<ActivationSpec>() // named activation origin, not only Self field
        + size_of::<LinearActivationSpec>()
        + size_of::<Result<LinearActivationSpec, Error>>();
    let phases = [
        matrix,
        size_of::<Geometry>(),
        numeric,
        implementation,
        interval,
        activation_phase,
        returned,
    ];
    assert_eq!(q.input_bytes, size_of::<Input>());
    assert_eq!(q.matrix_resources_phase_bytes, matrix);
    assert_eq!(q.geometry_phase_bytes, size_of::<Geometry>());
    assert_eq!(q.numeric_validation_phase_bytes, numeric);
    assert_eq!(q.implementation_validation_phase_bytes, implementation);
    assert_eq!(q.interval_phase_bytes, interval);
    assert_eq!(q.activation_phase_bytes, activation_phase);
    assert_eq!(q.return_phase_bytes, returned);
    for phase in phases {
        assert!(q.peak_bytes >= q.input_bytes + phase);
    }
    assert_eq!(
        q.peak_bytes,
        size_of::<Input>() + phases.into_iter().max().unwrap()
    );
    assert!(q.peak_bytes < size_of::<Input>() + phases.into_iter().sum::<usize>());
    // The quote prices a record, not its native allocation/stack allowances.
    let native = matrix::party::resources(spec(1, 1, 1, ActivationKind::Silu).matrix).unwrap();
    assert!(q.matrix_resources_phase_bytes < native.reservation_bytes);
}

#[test]
fn metadata_sums_check_overflow_instead_of_wrapping() {
    assert_eq!(metadata_sum(&[]).unwrap(), 0);
    assert_eq!(metadata_sum(&[usize::MAX, 0]).unwrap(), usize::MAX);
    assert!(matches!(
        metadata_sum(&[usize::MAX, 1]),
        Err(Error::Resource)
    ));
    assert!(matches!(
        metadata_sum(&[1, usize::MAX]),
        Err(Error::Resource)
    ));
}

#[test]
fn resource_calculation_quote_matches_independent_phase_inventory() {
    use crate::compact_dpf::gpu127;
    let q = LinearActivationSpec::checked_resource_calculation_resources().unwrap();
    // Shared typed inventory has independent owner-level slot checks.
    let matrix = MatrixPartyFor::<&Budget>::TYPED_CALC.unwrap();
    let arithmetic_geometry = size_of::<arithmetic::Spec>()
        + size_of::<Result<(gpu127::Profile, u8, usize), Error>>()
        + size_of::<Result<gpu127::Profile, gpu127::Error>>()
        + size_of::<Result<gpu127::Profile, Error>>()
        + size_of::<gpu127::Profile>();
    let native_arithmetic = size_of::<arithmetic::Spec>()
        + size_of::<Result<arithmetic::Resources, Error>>()
        + arithmetic_geometry.max(
            size_of::<gpu127::Profile>()
                + size_of::<gpu127::Resources>()
                + size_of::<gpu127::Profile>().max(size_of::<arithmetic::Spec>()),
        );
    let rescale =
        size_of::<arithmetic::Spec>() + size_of::<Result<usize, Error>>() + native_arithmetic;
    // Nested activation owner has its own independent inventory regression.
    let activation = size_of::<ActivationResources>()
        + ActivationSpec::checked_resource_calculation_resources()
            .unwrap()
            .peak_bytes;
    let sum = size_of::<std::array::IntoIter<usize, 6>>();
    let returned = size_of::<LinearActivationResources>()
        + size_of::<Result<LinearActivationResources, Error>>();
    let phases = [matrix, rescale, activation, sum, returned];
    assert_eq!(q.input_bytes, size_of::<LinearActivationSpec>());
    assert_eq!(q.root_array_bytes, size_of::<[usize; 2]>());
    assert_eq!(q.matrix_phase_bytes, matrix);
    assert_eq!(q.rescale_phase_bytes, rescale);
    assert_eq!(q.activation_phase_bytes, activation);
    assert_eq!(q.sum_phase_bytes, sum);
    assert_eq!(q.return_phase_bytes, returned);
    assert_eq!(
        q.peak_bytes,
        q.input_bytes + q.root_array_bytes + phases.into_iter().max().unwrap()
    );
    assert!(q.peak_bytes < q.input_bytes + q.root_array_bytes + phases.into_iter().sum::<usize>());
    assert_eq!(q.peak_with_retained_bytes(0).unwrap(), q.peak_bytes);
    // Input is a destination Copy; retained original is an additional owner.
    assert_eq!(
        q.peak_with_retained_bytes(size_of::<LinearActivationSpec>())
            .unwrap(),
        q.peak_bytes + size_of::<LinearActivationSpec>()
    );
    let exact = usize::MAX - q.peak_bytes;
    assert_eq!(q.peak_with_retained_bytes(exact).unwrap(), usize::MAX);
    assert_eq!(q.peak_with_retained_bytes(exact + 1), Err(Error::Resource));
}

fn legacy_resource_output(s: LinearActivationSpec) -> Result<LinearActivationResources, Error> {
    let roots = [
        root_bytes(s.edges[0].shape().elements()),
        root_bytes(s.edges[1].shape().elements()),
    ];
    let matrix = matrix_reservation_bytes(s.matrix)?;
    let rescale = arithmetic_reservation_bytes::<Ring>(arithmetic::Spec {
        bits: 50,
        lanes: s.edges[2].shape().elements(),
        gate: arithmetic::Gate::GapArs { shift: 12 },
    })?;
    let activation = activation_resources(s.activation)?.combined_reservation_bytes;
    let run = size_of::<LinearActivationRun<'static>>() - 2 * size_of::<LinearParty<'static>>();
    let party = 2 * party_metadata_bytes();
    let frame = 2 * party_frame_bytes();
    let step = (EXCHANGES - 1) * step_charge(false) + step_charge(true);
    let workspace = transition_workspace_bytes();
    let wrapper = run + party + frame + step + workspace;
    let total = [roots[0], roots[1], matrix, rescale, activation, wrapper]
        .into_iter()
        .try_fold(0usize, |n, part| n.checked_add(part))
        .ok_or(Error::Resource)?;
    if total > crate::shared_allocation::MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(LinearActivationResources {
        reservation_bytes: total,
        root_bytes: roots,
        matrix_bytes: matrix,
        rescale_bytes: rescale,
        activation_bytes: activation,
        wrapper_bytes: wrapper,
        run_metadata_bytes: run,
        party_metadata_bytes: party,
        frame_wrapper_bytes: frame,
        step_wrapper_bytes: step,
        transition_workspace_bytes: workspace,
    })
}

#[test]
fn resource_calculation_geometry_errors_and_legacy_outputs_unchanged() {
    let q = LinearActivationSpec::checked_resource_calculation_resources().unwrap();
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for (rows, inner, columns) in [(1, 1, 1), (1, 256, 1), (16, 16, 16), (256, 1, 1)] {
            let original = spec(rows, inner, columns, kind);
            for layout in [
                matrix::MatrixLayout::InputOutput,
                matrix::MatrixLayout::OutputInput,
            ] {
                let mut s = original;
                s.matrix.layout = layout;
                let weight = s.edges[1];
                s.edges[1] = EdgeDescriptor::new(
                    weight.identity(),
                    Shape::new(&match layout {
                        matrix::MatrixLayout::InputOutput => [inner, columns],
                        matrix::MatrixLayout::OutputInput => [columns, inner],
                    })
                    .unwrap(),
                    weight.numeric(),
                    weight.representation(),
                    weight.weight_visibility(),
                )
                .unwrap();
                let s = LinearActivationSpec::new(
                    s.matrix,
                    s.edges,
                    s.operations,
                    kind,
                    [(-8192, 8192); 2],
                )
                .unwrap();
                assert_eq!(linear_activation_resources(s), legacy_resource_output(s));
                for rows in [0, usize::MAX] {
                    let mut invalid = s;
                    invalid.matrix.rows = rows;
                    assert_eq!(linear_activation_resources(invalid), Err(Error::Resource));
                    assert_eq!(
                        linear_activation_resources(invalid),
                        legacy_resource_output(invalid)
                    );
                }
                let mut invalid = s;
                invalid.matrix.bits = 0;
                assert_eq!(linear_activation_resources(invalid), Err(Error::Domain));
                assert_eq!(
                    linear_activation_resources(invalid),
                    legacy_resource_output(invalid)
                );
                assert_eq!(
                    LinearActivationSpec::checked_resource_calculation_resources().unwrap(),
                    q
                );
            }
        }
    }
}

#[test]
fn staged_constructor_preserves_metadata_and_rejection_order_across_geometries() {
    let quote = LinearActivationSpec::checked_logical_resources().unwrap();
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for (rows, inner, columns) in [(1, 1, 1), (1, 256, 1), (16, 16, 16), (256, 1, 1)] {
            let original = spec(rows, inner, columns, kind);
            for layout in [
                matrix::MatrixLayout::InputOutput,
                matrix::MatrixLayout::OutputInput,
            ] {
                let mut matrix = original.matrix;
                matrix.layout = layout;
                let mut edges = original.edges;
                let weight_shape = match layout {
                    matrix::MatrixLayout::InputOutput => [inner, columns],
                    matrix::MatrixLayout::OutputInput => [columns, inner],
                };
                let weight = edges[1];
                edges[1] = EdgeDescriptor::new(
                    weight.identity(),
                    Shape::new(&weight_shape).unwrap(),
                    weight.numeric(),
                    weight.representation(),
                    weight.weight_visibility(),
                )
                .unwrap();
                let make = |matrix, edges, operations, domains| {
                    LinearActivationSpec::new(matrix, edges, operations, kind, domains)
                };
                let rebuilt = make(matrix, edges, original.operations, [(-8192, 8192); 2]).unwrap();
                assert_eq!(rebuilt.edges, edges);
                assert_eq!(rebuilt.operations, original.operations);
                assert_eq!(rebuilt.operands, original.operands);
                assert_eq!(rebuilt.sum, original.sum);
                assert_eq!(rebuilt.rescaled, original.rescaled);
                assert_eq!(rebuilt.activation, original.activation);
                assert_eq!(
                    LinearActivationSpec::checked_logical_resources().unwrap(),
                    quote
                );
                for (rows, bits, expected) in [
                    (0, 50, Error::Resource),
                    (usize::MAX, 50, Error::Resource),
                    (matrix.rows, 0, Error::Domain),
                    (matrix.rows, 49, Error::Resource),
                ] {
                    let mut invalid = matrix;
                    invalid.rows = rows;
                    invalid.bits = bits;
                    assert!(
                        matches!(make(invalid, edges, original.operations, [(-1, 1); 2]), Err(e) if e == expected)
                    );
                }
                let mut wrong_shape = matrix;
                wrong_shape.layout = match layout {
                    matrix::MatrixLayout::InputOutput => matrix::MatrixLayout::OutputInput,
                    matrix::MatrixLayout::OutputInput => matrix::MatrixLayout::InputOutput,
                };
                if inner != columns {
                    assert!(matches!(
                        make(wrong_shape, edges, original.operations, [(-1, 1); 2]),
                        Err(Error::Domain)
                    ));
                }
                let mut aliased = original.operations;
                aliased[2] = aliased[1];
                assert!(matches!(
                    make(matrix, edges, aliased, [(0, 0); 2]),
                    Err(Error::Context)
                ));
                assert!(matches!(
                    make(matrix, edges, original.operations, [(0, 0); 2]),
                    Err(Error::Domain)
                ));
                assert!(matches!(
                    make(
                        matrix,
                        edges,
                        original.operations,
                        [(-(1 << 49), 1 << 49); 2]
                    ),
                    Err(Error::Domain)
                ));
                assert_eq!(
                    LinearActivationSpec::checked_logical_resources().unwrap(),
                    quote
                );
            }
        }
    }
}

#[test]
fn quarter_ring_and_floor_endpoints_execute_without_extra_rounding() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let template = spec(1, 1, 1, ActivationKind::Silu);
    for value in [-(1i64 << 48), (1i64 << 48) - 1, -4097, -4096, -1, 0, 4095] {
        let s = LinearActivationSpec::new(
            template.matrix,
            template.edges,
            template.operations,
            ActivationKind::Silu,
            [(i128::from(value), i128::from(value) + 1), (1, 2)],
        )
        .unwrap();
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut run) = d
            .linear_activation_from_values_with(s, &[value], &[1], &mut |bytes| {
                bytes.fill(255);
                Ok(())
            })
            .unwrap();
        let frames = run.begin(&mut || false).unwrap();
        let output = finish(&mut run, frames);
        for party in output {
            assert_eq!(unmask(&mask, &party), oracle(s, &[value], &[1]));
        }
    }
}

#[test]
fn exact_aggregate_one_byte_short_and_zero_draw_domain_rejections() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 2, 1, ActivationKind::Gelu);
    let cost = linear_activation_resources(s).unwrap();
    let dealer_bytes = size_of::<LocalDealer<'static>>() + (4 << 10);
    for short in [true, false] {
        let budget =
            Budget::new(dealer_bytes + cost.reservation_bytes - usize::from(short)).unwrap();
        let mut d = dealer(&budget);
        let mut draws = 0;
        let result = d.linear_activation_from_values_with(s, &[-1, -1], &[1, 4095], &mut |bytes| {
            // Entire operation admitted before first root draw. Later retirement
            // may refund genuinely dead dealer workspace, never reacquire it.
            if draws == 0 {
                assert_eq!(budget.used_bytes(), dealer_bytes + cost.reservation_bytes);
            }
            draws += 1;
            bytes.fill(17);
            Ok(())
        });
        if short {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(draws, 0);
        } else {
            let (mask, mut run) = result.unwrap();
            let frames = run.begin(&mut || false).unwrap();
            let output = finish(&mut run, frames);
            assert_eq!(unmask(&mask, &output[0]), oracle(s, &[-1, -1], &[1, 4095]));
            drop((mask, output, run));
        }
        assert_eq!(budget.used_bytes(), dealer_bytes);
        for values in [&[][..], &[8192, 0][..], &[-8193, 0][..]] {
            assert!(matches!(
                d.linear_activation_from_values_with(s, values, &[1, 2], &mut |_| {
                    panic!("domain rejection drew entropy")
                }),
                Err(Error::Domain)
            ));
        }
    }
}

#[test]
fn maximum_256_lanes_has_same_numeric_contract() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let s = spec(1, 1, 256, ActivationKind::Gelu);
    let weights: Vec<i64> = (0..256).map(|i| i * 63 - 8192).collect();
    let (mask, mut run) = d
        .linear_activation_from_values(s, &[-65], &weights)
        .unwrap();
    let frames = run.begin(&mut || false).unwrap();
    let output = finish(&mut run, frames);
    for party in output {
        assert_eq!(unmask(&mask, &party), oracle(s, &[-65], &weights));
    }
}

#[test]
fn oversized_geometry_and_counter_exhaustion_reject_before_draws() {
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    for matrix in [
        matrix::Spec {
            rows: 257,
            ..s.matrix
        },
        matrix::Spec {
            inner: 257,
            ..s.matrix
        },
        matrix::Spec {
            columns: 257,
            ..s.matrix
        },
        matrix::Spec {
            rows: usize::MAX,
            inner: usize::MAX,
            ..s.matrix
        },
    ] {
        assert!(LinearActivationSpec::new(
            matrix,
            s.edges,
            s.operations,
            ActivationKind::Gelu,
            [(-1, 1); 2]
        )
        .is_err());
    }
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    d.counter = u64::MAX - 21;
    assert!(matches!(
        d.linear_activation_from_values_with(s, &[0], &[0], &mut |_| {
            panic!("counter rejection drew entropy")
        }),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn entropy_and_allocation_faults_release_partial_issuance() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    // Cross roots, matrix native issuance, and arithmetic key issuance.
    for fault in 0..12 {
        let mut draws = 0;
        let result = d.linear_activation_from_values_with(s, &[1], &[4095], &mut |bytes| {
            if draws == fault {
                return Err(Error::Randomness);
            }
            draws += 1;
            bytes.fill(7);
            Ok(())
        });
        assert!(matches!(result, Err(Error::Randomness)));
        assert_eq!(budget.used_bytes(), baseline);
    }
    for fault in 0..12 {
        budget.fail_after.store(fault, Ordering::SeqCst);
        let result = d.linear_activation_from_values_with(s, &[1], &[4095], &mut |bytes| {
            bytes.fill(7);
            Ok(())
        });
        assert!(matches!(result, Err(Error::Allocation)));
        budget.fail_after.store(-1, Ordering::SeqCst);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn every_round_failure_burns_both_peers_and_retained_matrix_frame_keeps_singleton() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    for fail_round in 0..EXCHANGES {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, mut run) = d.linear_activation_from_values(s, &[-65], &[4095]).unwrap();
        let original = run.begin(&mut || false).unwrap();
        let mut next = None;
        for _ in 0..fail_round {
            let frames = next.as_ref().unwrap_or(&original);
            let StepBody::Frames(frames, _metadata) = run
                .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                .unwrap()
                .body
            else {
                panic!()
            };
            next = Some(frames);
        }
        let frames = next.as_ref().unwrap_or(&original);
        // Party0 advances first; party1 then fails. Neither can continue.
        assert!(run
            .advance([frames[1].as_bytes(), &[]], &mut || false)
            .is_err());
        assert!(run
            .parties
            .iter()
            .all(|p| p.phase == Phase::Dead && p.ledger.next == USES as u8));
        assert!(matches!(run.begin(&mut || false), Err(Error::Consumed)));
        let retained = budget.used_bytes();
        run.cancel();
        assert_eq!(budget.used_bytes(), retained);
        assert!(retained >= baseline + size_of::<LinearActivationRun<'static>>());
        drop((mask, next, run));
        assert!(d.linear_activation_from_values(s, &[1], &[1]).is_err());
        assert!(budget.used_bytes() > baseline);
        drop(original);
        assert_eq!(budget.used_bytes(), baseline);
        // Actual frame drop, not wrapper cancellation, releases singleton.
        let replacement = d.linear_activation_from_values(s, &[1], &[1]).unwrap();
        drop(replacement);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn explicitly_local_public_surface_preserves_opaque_outputs_and_budget_identity() {
    use crate::masked_wire_reference::typed::{
        local_linear_activation_resources, LocalLinearActivationFrame,
        LocalLinearActivationResources, LocalLinearActivationRun, LocalLinearActivationSpec,
        LocalLinearActivationStep,
    };
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let foreign = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    assert!(d.is_budget(&budget));
    assert!(!d.is_budget(&foreign));
    let template = spec(1, 2, 1, ActivationKind::Silu);
    // Caller declarations only; there is deliberately no source certificate or
    // certified provenance in this native test/API.
    let s: LocalLinearActivationSpec = LocalLinearActivationSpec::new(
        template.matrix,
        template.edges,
        template.operations,
        ActivationKind::Silu,
        [(-8192, 8192); 2],
    )
    .unwrap();
    let quote: LocalLinearActivationResources = local_linear_activation_resources(s).unwrap();
    assert_eq!(
        quote.component_bytes().into_iter().sum::<usize>(),
        quote.reservation_bytes()
    );
    assert_eq!(
        quote.wrapper_component_bytes().into_iter().sum::<usize>(),
        quote.component_bytes()[5]
    );
    let (mask, mut run): (_, LocalLinearActivationRun<'_>) = d
        .local_linear_activation_from_values(s, &[-1, -1], &[1, 4095])
        .unwrap();
    let mut frames: [LocalLinearActivationFrame<'_>; 2] = run.begin(&mut || false).unwrap();
    let outputs = loop {
        let step: LocalLinearActivationStep<'_> = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        match step.into_frames() {
            Ok(next) => frames = next,
            Err(step) => {
                assert!(step.frames().is_none());
                break step.into_output().ok().unwrap();
            }
        }
    };
    for output in &outputs {
        assert_eq!(output.descriptor(), template.edges[4]);
        assert_eq!(unmask(&mask, output), oracle(s, &[-1, -1], &[1, 4095]));
    }
    run.cancel();
    drop((outputs, mask, frames, run));
    assert_eq!(foreign.used_bytes(), 0);
}

#[test]
fn cancellation_between_peer_advances_retires_both() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (mask, mut run) = d
        .linear_activation_from_values(spec(1, 1, 1, ActivationKind::Silu), &[1], &[1])
        .unwrap();
    let frames = run.begin(&mut || false).unwrap();
    let mut polls = 0;
    assert!(matches!(
        run.advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || {
            polls += 1;
            polls == 2
        }),
        Err(Error::Consumed)
    ));
    assert!(run.parties.iter().all(|p| p.phase == Phase::Dead));
    drop((mask, run, frames));
}

#[test]
fn live_retained_frame_and_native_exhaustion_reject_before_any_root_draw() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Silu);
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let (mask, mut run) = d.linear_activation_from_values(s, &[1], &[1]).unwrap();
    assert!(matches!(
        d.linear_activation_from_values_with(s, &[1], &[1], &mut |_| {
            panic!("live matrix must reject before root entropy")
        }),
        Err(Error::Resource)
    ));
    let [retained, other] = run.begin(&mut || false).unwrap();
    drop((mask, run, other));
    let frame_only = budget.used_bytes();
    assert!(matches!(
        d.linear_activation_from_values_with(s, &[1], &[1], &mut |_| {
            panic!("retained native frame must reject before root entropy")
        }),
        Err(Error::Resource)
    ));
    assert_eq!(budget.used_bytes(), frame_only);
    drop(retained);
    assert_eq!(budget.used_bytes(), baseline);
    matrix::party::with_exhausted_issuance(|| {
        assert!(matches!(
            d.linear_activation_from_values_with(s, &[1], &[1], &mut |_| {
                panic!("exhausted native issuance must reject before root entropy")
            }),
            Err(Error::Resource)
        ));
        assert_eq!(budget.used_bytes(), baseline);
    });
    // Failed counter acquisition and failed root entropy both release the actual
    // guard. First draw also proves exclusive control was acquired beforehand.
    assert!(matches!(
        d.linear_activation_from_values_with(s, &[1], &[1], &mut |_| {
            assert!(matches!(
                matrix::party::IssuanceTicket::acquire(),
                Err(Error::Resource)
            ));
            Err(Error::Randomness)
        }),
        Err(Error::Randomness)
    ));
    assert_eq!(budget.used_bytes(), baseline);
    drop(matrix::party::IssuanceTicket::acquire().unwrap());
}

#[test]
fn incremental_inventory_counts_each_existing_inline_owner_once() {
    let s = spec(1, 2, 2, ActivationKind::Gelu);
    let c = linear_activation_resources(s).unwrap();
    let a = s.matrix.rows * s.matrix.inner;
    let b = s.matrix.inner * s.matrix.columns;
    let n = s.matrix.rows * s.matrix.columns;
    // Independent inventory: native peak quotes plus typed payload/wrappers,
    // then only genuinely new metadata. No linear quote helper reused here.
    let roots = 24 * (a + b)
        + 6 * ALLOCATOR
        + 4 * size_of::<CommonMasked<'static, Ring>>()
        + 2 * size_of::<DealerMask<'static, Ring>>()
        + 2 * size_of::<RootInput<'static>>();
    fn h<T>() -> usize {
        size_of::<T>() + std::mem::align_of::<T>() - 1
    }
    type C = &'static Budget;
    type G = CommonMaskedFor<C, Ring>;
    type T = OpaqueTyped<C>;
    // Finish-transfer slots cover distinct return boundaries and destructured
    // predecessors, not separate charges for fields embedded in those owners.
    let transfer = 4 * h::<Result<G, Error>>()
        + h::<G>()
        + h::<CommonBackingFor<C>>()
        + h::<matrix::party::MaskedOutputFor<C>>()
        + h::<matrix::party::OpaqueNative<C>>()
        + 2 * h::<T>()
        + 2 * h::<Result<T, Error>>()
        + h::<Binding>()
        + h::<u8>()
        + h::<std::marker::PhantomData<Ring>>()
        + h::<LeaseFor<C>>()
        + h::<Error>()
        + h::<&mut MatrixPartyFor<C>>()
        + h::<&[u8]>();
    assert_eq!(MatrixPartyFor::<C>::FINISHED_TRANSFER_BYTES, Some(transfer));
    let matrix = matrix::party::resources(s.matrix)
        .unwrap()
        .reservation_bytes
        + 2 * size_of::<MatrixParty<'static>>()
        + 2 * size_of::<CommonMasked<'static, Ring>>()
        + 2 * transfer
        + 8 * n
        + ALLOCATOR
        + size_of::<DealerMask<'static, Ring>>();
    let arithmetic = arithmetic::resources(s.arithmetic())
        .unwrap()
        .combined_reservation_bytes
        + 2 * size_of::<ArithmeticParty<'static, Ring>>()
        + 2 * size_of::<CommonMasked<'static, Ring>>()
        + size_of::<DealerMask<'static, Ring>>();
    let new_inline = size_of::<LinearActivationRun<'static>>()
        - 2 * (size_of::<MatrixParty<'static>>()
            + size_of::<ArithmeticParty<'static, Ring>>()
            + size_of::<ActivationParty<'static>>()
            + 3 * size_of::<CommonMasked<'static, Ring>>());
    let frames = 22 * size_of::<LinearActivationFrame<'static>>()
        - 2 * size_of::<matrix::party::Frame<'static>>()
        - 4 * size_of::<arithmetic::Frame<'static>>()
        - 16 * size_of::<ActivationFrame<'static>>();
    let steps = 11 * size_of::<LinearActivationStep<'static>>()
        - 20 * size_of::<LinearActivationFrame<'static>>()
        - 2 * size_of::<CommonMasked<'static, Ring>>();
    let workspace = size_of::<LinearActivationSpec>()
        + size_of::<LinearActivationResources>()
        + 4 * size_of::<Binding>()
        + size_of::<matrix::party::IssuanceTicket>()
        + 12 * size_of::<Lease<'static>>()
        + 2 * (size_of::<PartyStepFor<&'static Budget>>()
            - size_of::<LinearActivationFrame<'static>>()
                .min(size_of::<CommonMasked<'static, Ring>>()));
    assert_eq!(c.run_metadata_bytes + c.party_metadata_bytes, new_inline);
    assert_eq!(c.frame_wrapper_bytes, frames);
    assert_eq!(c.step_wrapper_bytes, steps);
    assert_eq!(c.transition_workspace_bytes, workspace);
    assert_eq!(
        c.reservation_bytes,
        roots
            + matrix
            + arithmetic
            + activation_resources(s.activation)
                .unwrap()
                .combined_reservation_bytes
            + new_inline
            + frames
            + steps
            + workspace
    );
}

#[test]
fn pinned_checkpoint_submatrix_executes_against_frozen_independent_oracle() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let source_bytes = fixture::source_bytes();
    assert_eq!(
        fixture::REVISION,
        "7ae557604adf67be50417f59c2c2f167def9a775"
    );
    assert_eq!(fixture::SOURCE_SHAPE_OI, [4864, 896]);
    assert_eq!(
        format!("{:x}", Sha256::digest(&source_bytes)),
        "780a723ba14459a03b2e72508debd6d4551d6ca13da75534c0f456de84c680de"
    );
    let x: Vec<i64> = fixture::input()
        .into_iter()
        .map(|v| independent::encode(v).unwrap() as i64)
        .collect();
    let w: Vec<i64> = fixture::weights_io()
        .into_iter()
        .map(|v| independent::encode(v).unwrap() as i64)
        .collect();
    assert_eq!(w.iter().copied().min(), Some(-280));
    assert_eq!(w.iter().copied().max(), Some(324));
    let wide_x: Vec<i128> = x.iter().map(|&v| i128::from(v)).collect();
    let wide_w: Vec<i128> = w.iter().map(|&v| i128::from(v)).collect();
    let z =
        independent::rescale(independent::dot(&wide_x, &wide_w, fixture::INNER).unwrap()).unwrap();
    // Source Qwen uses SiLU. GELU is an extra bounded operator fixture, never
    // relabelled original-model execution or source-certification authority.
    for kind in [ActivationKind::Silu, ActivationKind::Gelu] {
        let template = spec(1, fixture::INNER, 1, kind);
        let s = LinearActivationSpec::new(
            template.matrix,
            template.edges,
            template.operations,
            kind,
            [(-4096, 4097), (-280, 325)],
        )
        .unwrap();
        assert_eq!(
            s.sum,
            Interval {
                lower: -339_738_624,
                upper: 339_738_625
            }
        );
        let table = match kind {
            ActivationKind::Silu => independent::Activation::Silu.table(),
            ActivationKind::Gelu => independent::Activation::Gelu.table(),
        };
        let expected = independent::activate(z, &table);
        let mut wrong = table.clone();
        let index = z.div_euclid(64).abs().min(table.len() as i128 - 1) as usize;
        wrong[index] += 1;
        assert_ne!(independent::activate(z, &wrong), expected);
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut run) = d.linear_activation_from_values(s, &x, &w).unwrap();
        let frames = run.begin(&mut || false).unwrap();
        let output = finish(&mut run, frames);
        // Final D20 backing and completed orchestration do not inherit matrix
        // singleton ownership after the original matrix frames have dropped.
        drop(matrix::party::IssuanceTicket::acquire().unwrap());
        for party in output {
            assert_eq!(unmask(&mask, &party), [expected]);
        }
    }
}

#[test]
fn reserved_exact_full_transfer_and_one_byte_short_preserve_source_guard() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 2, 1, ActivationKind::Silu);
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    let metadata = size_of::<ExternalCharge<'static>>();
    let dealer_bytes = size_of::<LocalDealer<'static>>() + (4 << 10);
    for short in [true, false] {
        let total = dealer_bytes + metadata + native - usize::from(short);
        let budget = Budget::new(total).unwrap();
        let mut d = dealer(&budget);
        let mut source = budget.charge_external(native - usize::from(short)).unwrap();
        let mut draws = 0;
        let result = d.linear_activation_from_reserved_values_with(
            s,
            &[-1, -1],
            &[1, 4095],
            &mut source,
            &mut |bytes| {
                if draws == 0 {
                    assert_eq!(budget.used_bytes(), total);
                    assert!(matches!(budget.reserve(1), Err(Error::Resource)));
                }
                draws += 1;
                bytes.fill(17);
                Ok(())
            },
        );
        if short {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(draws, 0);
            assert_eq!(source.payload_bytes(), native - 1);
            assert_eq!(budget.used_bytes(), total);
        } else {
            let (mask, mut run) = result.unwrap();
            assert!(draws > 0);
            assert_eq!(source.payload_bytes(), 0);
            assert_eq!(source.charged_bytes(), metadata);
            let frames = run.begin(&mut || false).unwrap();
            let output = finish(&mut run, frames);
            assert_eq!(unmask(&mask, &output[0]), oracle(s, &[-1, -1], &[1, 4095]));
            drop((mask, run));
            assert!(budget.used_bytes() > dealer_bytes + metadata);
            // Outputs own their native slices even after source guard drops.
            drop(source);
            assert!(budget.used_bytes() > dealer_bytes);
            drop(output);
            assert_eq!(budget.used_bytes(), dealer_bytes);
            continue;
        }
        drop(source);
        assert_eq!(budget.used_bytes(), dealer_bytes);
    }
}

#[test]
fn reserved_preflight_rejections_draw_nothing_and_do_not_transfer() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    let budget = Budget::new(64 << 20).unwrap();
    let foreign = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let mut source = budget.charge_external(native).unwrap();
    let mut wrong = foreign.charge_external(native).unwrap();
    let baseline = budget.used_bytes();
    let counter = d.counter;
    assert!(matches!(
        d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut wrong, &mut |_| {
            panic!("foreign budget drew entropy")
        }),
        Err(Error::Resource)
    ));
    assert_eq!(wrong.payload_bytes(), native);
    assert_eq!(foreign.used_bytes(), wrong.charged_bytes());
    for payload in [0, 1] {
        let mut tiny = budget.charge_external(payload).unwrap();
        assert!(matches!(
            d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut tiny, &mut |_| {
                panic!("zero/one-byte source drew entropy")
            }),
            Err(Error::Resource)
        ));
        assert_eq!(tiny.payload_bytes(), payload);
        assert_eq!(budget.used_bytes(), baseline + tiny.charged_bytes());
        assert_eq!(d.counter, counter);
        drop(tiny);
    }
    for (x, w) in [
        (&[][..], &[1][..]),
        (&[8192][..], &[1][..]),
        (&[1][..], &[-8193][..]),
    ] {
        assert!(matches!(
            d.linear_activation_from_reserved_values_with(s, x, w, &mut source, &mut |_| {
                panic!("invalid clear input drew entropy")
            }),
            Err(Error::Domain)
        ));
        assert_eq!(source.payload_bytes(), native);
    }
    d.plan = [99; 32];
    assert!(matches!(
        d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut source, &mut |_| {
            panic!("foreign binding drew entropy")
        }),
        Err(Error::Context)
    ));
    d.plan = [1; 32];
    d.counter = u64::MAX - 21;
    assert!(matches!(
        d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut source, &mut |_| {
            panic!("counter exhaustion drew entropy")
        }),
        Err(Error::Resource)
    ));
    d.counter = counter;
    let ticket = matrix::party::IssuanceTicket::acquire().unwrap();
    assert!(matches!(
        d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut source, &mut |_| {
            panic!("live singleton drew entropy")
        }),
        Err(Error::Resource)
    ));
    drop(ticket);
    matrix::party::with_exhausted_issuance(|| {
        assert!(matches!(
            d.linear_activation_from_reserved_values_with(s, &[1], &[1], &mut source, &mut |_| {
                panic!("issuance exhaustion drew entropy")
            }),
            Err(Error::Resource)
        ));
    });
    assert_eq!(d.counter, counter);
    assert_eq!(source.payload_bytes(), native);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn reserved_post_transfer_entropy_and_allocation_faults_consume_only_native_slice() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Gelu);
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let retained = 13;
    for allocation in [false, true] {
        for fault in 0..12 {
            let mut source = budget.charge_external(native + retained).unwrap();
            let mut draws = 0;
            if allocation {
                budget.fail_after.store(fault, Ordering::SeqCst);
            }
            let result = d.linear_activation_from_reserved_values_with(
                s,
                &[1],
                &[4095],
                &mut source,
                &mut |bytes| {
                    if !allocation && draws == fault {
                        return Err(Error::Randomness);
                    }
                    draws += 1;
                    bytes.fill(7);
                    Ok(())
                },
            );
            budget.fail_after.store(-1, Ordering::SeqCst);
            assert_eq!(
                result.err(),
                Some(if allocation {
                    Error::Allocation
                } else {
                    Error::Randomness
                })
            );
            assert_eq!(source.payload_bytes(), retained);
            assert_eq!(budget.used_bytes(), baseline + source.charged_bytes());
            drop(matrix::party::IssuanceTicket::acquire().unwrap());
            drop(source);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn reserved_cancel_or_peer_fault_keeps_source_and_retained_native_frame_alive() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();
    let s = spec(1, 1, 1, ActivationKind::Silu);
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    for cancel in [true, false] {
        for source_first in [true, false] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let mut source = budget.charge_external(native + 13).unwrap();
            // Exercise the narrow public adapter, not only deterministic test entry.
            let (mask, mut run) = d
                .local_linear_activation_from_reserved_values(s, &[1], &[1], &mut source)
                .unwrap();
            let [retained, other] = run.begin(&mut || false).unwrap();
            assert_eq!(source.payload_bytes(), 13);
            let mut source = Some(source);
            if source_first {
                drop(source.take());
            }
            let source_bytes = source.as_ref().map_or(0, ExternalCharge::charged_bytes);
            if cancel {
                run.cancel();
            } else {
                assert!(run.advance([other.as_bytes(), &[]], &mut || false).is_err());
            }
            assert!(matches!(run.begin(&mut || false), Err(Error::Consumed)));
            drop((mask, run, other));
            assert!(budget.used_bytes() > baseline + source_bytes);
            assert!(matches!(
                matrix::party::IssuanceTicket::acquire(),
                Err(Error::Resource)
            ));
            drop(retained);
            assert_eq!(budget.used_bytes(), baseline + source_bytes);
            drop(matrix::party::IssuanceTicket::acquire().unwrap());
            drop(source);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn reserved_matches_legacy_transcript_except_unique_matrix_issuance_binding() {
    let _guard = matrix::TEST_SERIAL.lock().unwrap();

    #[derive(Debug, PartialEq, Eq)]
    struct Transcript {
        requests: Vec<usize>,
        frames: Vec<[Vec<u8>; 2]>,
        mask: Vec<u64>,
        masked: [Vec<u64>; 2],
        bindings: [Binding; 3],
        roles: [u8; 2],
        counter: u64,
    }

    fn execute(s: LinearActivationSpec, reserved: bool) -> Transcript {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        // Dealer construction uses OS entropy. Pin its private test identity so
        // only admission strategy and native singleton issuance differ.
        d.namespace = [77; 32];
        let baseline = budget.used_bytes();
        let x = [-1, -4095];
        let w = [1, 4095, -1, -4095];
        let mut requests = Vec::new();
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut fill = |bytes: &mut [u8]| {
            requests.push(bytes.len());
            for byte in bytes {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state as u8;
            }
            Ok(())
        };
        let issued = if reserved {
            let native = linear_activation_resources(s).unwrap().reservation_bytes();
            let mut source = budget.charge_external(native).unwrap();
            let issued = d
                .linear_activation_from_reserved_values_with(s, &x, &w, &mut source, &mut fill)
                .unwrap();
            assert_eq!(source.payload_bytes(), 0);
            // Native owners must not borrow the source guard.
            drop(source);
            issued
        } else {
            d.linear_activation_from_values_with(s, &x, &w, &mut fill)
                .unwrap()
        };
        let (mask, mut run) = issued;
        let mut frames = run.begin(&mut || false).unwrap();
        let mut transcript_frames = Vec::new();
        let output = loop {
            transcript_frames.push(std::array::from_fn(|i| frames[i].as_bytes().to_vec()));
            let step = run
                .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
                .unwrap();
            match step.into_frames() {
                Ok(next) => frames = next,
                Err(step) => break step.into_output().ok().unwrap(),
            }
        };
        assert_eq!(transcript_frames.len(), EXCHANGES);
        for party in &output {
            assert_eq!(unmask(&mask, party), oracle(s, &x, &w));
        }
        let transcript = Transcript {
            requests,
            frames: transcript_frames,
            mask: mask.backing.values().to_vec(),
            masked: std::array::from_fn(|i| output[i].backing.values().to_vec()),
            bindings: [mask.binding, output[0].binding, output[1].binding],
            roles: [output[0].role, output[1].role],
            counter: d.counter,
        };
        drop((mask, run, frames, output));
        assert_eq!(budget.used_bytes(), baseline);
        transcript
    }

    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for layout in [
            matrix::MatrixLayout::InputOutput,
            matrix::MatrixLayout::OutputInput,
        ] {
            let mut s = spec(1, 2, 2, kind);
            s.matrix.layout = layout;
            let legacy = execute(s, false);
            let mut reserved = execute(s, true);
            for party in 0..2 {
                // Separate real native issuances MUST have different matrix
                // context bindings, despite identical injected entropy. Compare
                // every other byte, then normalize ONLY that mandatory 32-byte
                // first-frame field. No payload or later-frame bytes excluded.
                assert_eq!(&legacy.frames[0][party][..8], b"PLLMMX01");
                assert_eq!(&reserved.frames[0][party][..8], b"PLLMMX01");
                assert_ne!(
                    &legacy.frames[0][party][8..40],
                    &reserved.frames[0][party][8..40]
                );
                assert_eq!(
                    &legacy.frames[0][party][40..],
                    &reserved.frames[0][party][40..]
                );
                reserved.frames[0][party][8..40].copy_from_slice(&legacy.frames[0][party][8..40]);
            }
            assert_eq!(legacy, reserved);
        }
    }
}
