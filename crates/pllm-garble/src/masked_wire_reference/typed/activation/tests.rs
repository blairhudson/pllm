use super::*;
use pllm_core::edge_contract::Shape;
use std::sync::atomic::Ordering;

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}
fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}
fn edge(wire: u8, lanes: usize) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [1; 32], [wire + 64; 32], [wire; 32], 19),
        Shape::new(&[lanes]).unwrap(),
        Numeric::new(50, 12, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap()
}
fn spec(kind: ActivationKind, lanes: usize) -> ActivationSpec {
    ActivationSpec::new(kind, [7; 32], edge(3, lanes), edge(4, lanes), [8; 32]).unwrap()
}

#[test]
fn resource_calculation_quote_matches_independent_phase_inventory() {
    use crate::compact_dpf::gpu127;
    let q = ActivationSpec::checked_resource_calculation_resources().unwrap();
    let m = ActivationSpec::checked_logical_resources().unwrap();
    let core = EdgeContractResourceQuote::checked().unwrap();
    let a = size_of::<arithmetic::Resources>();
    let s = size_of::<selection::Resources>();
    let l = size_of::<source_lookup::Resources>();
    // Enumerate native helper objects independently; never use capacity values.
    let arithmetic_geometry = size_of::<arithmetic::Spec>()
        + size_of::<Result<(gpu127::Profile, u8, usize), Error>>()
        + size_of::<Result<gpu127::Profile, gpu127::Error>>()
        + size_of::<Result<gpu127::Profile, Error>>()
        + size_of::<gpu127::Profile>();
    let arithmetic = size_of::<arithmetic::Spec>()
        + size_of::<Result<arithmetic::Resources, Error>>()
        + arithmetic_geometry.max(
            size_of::<gpu127::Profile>()
                + size_of::<gpu127::Resources>()
                + size_of::<gpu127::Profile>().max(size_of::<arithmetic::Spec>()),
        );
    let selection = 2 * size_of::<selection::Spec>()
        + size_of::<Result<selection::Resources, Error>>()
        + (size_of::<selection::Spec>() + size_of::<Result<(), Error>>())
            .max(size_of::<[usize; 5]>());
    let lookup_phases = [
        size_of::<source_lookup::Spec>() + size_of::<Result<(), Error>>(),
        size_of::<gpu127::Resources>()
            + (size_of::<Result<gpu127::Profile, gpu127::Error>>()
                + size_of::<Result<gpu127::Profile, Error>>())
            .max(size_of::<gpu127::Profile>()),
        size_of::<gpu127::Resources>() + s + selection,
        size_of::<gpu127::Resources>()
            + s
            + size_of::<[usize; 14]>()
            + 2 * size_of::<source_lookup::Spec>(),
    ];
    let lookup = size_of::<source_lookup::Spec>()
        + size_of::<Result<source_lookup::Resources, Error>>()
        + lookup_phases.into_iter().max().unwrap();
    assert_eq!(arithmetic_resource_calculation_bytes().unwrap(), arithmetic);
    assert_eq!(selection_resource_calculation_bytes().unwrap(), selection);
    assert_eq!(lookup_resource_calculation_bytes().unwrap(), lookup);
    let validation_phases = [
        m.table_validation_scratch_bytes,
        m.identity_validation_scratch_bytes,
        m.collision_validation_scratch_bytes,
        m.wire_digest_scratch_bytes,
        m.wire_constructor_scratch_bytes,
        size_of::<Numeric>() + core.numeric_construction_bytes,
        core.consumer_use_construction_bytes,
    ];
    let revalidation = m.workspace_bytes
        + size_of::<Result<(), Error>>()
        + validation_phases.into_iter().max().unwrap();
    let costs = size_of::<[usize; 7]>();
    let native_costs = 2 * costs
        + size_of::<Result<[usize; 7], Error>>()
        + [arithmetic, selection, lookup].into_iter().max().unwrap();
    // Explicit t,d,p,v,l,f destinations plus nested helper Result/scratch.
    let record_phases = [
        a + arithmetic,
        2 * a + arithmetic,
        2 * a + s + selection,
        2 * a + 2 * s + selection,
        2 * a + 2 * s + l + lookup,
        2 * a + 3 * s + l + selection,
    ];
    let native_records = costs + record_phases.into_iter().max().unwrap();
    let returned = costs
        + 2 * a
        + 3 * s
        + l
        + size_of::<ActivationResources>()
        + size_of::<Result<ActivationResources, Error>>();
    let phases = [revalidation, native_costs, native_records, returned];
    assert_eq!(q.input_bytes, size_of::<ActivationSpec>());
    assert_eq!(q.revalidation_phase_bytes, revalidation);
    assert_eq!(q.native_costs_phase_bytes, native_costs);
    assert_eq!(q.native_records_phase_bytes, native_records);
    assert_eq!(q.return_phase_bytes, returned);
    assert_eq!(
        q.peak_bytes,
        q.input_bytes + phases.into_iter().max().unwrap()
    );
    assert!(q.peak_bytes < q.input_bytes + phases.into_iter().sum::<usize>());
    assert_eq!(q.peak_with_retained_bytes(0).unwrap(), q.peak_bytes);
    let exact = usize::MAX - q.peak_bytes;
    assert_eq!(q.peak_with_retained_bytes(exact).unwrap(), usize::MAX);
    assert_eq!(q.peak_with_retained_bytes(exact + 1), Err(Error::Resource));
    assert_eq!(resource_metadata_sum(&[]).unwrap(), 0);
    assert_eq!(resource_metadata_sum(&[usize::MAX, 0]).unwrap(), usize::MAX);
    assert_eq!(
        resource_metadata_sum(&[usize::MAX, 1]),
        Err(Error::Resource)
    );
}

// Independent pre-refactor resource calculation. No native_costs helper or new
// pre-calculation quote participates; native outputs retain their original order.
fn legacy_resources(s: ActivationSpec) -> Result<ActivationResources, Error> {
    s.validate()?;
    let n = s.input.shape().elements();
    let t = arithmetic::resources(arithmetic::Spec {
        bits: 38,
        lanes: n,
        gate: arithmetic::Gate::Tr { shift: 6 },
    })?;
    let d = arithmetic::resources(arithmetic::Spec {
        bits: 32,
        lanes: n,
        gate: arithmetic::Gate::Drelu,
    })?;
    let p = selection::resources(selection::Spec { bits: 32, lanes: n })?;
    let v = selection::resources(selection::Spec {
        bits: s.kind.index_bits(),
        lanes: n,
    })?;
    let l = source_lookup::resources(source_lookup::Spec {
        index_bits: s.kind.index_bits(),
        output_bits: 50,
        lanes: n,
        boundary: selection::Boundary::Deferred,
    })?;
    let f = selection::resources_for(
        selection::Spec { bits: 50, lanes: n },
        selection::Boundary::Deferred,
    )?;
    let native = t.combined_reservation_bytes
        + 2 * d.combined_reservation_bytes
        + p.combined_reservation_bytes
        + v.combined_reservation_bytes
        + l.combined_reservation_bytes
        + f.combined_reservation_bytes;
    let party = size_of::<ActivationParty<'static>>()
        + common_charge()
        + 8 * (size_of::<ActivationFrame<'static>>() + size_of::<ActivationStep<'static>>())
        + 6 * (8 * n + ALLOCATOR)
        + 3 * (76 + 8 * n + ALLOCATOR)
        + (16 << 10);
    let workspace = 2 * size_of::<ActivationParty<'static>>()
        + 2 * size_of::<UseLedger<20>>()
        + 17 * size_of::<Binding>()
        + 4 * (8 * n + ALLOCATOR)
        + (32 << 10);
    let wrapper = 2 * party + mask_charge() + 8 * n + ALLOCATOR + workspace;
    let root = 2 * (8 * n + ALLOCATOR + common_charge())
        + 8 * n
        + ALLOCATOR
        + mask_charge()
        + size_of::<RootInput<'static>>();
    let random = t.random_bytes
        + 2 * d.random_bytes
        + p.random_bytes
        + v.random_bytes
        + l.random_bytes
        + f.random_bytes;
    Ok(ActivationResources {
        lanes: n,
        exchanges: 8,
        total_peer_frame_bytes: 1224 + 144 * n,
        random_bytes: random,
        party_material_payload_bytes: t.party_material_payload_bytes
            + 2 * d.party_material_payload_bytes
            + p.party_material_payload_bytes
            + v.party_material_payload_bytes
            + l.party_material_payload_bytes
            + f.party_material_payload_bytes,
        combined_reservation_bytes: native + wrapper,
        native_reservation_bytes: native,
        wrapper_reservation_bytes: wrapper,
        clear_input_reservation_bytes: native + wrapper + root,
        clear_input_random_bytes: 8 * n + random,
    })
}

#[test]
fn resource_calculation_quote_geometry_and_legacy_outputs() {
    let before = ActivationSpec::checked_resource_calculation_resources().unwrap();
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for lanes in [1, 17, 256] {
            let s = spec(kind, lanes);
            assert_eq!(activation_resources(s), legacy_resources(s));
            assert_eq!(
                ActivationSpec::checked_resource_calculation_resources().unwrap(),
                before
            );
            let mut invalid = s;
            invalid.output = invalid.input;
            assert_eq!(activation_resources(invalid), Err(Error::Context));
            assert_eq!(activation_resources(invalid), legacy_resources(invalid));
        }
    }
    let mut invalid = spec(ActivationKind::Silu, 1);
    invalid.input = edge(3, 257);
    invalid.output = edge(4, 257);
    assert_eq!(activation_resources(invalid), Err(Error::Domain));
    assert_eq!(
        ActivationSpec::checked_resource_calculation_resources().unwrap(),
        before
    );
}

// Pre-workspace sequence: intentionally independent of the new stage/schedule
// helpers, including the old argument-evaluation and array construction order.
fn legacy_numeric(s: &ActivationSpec) -> [u8; 32] {
    digest(&[
        b"pllm.sigma.typed.activation-numeric.v1\0",
        b"EzPC:aa0f2f860266fb79901405ef5e9965b8c9fe2e88:mode1",
        b"LocalAssumption:X=[-2^37,2^37);50/Q12;38/Q12;TR6;32/Q6;A=unsigned[0,2^31]",
        &[s.kind.index_bits()],
        &s.kind.table_digest(),
    ])
}
fn legacy_declaration(s: &ActivationSpec) -> [u8; 32] {
    digest(&[
        DOMAIN.as_bytes(),
        &s.operation,
        &s.input.digest(),
        &s.output.digest(),
        &s.next_consumer,
        &legacy_numeric(s),
    ])
}
fn legacy_stage(s: &ActivationSpec, index: u8) -> [u8; 32] {
    digest(&[
        b"pllm.sigma.typed.activation-stage.v1\0",
        &legacy_declaration(s),
        &[index],
    ])
}
fn legacy_metadata(s: &ActivationSpec) -> ([EdgeDescriptor; 17], [ConsumerUse; 20], [u8; 32]) {
    let mut wires = [s.input; 17];
    for (i, edge) in wires.iter_mut().enumerate() {
        let (bits, frac, interpretation) = match i {
            0 => (38, 12, Interpretation::Signed),
            1 | 3 | 5 => (32, 6, Interpretation::Signed),
            4 => (32, 6, Interpretation::Unsigned),
            2 | 6 | 7 | 15 => (1, 0, Interpretation::Predicate),
            8..=10 => (s.kind.index_bits(), 6, Interpretation::Unsigned),
            _ => (50, 12, Interpretation::Signed),
        };
        let id = s.input.identity();
        *edge = EdgeDescriptor::new(
            EdgeIdentity::new(
                id.source(),
                id.plan(),
                s.operation,
                digest(&[
                    b"pllm.sigma.typed.activation-wire.v1\0",
                    &legacy_declaration(s),
                    &[i as u8],
                ]),
                id.first_tensor_index(),
            ),
            s.input.shape(),
            Numeric::new(bits, frac, interpretation).unwrap(),
            if matches!(i, 11 | 12 | 13 | 16) {
                Representation::AdditiveShare
            } else if interpretation == Interpretation::Predicate {
                Representation::CommonXorMasked
            } else {
                Representation::CommonAdditiveMasked
            },
            None,
        )
        .unwrap();
    }
    let edges = [
        s.input, wires[0], wires[1], wires[2], wires[1], wires[3], wires[1], wires[4], wires[5],
        wires[6], wires[5], wires[7], wires[8], wires[9], wires[10], wires[2], s.input, wires[12],
        wires[11], wires[13],
    ];
    let stages = [
        0, 1, 2, 3, 3, 5, 5, 7, 8, 9, 10, 11, 11, 13, 14, 15, 15, 17, 17, 19,
    ];
    let operands = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0];
    let uses = std::array::from_fn(|i| {
        ConsumerUse::new(edges[i], legacy_stage(s, stages[i]), operands[i], i as u32)
    });
    let mut h = Sha256::new();
    h.update(legacy_declaration(s));
    h.update(arithmetic::DOMAIN.as_bytes());
    h.update(selection::DOMAIN.as_bytes());
    h.update(source_lookup::DOMAIN.as_bytes());
    h.update(Profile::DOMAIN.as_bytes());
    for edge in &wires {
        h.update(edge.digest());
    }
    for slot in &uses {
        h.update(slot.digest());
    }
    (wires, uses, h.finalize().into())
}

#[test]
fn metadata_workspace_reuse_preserves_legacy_sequences() {
    let mut workspace = ActivationMetadataWorkspace::new(&edge(99, 7));
    for kind in [
        ActivationKind::Gelu,
        ActivationKind::Silu,
        ActivationKind::Gelu,
    ] {
        for lanes in [1, 17, 256] {
            let expected = spec(kind, lanes);
            let s = ActivationSpec::new_with_workspace(
                kind,
                expected.operation,
                expected.input,
                expected.output,
                expected.next_consumer,
                &mut workspace,
            )
            .unwrap();
            assert_eq!(s, expected);
            assert_eq!(s.numeric_contract(), legacy_numeric(&s));
            assert_eq!(s.declaration(&mut workspace.core), legacy_declaration(&s));
            for stage in 0..20 {
                assert_eq!(s.stage(stage), legacy_stage(&s, stage));
            }
            let (wires, uses, hash) = legacy_metadata(&s);
            assert_eq!(s.digest_with_workspace(&mut workspace), hash);
            assert_eq!(workspace.wires, wires);
            assert_eq!(workspace.uses, uses);
            assert_eq!(s.digest(), hash);
            assert_eq!(s.wires().unwrap(), wires);
            assert_eq!(s.uses(&wires), uses);
        }
    }
    let s = spec(ActivationKind::Gelu, 1);
    assert_eq!(
        ActivationSpec::new_with_workspace(
            s.kind,
            s.operation,
            s.input,
            s.input,
            s.next_consumer,
            &mut workspace,
        )
        .err(),
        Some(Error::Context)
    );
    assert_eq!(s.digest_with_workspace(&mut workspace), s.digest());
}

#[test]
fn metadata_quote_matches_independent_phase_formulas() {
    // Available before constructing a declaration or hashing either table.
    let before_construction = ActivationSpec::checked_logical_resources().unwrap();
    let s = spec(ActivationKind::Gelu, 1);
    let q = s.logical_resources().unwrap();
    assert_eq!(before_construction, q);
    let core = EdgeContractResourceQuote::checked().unwrap();
    let h = size_of::<Sha256>();
    let d = size_of::<[u8; 32]>();
    let r = size_of::<&[u8]>();
    let numeric = 1 + d + 5 * r + h + 8 + d;
    let declaration = (2 * d).max(2 * d + numeric).max(3 * d + 6 * r + h + 8 + d);
    let indexed = declaration.max(d + 1 + 3 * r + h + 8 + d);
    let wire_constructor = d
        + size_of::<EdgeIdentity>()
        + core
            .edge_identity_construction_bytes
            .max(size_of::<EdgeIdentity>() + size_of::<Shape>() + core.numeric_construction_bytes)
            .max(core.edge_descriptor_construction_bytes);
    assert_eq!(q.spec_bytes, size_of::<ActivationSpec>());
    assert_eq!(q.workspace_bytes, size_of::<ActivationMetadataWorkspace>());
    // Workspace includes both fixed output arrays plus the one shared core owner.
    assert!(
        q.workspace_bytes
            >= 17 * size_of::<EdgeDescriptor>()
                + 20 * size_of::<ConsumerUse>()
                + EdgeContractWorkspace::logical_bytes()
    );
    assert_eq!(q.table_validation_scratch_bytes, h + 8 + 2 * d);
    assert_eq!(
        q.identity_validation_scratch_bytes,
        size_of::<ActivationIdentityValidation>() + 3 * r + h + 8
    );
    assert_eq!(q.collision_validation_scratch_bytes, 4 * d);
    assert_eq!(q.numeric_digest_scratch_bytes, numeric);
    assert_eq!(
        q.edge_digest_scratch_bytes,
        core.descriptor_digest_bytes
            - size_of::<EdgeDescriptor>()
            - EdgeContractWorkspace::logical_bytes()
    );
    assert_eq!(
        q.use_digest_scratch_bytes,
        core.use_digest_bytes - size_of::<ConsumerUse>() - EdgeContractWorkspace::logical_bytes()
    );
    assert_eq!(q.edge_digest_scratch_bytes, d);
    assert_eq!(q.use_digest_scratch_bytes, d);
    assert_eq!(q.declaration_digest_scratch_bytes, declaration);
    assert_eq!(q.stage_digest_scratch_bytes, indexed);
    assert_eq!(q.wire_digest_scratch_bytes, indexed);
    assert_eq!(q.wire_constructor_scratch_bytes, wire_constructor);
    assert_eq!(
        q.use_constructor_scratch_bytes,
        core.consumer_use_construction_bytes
    );
    let nested = [
        declaration,
        indexed,
        wire_constructor,
        core.consumer_use_construction_bytes,
        d,
    ]
    .into_iter()
    .max()
    .unwrap();
    assert_eq!(q.activation_digest_scratch_bytes, h + nested);
    let constructor_scratch = [
        h + 8 + 2 * d,
        q.identity_validation_scratch_bytes,
        4 * d,
        indexed,
        wire_constructor,
        size_of::<Numeric>() + core.numeric_construction_bytes,
        core.consumer_use_construction_bytes,
    ]
    .into_iter()
    .max()
    .unwrap();
    assert_eq!(q.constructor_input_bytes, size_of::<ActivationSpec>());
    assert_eq!(
        q.constructor_result_bytes,
        size_of::<Result<ActivationSpec, Error>>()
    );
    assert_eq!(
        q.constructor_peak_bytes,
        q.constructor_input_bytes
            + q.constructor_result_bytes
            + q.workspace_bytes
            + constructor_scratch
    );
    assert_eq!(
        q.digest_peak_bytes,
        q.spec_bytes + q.workspace_bytes + h + nested
    );
    assert_eq!(
        q.peak_bytes,
        q.constructor_peak_bytes.max(q.digest_peak_bytes)
    );
    // Metadata footprint is independent of lanes and activation table length.
    assert_eq!(
        q,
        spec(ActivationKind::Silu, 256).logical_resources().unwrap()
    );
}
fn oracle(kind: ActivationKind, values: &[i64]) -> Vec<i128> {
    // Independent signed wide-integer floor/absolute/clamp. This does not use
    // native ring helpers or the region's clipping implementation.
    let k = 1i128 << kind.index_bits();
    values
        .iter()
        .map(|&x| {
            let y = i128::from(x).div_euclid(64);
            let index = y.abs().min(k - 1) as usize;
            (if y >= 0 { i128::from(x) } else { 0 }) - i128::from(kind.table()[index])
        })
        .collect()
}
fn unmask(mask: &DealerMask<'_, Ring>, output: &CommonMasked<'_, Ring>) -> Vec<i128> {
    assert_eq!(mask.binding, output.binding);
    output
        .backing
        .values()
        .iter()
        .zip(mask.backing.values())
        .map(|(&x, &r)| {
            let value = i128::from(x.wrapping_sub(r) & super::super::super::mask(50));
            if value >= 1i128 << 49 {
                value - (1i128 << 50)
            } else {
                value
            }
        })
        .collect()
}
fn next<'b>(
    parties: &mut [ActivationParty<'b>; 2],
    frames: &[ActivationFrame<'b>; 2],
) -> [ActivationFrame<'b>; 2] {
    let [a, b] = parties;
    match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Frame(a), ActivationStep::Frame(b)) => [a, b],
        _ => panic!("unexpected final output"),
    }
}
fn begin<'b>(
    parties: &mut [ActivationParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [ActivationFrame<'b>; 2] {
    let [x0, x1] = inputs;
    [parties[0].begin(x0).unwrap(), parties[1].begin(x1).unwrap()]
}
fn run<'b>(
    mut parties: [ActivationParty<'b>; 2],
    inputs: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, Ring>; 2] {
    let mut frames = begin(&mut parties, inputs);
    let mut total = 0;
    let n = parties[0].spec.input.shape().elements();
    let expected_magic: [&[u8]; 8] = [
        b"PLLMSA01",
        b"PLLMSA01",
        b"PLLMSA01",
        b"PLLMSL01",
        b"PLLMSA01",
        b"PLLMSL01",
        b"PLLMLU01",
        b"PLLMAC01",
    ];
    for (phase, &magic) in expected_magic.iter().enumerate() {
        for frame in &frames {
            let bytes = frame.as_bytes();
            assert_eq!(&bytes[..8], magic);
            assert_eq!(
                bytes.len(),
                if phase == 6 { 80 + 16 * n } else { 76 + 8 * n }
            );
            total += bytes.len();
        }
        if phase < 7 {
            frames = next(&mut parties, &frames);
        }
    }
    assert_eq!(total, 1224 + 144 * n);
    for (role, frame) in frames.iter().enumerate() {
        assert_eq!(frame.as_bytes()[72], role as u8);
        assert_eq!(frame.as_bytes()[73], 8);
    }
    let [a, b] = &mut parties;
    let outputs = match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
        _ => panic!("missing final reconstruction"),
    };
    for party in &mut parties {
        assert_eq!(party.advance(&[]).err(), Some(Error::Consumed));
        assert!(party.x.is_none() && party.y.is_none() && party.d.is_none());
        assert!(party.local.is_empty());
    }
    outputs
}

#[test]
fn signed_endpoints_negative_bins_and_clip_edges_match_wide_oracle() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let k = 1i64 << kind.index_bits();
            let values = [
                -(1i64 << 37),
                -(1i64 << 37) + 1,
                -64 * (k + 1),
                -64 * k,
                -64 * (k - 1),
                -129,
                -128,
                -127,
                -65,
                -64,
                -63,
                -1,
                0,
                1,
                63,
                64,
                65,
                64 * (k - 1),
                64 * k,
                64 * (k + 1),
                (1i64 << 37) - 1,
            ];
            let s = spec(kind, values.len());
            let (mask, parties, inputs) = d.activation_from_values(s, &values).unwrap();
            let outputs = run(parties, inputs);
            let expected = oracle(kind, &values);
            for output in &outputs {
                assert_eq!(output.descriptor(), s.output);
                assert_eq!(output.binding.consumer, [8; 32]);
                assert_eq!(unmask(&mask, output), expected);
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn gelu_explicit_floor_examples_and_minimum_unsigned_absolute_residue() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let values = [-1, 63, 64, -(1i64 << 37)];
    let s = spec(ActivationKind::Gelu, values.len());
    let (mask, mut parties, inputs) = d.activation_from_values(s, &values).unwrap();
    let mut frames = begin(&mut parties, inputs);
    // Three deliveries reach prefix Select; fourth computes affine C=A-K.
    for _ in 0..4 {
        frames = next(&mut parties, &frames);
    }
    for party in &parties {
        assert!(party.phase == Phase::ClipDrelu);
        assert_eq!(
            party.wires[A].descriptor.numeric().interpretation(),
            Interpretation::Unsigned
        );
        assert_eq!(party.wires[A].interval.unwrap().upper, (1i128 << 31) + 1);
        assert!(party.x.is_some());
    }
    for _ in 0..3 {
        frames = next(&mut parties, &frames);
    }
    let [a, b] = &mut parties;
    let outputs = match (
        a.advance(frames[1].as_bytes()).unwrap(),
        b.advance(frames[0].as_bytes()).unwrap(),
    ) {
        (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
        _ => panic!("expected output"),
    };
    assert_eq!(&unmask(&mask, &outputs[0])[..3], &[-31, 63, 33]);
    assert_eq!(
        unmask(&mask, &outputs[1]),
        oracle(ActivationKind::Gelu, &values)
    );
}

#[test]
fn full_ring_forced_draws_preserve_affine_and_predicate_masks() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let mut words = vec![0, 1];
        for width in [38, 32, 8, 10, 50] {
            words.extend([1u64 << (width - 1), (1u64 << width) - 1]);
        }
        for word in words {
            for bit in [0u8, 1] {
                let budget = Budget::new(64 << 20).unwrap();
                let mut d = dealer(&budget);
                let values = [-(1i64 << 37), -65, -1, 0, 63, 64, (1i64 << 37) - 1];
                let s = spec(kind, values.len());
                // u64 draws exercise full-ring masks; other root/nonce draws
                // remain deterministic test data. Alternate bit parity explicitly.
                let (mask, parties, inputs) = d
                    .activation_from_values_with(s, &values, |bytes| {
                        if bytes.len() == 8 {
                            bytes.copy_from_slice(&((word & !1) | u64::from(bit)).to_le_bytes());
                        } else {
                            bytes.fill(bit);
                        }
                        Ok(())
                    })
                    .unwrap();
                let outputs = run(parties, inputs);
                assert_eq!(unmask(&mask, &outputs[0]), oracle(kind, &values));
                assert_eq!(unmask(&mask, &outputs[1]), oracle(kind, &values));
            }
        }
    }
    assert_eq!(affine_mask(u32::MAX as u64, 1), u32::MAX as u64 - 2);
}

#[test]
fn complete_clear_preflight_is_zero_draw_and_zero_allocation() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let s = spec(ActivationKind::Gelu, 1);
    budget.fail_after.store(0, Ordering::SeqCst);
    for values in [vec![], vec![1i64 << 37], vec![-(1i64 << 37) - 1]] {
        let mut draws = 0;
        let result = d.activation_from_values_with(s, &values, |_| {
            draws += 1;
            Ok(())
        });
        assert_eq!(result.err(), Some(Error::Domain));
        assert_eq!(draws, 0);
        assert_eq!(budget.used_bytes(), baseline);
    }
    assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    budget.fail_after.store(-1, Ordering::SeqCst);
}

#[test]
fn immutable_hashes_profiles_geometry_and_wire_alias_rejected() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        kind.validate_table().unwrap();
    }
    let s = spec(ActivationKind::Gelu, 1);
    let t = spec(ActivationKind::Silu, 1);
    assert_ne!(s.digest(), t.digest());
    assert_ne!(s.numeric_contract(), t.numeric_contract());
    let wires = s.wires().unwrap();
    for index in [L, F, O] {
        assert_eq!(wires[index].representation(), Representation::AdditiveShare);
    }
    assert!(ActivationSpec::new(s.kind, s.operation, s.input, s.input, s.next_consumer).is_err());
    assert!(
        ActivationSpec::new(s.kind, s.operation, s.input, edge(4, 2), s.next_consumer).is_err()
    );
    assert!(ActivationSpec::new(
        s.kind,
        s.operation,
        edge(3, 257),
        edge(4, 257),
        s.next_consumer
    )
    .is_err());
    let wrong = EdgeDescriptor::new(
        s.input.identity(),
        s.input.shape(),
        Numeric::new(50, 11, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        None,
    )
    .unwrap();
    assert!(ActivationSpec::new(s.kind, s.operation, wrong, s.output, s.next_consumer).is_err());
}

fn root_bytes(lanes: usize) -> usize {
    2 * (8 * lanes + ALLOCATOR + common_charge())
        + 8 * lanes
        + ALLOCATOR
        + mask_charge()
        + size_of::<RootInput<'static>>()
}
#[test]
fn exact_aggregate_budget_and_one_byte_short_for_one_and_256_lanes() {
    for lanes in [1, 256] {
        for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
            let s = spec(kind, lanes);
            let cost = activation_resources(s).unwrap();
            let dealer_bytes = size_of::<LocalDealer<'static>>() + (4 << 10);
            let exact = dealer_bytes + root_bytes(lanes) + cost.combined_reservation_bytes;
            let values = vec![0; lanes];
            let budget = Budget::new(exact).unwrap();
            {
                let mut d = dealer(&budget);
                let (mask, parties, inputs) = d.activation_from_values(s, &values).unwrap();
                let outputs = run(parties, inputs);
                assert_eq!(unmask(&mask, &outputs[0]), vec![0; lanes]);
            }
            assert_eq!(budget.used_bytes(), 0);
            let budget = Budget::new(exact - 1).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let mut draws = 0;
            assert_eq!(
                d.activation_from_values_with(s, &values, |_| {
                    draws += 1;
                    Ok(())
                })
                .err(),
                Some(Error::Resource)
            );
            assert_eq!(draws, 0);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn wrong_external_role_binding_and_session_burn_local_party() {
    for fault in 0..3 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut parties, [mut x0, x1]) = d
            .activation_from_values(spec(ActivationKind::Gelu, 1), &[64])
            .unwrap();
        match fault {
            0 => x0.role = 1,
            1 => x0.binding.mask_id[0] ^= 1,
            _ => x0.binding.session[0] ^= 1,
        }
        assert_eq!(parties[0].begin(x0).err(), Some(Error::Context));
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        assert!(parties[0].x.is_none());
        parties[1].cancel();
        drop((x1, mask, parties));
    }
}

#[test]
fn ordered_slots_replay_and_failure_burn() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let (_, mut parties, inputs) = d
        .activation_from_values(spec(ActivationKind::Gelu, 1), &[1])
        .unwrap();
    let frames = begin(&mut parties, inputs);
    assert_eq!(
        parties[0].advance(frames[0].as_bytes()).err(),
        Some(Error::Context)
    );
    assert_eq!(
        parties[0].advance(frames[1].as_bytes()).err(),
        Some(Error::Consumed)
    );
    parties[1].cancel();
    let s = spec(ActivationKind::Gelu, 1);
    let (_, mut parties, _) = d.activation_from_values(s, &[1]).unwrap();
    assert_eq!(parties[0].burn(1), Err(Error::Consumed));
    assert_eq!(parties[0].burn(0), Err(Error::Consumed));
    parties[0].cancel();
    parties[1].cancel();
}

#[test]
fn final_codec_rejects_complete_header_payload_and_length_before_mutation() {
    // Magic, region, common input, role, phase, lanes, noncanonical LAST word,
    // truncation and suffix. Each fixture is fresh because failures burn state.
    for fault in 0..9 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (_, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Gelu, 2), &[-1, 64])
            .unwrap();
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..7 {
            frames = next(&mut parties, &frames);
        }
        let mut bad = frames[1].as_bytes().to_vec();
        match fault {
            0 => bad[0] ^= 1,
            1 => bad[8] ^= 1,
            2 => bad[40] ^= 1,
            3 => bad[72] ^= 1,
            4 => bad[73] ^= 1,
            5 => bad[74] ^= 1,
            6 => *bad.last_mut().unwrap() |= 0x80,
            7 => {
                bad.pop();
            }
            _ => bad.push(0),
        }
        let before = parties[0].local.to_vec();
        assert_eq!(
            parties[0].validate_final(&bad),
            Err(if fault == 6 {
                Error::Domain
            } else {
                Error::Context
            })
        );
        assert_eq!(&*parties[0].local, &before);
        assert!(parties[0].advance(&bad).is_err());
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(Error::Consumed)
        );
        parties[1].cancel();
    }
}

#[test]
fn deferred_invocations_distinct_and_wrong_issuance_rejected() {
    for fault in [0, 1] {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (_, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Silu, 1), &[-65])
            .unwrap();
        for party in &parties {
            assert_ne!(party.lookup_binding, party.select_binding);
        }
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..6 {
            frames = next(&mut parties, &frames);
        }
        assert!(parties[0].x.is_some());
        if fault == 0 {
            parties[0].lookup_binding[0] ^= 1;
        } else {
            parties[0].select_binding[0] ^= 1;
        }
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(Error::Context)
        );
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        parties[1].cancel();
    }
}

#[test]
fn allocation_and_entropy_fault_sweeps_release_every_partial_owner() {
    for entropy in [false, true] {
        let mut reached_success = false;
        for fail in 0..512 {
            let budget = Budget::new(64 << 20).unwrap();
            {
                let mut d = dealer(&budget);
                let baseline = budget.used_bytes();
                if !entropy {
                    budget.fail_after.store(fail, Ordering::SeqCst);
                }
                let mut draws = 0;
                let result =
                    d.activation_from_values_with(spec(ActivationKind::Gelu, 1), &[-1], |bytes| {
                        if entropy && draws == fail {
                            return Err(Error::Randomness);
                        }
                        draws += 1;
                        bytes.fill(1);
                        Ok(())
                    });
                match result {
                    Ok((mask, mut parties, inputs)) => {
                        parties[0].cancel();
                        parties[1].cancel();
                        drop((parties, mask, inputs));
                        reached_success = true;
                    }
                    Err(error) => assert_eq!(
                        error,
                        if entropy {
                            Error::Randomness
                        } else {
                            Error::Allocation
                        }
                    ),
                }
                assert_eq!(budget.used_bytes(), baseline);
            }
            assert_eq!(budget.used_bytes(), 0);
            if reached_success {
                break;
            }
        }
        assert!(
            reached_success,
            "fault sweep did not reach complete issuance"
        );
    }
}

#[test]
fn retained_sends_outlive_cancelled_regions_without_early_refund() {
    let budget = Budget::new(64 << 20).unwrap();
    let mut d = dealer(&budget);
    let baseline = budget.used_bytes();
    let (mask, mut parties, inputs) = d
        .activation_from_values(spec(ActivationKind::Gelu, 1), &[64])
        .unwrap();
    let mut frames = begin(&mut parties, inputs);
    let mut retained = Vec::new();
    for _ in 0..7 {
        let new = next(&mut parties, &frames);
        retained.push(frames);
        frames = new;
    }
    parties[0].cancel();
    parties[1].cancel();
    drop((mask, parties));
    assert!(budget.used_bytes() > baseline);
    // Frames own their original native/wrapper splits; cancellation cannot
    // refund or corrupt any retained frame backing.
    assert_eq!(&frames[0].as_bytes()[..8], MAGIC);
    drop(frames);
    retained.reverse();
    drop(retained);
    assert_eq!(budget.used_bytes(), baseline);
}

#[test]
fn retained_cancelled_and_completed_parties_keep_exact_metadata_until_each_drop() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for complete in [false, true] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let (mask, mut parties, inputs) =
                d.activation_from_values(spec(kind, 1), &[-65]).unwrap();
            if complete {
                let mut frames = begin(&mut parties, inputs);
                for _ in 0..7 {
                    frames = next(&mut parties, &frames);
                }
                let [a, b] = &mut parties;
                let outputs = match (
                    a.advance(frames[1].as_bytes()).unwrap(),
                    b.advance(frames[0].as_bytes()).unwrap(),
                ) {
                    (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
                    _ => panic!("expected completed parties"),
                };
                assert_eq!(unmask(&mask, &outputs[0]), oracle(kind, &[-65]));
                drop((outputs, frames));
            } else {
                // Exercise cancellation after the original input owners move in.
                let frames = begin(&mut parties, inputs);
                for party in &mut parties {
                    party.cancel();
                }
                drop(frames);
            }
            drop(mask);
            let metadata = size_of::<ActivationParty<'static>>();
            assert_eq!(budget.used_bytes(), baseline + 2 * metadata);
            for party in &mut parties {
                assert_eq!(party.lease.as_ref().unwrap().bytes, metadata);
                party.cancel();
                party.cancel();
                assert_eq!(party.lease.as_ref().unwrap().bytes, metadata);
                assert_eq!(party.advance(&[]).err(), Some(Error::Consumed));
                assert_eq!(budget.used_bytes(), baseline + 2 * metadata);
            }
            let [a, b] = parties;
            drop(a);
            assert_eq!(budget.used_bytes(), baseline + metadata);
            drop(b);
            assert_eq!(budget.used_bytes(), baseline);
            drop(d);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn imported_and_clear_issuance_measure_exact_requested_random_bytes() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for lanes in [1, 256] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, lanes);
            let cost = activation_resources(s).unwrap();
            let values = vec![0; lanes];
            let mut bytes_requested = 0;
            let clear = d
                .activation_from_values_with(s, &values, |bytes| {
                    bytes_requested += bytes.len();
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            assert_eq!(bytes_requested, cost.clear_input_random_bytes);
            drop(clear);
            assert_eq!(budget.used_bytes(), baseline);
            let (input_mask, inputs) = d
                .prepare_root(
                    s.input,
                    BoundRange::new(-(1i128 << 37), 1i128 << 37).unwrap(),
                    s.operation,
                )
                .unwrap()
                .encode(&values)
                .unwrap();
            let mut bytes_requested = 0;
            let imported = d
                .activation_with(s, input_mask, |bytes| {
                    bytes_requested += bytes.len();
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            assert_eq!(bytes_requested, cost.random_bytes);
            assert_eq!(cost.clear_input_random_bytes - bytes_requested, 8 * lanes);
            drop((imported, inputs));
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn imported_missing_or_wider_range_rejected_before_draws_or_admission() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for range in [
            None,
            Some(Interval {
                lower: -(1i128 << 37) - 1,
                upper: 1,
            }),
            Some(Interval {
                lower: 0,
                upper: (1i128 << 37) + 1,
            }),
        ] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, 1);
            let (mut input, inputs) = d
                .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                .unwrap()
                .encode(&[0])
                .unwrap();
            input.binding.interval = range;
            budget.fail_after.store(0, Ordering::SeqCst);
            let mut draws = 0;
            assert_eq!(
                d.activation_with(s, input, |_| {
                    draws += 1;
                    Ok(())
                })
                .err(),
                Some(Error::Domain)
            );
            assert_eq!(draws, 0);
            assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
            budget.fail_after.store(-1, Ordering::SeqCst);
            drop(inputs);
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}

#[test]
fn actual_deferred_select_owner_substitutions_burn_region() {
    for fault in 0..4 {
        let budget = Budget::new(64 << 20).unwrap();
        let mut d = dealer(&budget);
        let (mask, mut parties, inputs) = d
            .activation_from_values(spec(ActivationKind::Silu, 1), &[-65])
            .unwrap();
        // Wider donor accepts all canonical ring50 inputs, so rejection reaches
        // the deferred capability width check rather than input validation.
        let bits = if fault == 1 { 64 } else { 50 };
        let lanes = if fault == 2 { 2 } else { 1 };
        let context = selection::Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [91; 32],
            predicate_wire: [92; 32],
            input_wire: [93; 32],
            output_wire: [94; 32],
            first_tensor_index: 19,
        };
        let (donor_mask, [left, right]) = selection::issue_deferred(
            selection::Spec { bits, lanes },
            context,
            &vec![0; lanes],
            &vec![0; lanes],
            &budget,
        )
        .unwrap();
        let donor = if fault == 0 {
            drop(left);
            right
        } else {
            drop(right);
            left
        };
        // Match the expected binding for role/width/lane faults to isolate those
        // checks. Cross-invocation fault keeps the original expected binding.
        if fault != 3 {
            parties[0].select_binding = donor.internal_binding();
        }
        parties[0].final_select = Some(donor);
        let mut frames = begin(&mut parties, inputs);
        for _ in 0..6 {
            frames = next(&mut parties, &frames);
        }
        assert_eq!(
            parties[0].advance(frames[1].as_bytes()).err(),
            Some(if fault == 2 {
                Error::Domain
            } else {
                Error::Context
            })
        );
        assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
        parties[1].cancel();
        drop((parties, frames, mask, donor_mask));
        drop(d);
        assert_eq!(budget.used_bytes(), 0);
    }
}

fn matrix_probe(budget: &Budget) -> Result<[matrix::party::Party<'_>; 2], Error> {
    matrix::party::issue_local(
        matrix::Spec {
            bits: 50,
            rows: 1,
            inner: 1,
            columns: 1,
            layout: matrix::MatrixLayout::InputOutput,
        },
        matrix::Context {
            plan: [1; 32],
            session: [2; 32],
            operation: [91; 32],
            input_wire: [92; 32],
            weight_wire: [93; 32],
            output_wire: [94; 32],
        },
        &[0],
        &[0],
        &[0],
        budget,
    )
}

#[test]
fn original_matrix_guard_survives_projection_through_final_select_or_cancel() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    for cancel_after in [Some(0), Some(1), Some(4), Some(6), None] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut d = dealer(&budget);
            let edge = |wire, fractional, private| {
                EdgeDescriptor::new(
                    EdgeIdentity::new(source(), [1; 32], [wire; 32], [wire; 32], 19),
                    Shape::new(&[1, 1]).unwrap(),
                    Numeric::new(50, fractional, Interpretation::Signed).unwrap(),
                    Representation::CommonAdditiveMasked,
                    if private {
                        Some(WeightVisibility::Private)
                    } else {
                        None
                    },
                )
                .unwrap()
            };
            let s = ActivationSpec::new(
                ActivationKind::Gelu,
                [7; 32],
                edge(3, 12, false),
                edge(4, 12, false),
                [8; 32],
            )
            .unwrap();
            let (xm, [x0, x1]) = d
                .prepare_root(
                    edge(21, 6, false),
                    BoundRange::new(0, 65).unwrap(),
                    [20; 32],
                )
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (wm, [w0, w1]) = d
                .prepare_root(edge(22, 6, true), BoundRange::new(0, 65).unwrap(), [20; 32])
                .unwrap()
                .encode(&[64])
                .unwrap();
            let (xm, [mut m0, mut m1]) = d
                .matrix(
                    [20; 32],
                    matrix::Spec {
                        bits: 50,
                        rows: 1,
                        inner: 1,
                        columns: 1,
                        layout: matrix::MatrixLayout::InputOutput,
                    },
                    xm,
                    wm,
                    s.input,
                    s.operation,
                    MatrixRange::RequireSignedNoWrap,
                )
                .unwrap();
            let ml = m0.begin(x0, w0, &mut || false).unwrap();
            let mr = m1.begin(x1, w1, &mut || false).unwrap();
            let x0 = m0.finish(mr.as_bytes()).unwrap();
            let x1 = m1.finish(ml.as_bytes()).unwrap();
            drop((m0, m1));
            let (mask, mut parties) = d.activation(s, xm).unwrap();
            let mut frames = begin(&mut parties, [x0, x1]);
            drop((ml, mr));
            let mut retained = Vec::new();
            for phase in 0..7 {
                // At phase six, lookup exchange is about to trigger final
                // Select against ORIGINAL ring50 X. Guard must still be held.
                assert_eq!(matrix_probe(&budget).err(), Some(Error::Resource));
                assert!(parties[0].x.is_some() && parties[1].x.is_some());
                if cancel_after == Some(phase) {
                    break;
                }
                let new = next(&mut parties, &frames);
                retained.push(frames);
                frames = new;
            }
            if cancel_after.is_none() {
                assert!(parties[0].phase == Phase::Final);
                assert!(parties[0].x.is_none() && parties[1].x.is_none());
                // Neither final output nor any retained activation frame
                // inherits/replaces the original matrix guard.
                drop(matrix_probe(&budget).unwrap());
                let [a, b] = &mut parties;
                let outputs = match (
                    a.advance(frames[1].as_bytes()).unwrap(),
                    b.advance(frames[0].as_bytes()).unwrap(),
                ) {
                    (ActivationStep::Output(a), ActivationStep::Output(b)) => [a, b],
                    _ => panic!("expected output"),
                };
                assert_eq!(
                    unmask(&mask, &outputs[0]),
                    oracle(ActivationKind::Gelu, &[4096])
                );
                drop(matrix_probe(&budget).unwrap());
            } else {
                parties[0].cancel();
                parties[1].cancel();
                drop(matrix_probe(&budget).unwrap());
            }
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn execution_allocation_faults_burn_and_refund_after_retained_frames_drop() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        let mut completed = false;
        for fail in 0..128 {
            let budget = Budget::new(64 << 20).unwrap();
            {
                let mut d = dealer(&budget);
                let baseline = budget.used_bytes();
                let (mask, mut parties, [x0, x1]) =
                    d.activation_from_values(spec(kind, 1), &[-65]).unwrap();
                budget.fail_after.store(fail, Ordering::SeqCst);
                let mut retained = Vec::new();
                let result = (|| -> Result<(), Error> {
                    let l = parties[0].begin(x0)?;
                    let r = parties[1].begin(x1)?;
                    let mut frames = [l, r];
                    for _ in 0..7 {
                        let a = parties[0].advance(frames[1].as_bytes())?;
                        let b = parties[1].advance(frames[0].as_bytes())?;
                        let new = match (a, b) {
                            (ActivationStep::Frame(a), ActivationStep::Frame(b)) => [a, b],
                            _ => return Err(Error::Context),
                        };
                        retained.push(frames);
                        frames = new;
                    }
                    let a = parties[0].advance(frames[1].as_bytes())?;
                    let b = parties[1].advance(frames[0].as_bytes())?;
                    assert!(matches!(a, ActivationStep::Output(_)));
                    assert!(matches!(b, ActivationStep::Output(_)));
                    Ok(())
                })();
                match result {
                    Ok(()) => completed = true,
                    Err(error) => assert_eq!(error, Error::Allocation),
                }
                parties[0].cancel();
                parties[1].cancel();
                drop((mask, parties, retained));
                assert_eq!(budget.used_bytes(), baseline);
            }
            assert_eq!(budget.used_bytes(), 0);
            if completed {
                break;
            }
        }
        assert!(completed);
    }
}

#[test]
fn silu_and_max_lane_imported_issuance_selected_faults_release_all_owners() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for lanes in [1, 256] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let s = spec(kind, lanes);
            let values = vec![0; lanes];
            let (input, inputs) = d
                .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                .unwrap()
                .encode(&values)
                .unwrap();
            budget.fail_after.store(1_000_000, Ordering::SeqCst);
            let mut draws = 0isize;
            let issued = d
                .activation_with(s, input, |bytes| {
                    draws += 1;
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap();
            let allocations = 1_000_000 - budget.fail_after.load(Ordering::SeqCst);
            budget.fail_after.store(-1, Ordering::SeqCst);
            drop((issued, inputs));
            assert_eq!(budget.used_bytes(), baseline);
            assert!(draws > 2 && allocations > 2);
            for entropy in [false, true] {
                let count = if entropy { draws } else { allocations };
                // Bound max-lane key issuance work: beginning, midpoint, final
                // failure, and exact successful boundary, not an exhaustive sweep.
                for fail in [0, 1, count / 2, count - 1, count] {
                    let (input, inputs) = d
                        .prepare_root(s.input, BoundRange::new(0, 1).unwrap(), s.operation)
                        .unwrap()
                        .encode(&values)
                        .unwrap();
                    if !entropy {
                        budget.fail_after.store(fail, Ordering::SeqCst);
                    }
                    let mut calls = 0;
                    let result = d.activation_with(s, input, |bytes| {
                        if entropy && calls == fail {
                            return Err(Error::Randomness);
                        }
                        calls += 1;
                        bytes.fill(1);
                        Ok(())
                    });
                    budget.fail_after.store(-1, Ordering::SeqCst);
                    if fail == count {
                        drop(result.unwrap());
                    } else {
                        assert_eq!(
                            result.err(),
                            Some(if entropy {
                                Error::Randomness
                            } else {
                                Error::Allocation
                            })
                        );
                    }
                    drop(inputs);
                    assert_eq!(budget.used_bytes(), baseline);
                }
            }
            drop(d);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn max_lane_selected_transition_allocation_faults_burn_and_preserve_retained_sends() {
    for kind in [ActivationKind::Gelu, ActivationKind::Silu] {
        for phase in [0, 3, 5, 6] {
            let budget = Budget::new(64 << 20).unwrap();
            let mut d = dealer(&budget);
            let baseline = budget.used_bytes();
            let (mask, mut parties, inputs) = d
                .activation_from_values(spec(kind, 256), &vec![-65; 256])
                .unwrap();
            let mut frames = begin(&mut parties, inputs);
            let mut retained = Vec::new();
            for _ in 0..phase {
                let new = next(&mut parties, &frames);
                retained.push(frames);
                frames = new;
            }
            budget.fail_after.store(0, Ordering::SeqCst);
            assert_eq!(
                parties[0].advance(frames[1].as_bytes()).err(),
                Some(Error::Allocation)
            );
            budget.fail_after.store(-1, Ordering::SeqCst);
            assert_eq!(parties[0].advance(&[]).err(), Some(Error::Consumed));
            parties[0].cancel();
            parties[1].cancel();
            drop((parties, mask));
            assert!(budget.used_bytes() > baseline);
            assert_eq!(
                frames[0].as_bytes().len(),
                if phase == 6 {
                    80 + 16 * 256
                } else {
                    76 + 8 * 256
                }
            );
            drop((frames, retained));
            assert_eq!(budget.used_bytes(), baseline);
        }
    }
}
