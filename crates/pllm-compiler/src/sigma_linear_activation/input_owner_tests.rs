use super::super::tests::{plan, request};
use super::*;
use pllm_core::edge_contract::EdgeIdentity;

#[path = "source_input_tests.rs"]
mod source_input;

const LIMIT: usize = 64 << 20;
const WORK: u128 = 1 << 28;

fn raw() -> Vec<u8> {
    (0..128).flat_map(|_| 0.25_f32.to_le_bytes()).collect()
}

fn fixture_request<'a>(
    p: &'a DecoderPlan,
    values: &'a [f64],
    encoded: &'a [i64],
) -> SigmaLinearActivationRequest<'a> {
    let mut r = request(p, values, encoded);
    r.selection.immutable_revision = "0123456789abcdef0123456789abcdef01234567".into();
    r.selection.content_sha256 = Sha256::digest(raw()).into();
    let source = SourceIdentity::new(
        r.selection.content_sha256,
        r.source.numeric_profile(),
        r.source.implementation(),
    );
    let replace = |e: EdgeDescriptor, producer| {
        let old = e.identity();
        EdgeDescriptor::new(
            EdgeIdentity::new(source, old.plan(), producer, old.wire(), 0),
            e.shape(),
            e.numeric(),
            e.representation(),
            e.weight_visibility(),
        )
        .unwrap()
    };
    r.source = source;
    r.input = replace(r.input, r.input.identity().producer());
    r.private_weights = replace(
        r.private_weights,
        sigma_weight_selection_identity(&r.selection),
    );
    r.complete_sum = replace(r.complete_sum, r.complete_sum.identity().producer());
    r.rescaled = replace(r.rescaled, r.rescaled.identity().producer());
    r.activation_output = replace(
        r.activation_output,
        r.activation_output.identity().producer(),
    );
    r.next_use = ConsumerUse::new(
        r.activation_output,
        r.next_use.consumer(),
        r.next_use.operand_position(),
        r.next_use.ordinal(),
    );
    r
}

fn declaration() -> PublisherDeclaration {
    let p = plan();
    let r = fixture_request(&p, &[], &[]);
    PublisherDeclaration {
        artifact: NamedDigest {
            id: "generated-complete-f32".into(),
            digest: Digest::from_sha256(r.selection.content_sha256),
        },
        selection: r.selection,
        expected_config: p.config_digest.clone(),
        expected_semantic_plan: p.digest(),
        selected_bytes: 512,
        dtype: SourceDtype::F32,
        endian: SourceEndian::Little,
        tensor_offset: 128,
    }
}

// Fixture origins only. Actual public/Rust parsing remains a separate admission
// gate; neither this test builder nor prepare claims to implement that gate.
fn owner_plan(root: &OwnedBudget) -> OwnedCharged<DecoderPlan> {
    let p = plan();
    let backing = p.checked_owned_backing_bytes(LIMIT).unwrap();
    root.charge_external(backing)
        .unwrap()
        .try_build(|_| Ok(p))
        .unwrap()
}
fn owner_publisher(
    root: &OwnedBudget,
    p: PublisherDeclaration,
) -> OwnedCharged<PublisherDeclaration> {
    root.charge_external(p.backing_bytes().unwrap())
        .unwrap()
        .try_build(|_| Ok(p))
        .unwrap()
}
fn source_work(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(bootstrap_bytes().unwrap()).unwrap()
}
fn inventory_work(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(inventory_bootstrap_bytes().unwrap())
        .unwrap()
}
fn build(root: &OwnedBudget) -> Result<OwnedSigmaInput, InputError> {
    prepare(
        owner_plan(root),
        owner_publisher(root, declaration()),
        source_work(root),
        inventory_work(root),
        WORK,
    )
}
fn reset(fail: Option<usize>, panic: Option<usize>) {
    COPY_FACTORIES.with(|c| c.set((0, fail, panic)));
}
fn copies() -> usize {
    COPY_FACTORIES.with(|c| c.get().0)
}

fn input_limits() -> PlanInputLimits {
    PlanInputLimits {
        parse: crate::plan_input::Limits {
            max_source_bytes: LIMIT,
            max_nodes: 1 << 20,
            max_depth: 32,
            max_string_bytes: 1 << 20,
            max_number_bytes: 128,
            max_number_work: WORK,
            max_work: 1 << 30,
            max_payload_bytes: LIMIT,
        },
        inventory_work: WORK,
        max_work: 1 << 30,
    }
}

fn raw_plan(root: &OwnedBudget) -> OwnedCharged<Vec<u8>> {
    // Test-only upstream byte creation, not another parser/import path. Count
    // without allocation, then serialize into ONE fixed exact backing under its
    // real original charge. Deliberate spare capacity exercises the real owner.
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len()).unwrap();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let p = plan();
    let control = size_of::<Count>()
        + size_of::<std::io::Cursor<&mut [u8]>>()
        + size_of::<serde_json::Serializer<&mut Count>>()
        + size_of::<serde_json::Serializer<std::io::Cursor<&mut [u8]>>>()
        + size_of::<Result<(), serde_json::Error>>()
        + size_of::<Box<[u8]>>()
        + size_of::<Vec<u8>>();
    let mut charge = root.charge_external(control).unwrap();
    let mut count = Count(0);
    serde_json::to_writer(&mut count, &p).unwrap();
    let capacity = count.0 + 73;
    charge.resize_payload(control + capacity).unwrap();
    charge
        .try_build(|_| {
            let mut raw = vec![0; capacity].into_boxed_slice().into_vec();
            raw.truncate(count.0);
            let mut cursor = std::io::Cursor::new(raw.as_mut_slice());
            serde_json::to_writer(&mut cursor, &p).unwrap();
            assert_eq!(cursor.position() as usize, count.0);
            Ok(raw)
        })
        .unwrap()
}

fn parse_work(root: &OwnedBudget) -> OwnedExternalCharge {
    root.charge_external(crate::plan_input::bootstrap_bytes().unwrap())
        .unwrap()
}

#[test]
fn raw_import_factory_preserves_original_byte_owner_and_uses_disjoint_work() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    let raw = raw_plan(&root);
    let ptr = raw.payload().as_ptr();
    let capacity = raw.payload().capacity();
    let raw_charge = raw.charged_bytes();
    let input = prepare_from_plan_bytes(
        raw,
        owner_publisher(&root, declaration()),
        parse_work(&root),
        source_work(&root),
        inventory_work(&root),
        input_limits(),
    )
    .unwrap();
    let retained = input.semantic.raw.as_ref().unwrap();
    assert_eq!(retained.payload().as_ptr(), ptr);
    assert_eq!(retained.payload().capacity(), capacity);
    assert_eq!(retained.charged_bytes(), raw_charge);
    assert!(input.is_budget(root.budget()));
    let work = input.preparation_work().unwrap();
    assert!(work.parse_prepaid() > 0);
    assert!(work.metadata_prepaid() > 0);
    assert_eq!(work.inventory_prepaid(), WORK);
    assert_eq!(
        work.total_prepaid(),
        work.parse_prepaid() + work.metadata_prepaid() + WORK
    );
    assert!(work.total_prepaid() <= input_limits().max_work);
    assert_eq!(input.semantic.plan(), &plan());
    drop(input);
    assert_eq!(root.budget().used_bytes(), initial);
}

#[test]
fn raw_import_global_exact_and_one_short_refuse_before_parse_or_copy_factories() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let input = prepare_from_plan_bytes(
        raw_plan(&root),
        owner_publisher(&root, declaration()),
        parse_work(&root),
        source_work(&root),
        inventory_work(&root),
        input_limits(),
    )
    .unwrap();
    let exact = input.preparation_work().unwrap().total_prepaid();
    drop(input);
    let initial = root.budget().used_bytes();
    for short in [false, true] {
        let raw = raw_plan(&root);
        let publisher = owner_publisher(&root, declaration());
        let mut limits = input_limits();
        limits.max_work = exact - u128::from(short);
        let before = crate::plan_input::allocation_factories_for_test();
        reset(None, None);
        let result = prepare_from_plan_bytes(
            raw,
            publisher,
            parse_work(&root),
            source_work(&root),
            inventory_work(&root),
            limits,
        );
        if short {
            assert!(matches!(
                result,
                Err(InputError::Import(crate::plan_input::Error::WorkLimit))
            ));
            assert_eq!(crate::plan_input::allocation_factories_for_test(), before);
            assert_eq!(copies(), 0);
        } else {
            assert_eq!(
                result
                    .as_ref()
                    .unwrap()
                    .preparation_work()
                    .unwrap()
                    .total_prepaid(),
                exact
            );
        }
        drop(result);
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn raw_import_wrong_counter_and_zero_global_allowance_never_enter_parse() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let foreign = OwnedBudget::new(LIMIT).unwrap();
    for wrong_counter in [false, true] {
        let before = crate::plan_input::allocation_factories_for_test();
        let mut limits = input_limits();
        if !wrong_counter {
            limits.max_work = 0;
        }
        reset(None, None);
        let result = prepare_from_plan_bytes(
            raw_plan(&root),
            owner_publisher(&root, declaration()),
            parse_work(if wrong_counter { &foreign } else { &root }),
            source_work(&root),
            inventory_work(&root),
            limits,
        );
        assert!(matches!(
            result,
            Err(InputError::BudgetIdentity | InputError::Work)
        ));
        assert_eq!(crate::plan_input::allocation_factories_for_test(), before);
        assert_eq!(copies(), 0);
    }
}

#[test]
fn raw_import_does_not_derive_publisher_authority_from_parsed_bytes() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    let mut publisher = declaration();
    publisher.expected_semantic_plan = Digest::from_sha256([0x44; 32]);
    reset(None, None);
    let result = prepare_from_plan_bytes(
        raw_plan(&root),
        owner_publisher(&root, publisher),
        parse_work(&root),
        source_work(&root),
        inventory_work(&root),
        input_limits(),
    );
    assert!(matches!(result, Err(InputError::PublisherIdentity)));
    assert_eq!(copies(), 0);
    assert_eq!(root.budget().used_bytes(), initial);
}

#[test]
fn actual_publisher_capacities_and_original_plan_backings_are_retained() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let mut p = declaration();
    p.selection.checkpoint.reserve(257);
    p.selection.inner_coordinates.reserve(129);
    p.artifact.id.reserve(113);
    let publisher = owner_publisher(&root, p);
    let q = quote(&publisher).unwrap();
    assert_eq!(
        q.publisher_owner_bytes,
        size_of::<OwnedCharged<PublisherDeclaration>>()
            + publisher.payload().backing_bytes().unwrap()
    );
    let plan = owner_plan(&root);
    let ptr = plan.payload().prefill.operations[0].id.as_ptr();
    let publisher_ptr = publisher.payload().selection.checkpoint.as_ptr();
    reset(None, None);
    let input = prepare(
        plan,
        publisher,
        source_work(&root),
        inventory_work(&root),
        WORK,
    )
    .unwrap();
    assert_eq!(input.semantic.plan().prefill.operations[0].id.as_ptr(), ptr);
    assert_eq!(
        input.publisher.payload().selection.checkpoint.as_ptr(),
        publisher_ptr
    );
    assert_ne!(input.pin.selection().checkpoint.as_ptr(), publisher_ptr);
    assert_ne!(
        input.selection.checkpoint.as_ptr(),
        input.pin.selection().checkpoint.as_ptr()
    );
    assert_eq!(
        input.selection.checkpoint.capacity(),
        input.selection.checkpoint.len()
    );
    assert_eq!(
        input.selection.inner_coordinates.capacity(),
        input.selection.inner_coordinates.len()
    );
    assert_eq!(input._charge.payload_bytes(), q.payload_bytes);
    assert!(copies() > 0);
}

#[test]
fn exact_total_reservation_and_one_short_precede_metadata_factories() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let input = build(&root).unwrap();
    let exact = root.budget().used_bytes();
    drop((input, root));
    for short in [false, true] {
        let root = OwnedBudget::new(exact - usize::from(short)).unwrap();
        let initial = root.budget().used_bytes();
        reset(None, None);
        let result = build(&root);
        if short {
            assert!(result.is_err());
            assert_eq!(copies(), 0);
        } else {
            assert!(result.is_ok());
            assert_eq!(root.budget().used_bytes(), exact);
        }
        drop(result);
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn source_and_inventory_bootstrap_shortfalls_are_factory_free() {
    for inventory_short in [false, true] {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let initial = root.budget().used_bytes();
        let source = root
            .charge_external(bootstrap_bytes().unwrap() - usize::from(!inventory_short))
            .unwrap();
        let inventory = root
            .charge_external(inventory_bootstrap_bytes().unwrap() - usize::from(inventory_short))
            .unwrap();
        reset(None, None);
        assert!(matches!(
            prepare(
                owner_plan(&root),
                owner_publisher(&root, declaration()),
                source,
                inventory,
                WORK
            ),
            Err(InputError::Resource)
        ));
        assert_eq!(copies(), 0);
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn foreign_counters_and_undercharged_publisher_never_copy_metadata() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let other = OwnedBudget::new(LIMIT).unwrap();
    reset(None, None);
    assert!(matches!(
        prepare(
            owner_plan(&root),
            owner_publisher(&other, declaration()),
            source_work(&root),
            inventory_work(&root),
            WORK
        ),
        Err(InputError::BudgetIdentity)
    ));
    let p = declaration();
    let publisher = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(p))
        .unwrap();
    assert!(matches!(
        prepare(
            owner_plan(&root),
            publisher,
            source_work(&root),
            inventory_work(&root),
            WORK
        ),
        Err(InputError::PublisherOwner)
    ));
    assert_eq!(copies(), 0);
}

#[test]
fn publisher_tuple_must_match_genuine_inventory_before_copy() {
    for config in [false, true] {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let initial = root.budget().used_bytes();
        let mut p = declaration();
        if config {
            p.expected_config = Digest::from_sha256([91; 32]);
        } else {
            p.expected_semantic_plan = Digest::from_sha256([92; 32]);
        }
        reset(None, None);
        assert!(matches!(
            prepare(
                owner_plan(&root),
                owner_publisher(&root, p),
                source_work(&root),
                inventory_work(&root),
                WORK
            ),
            Err(InputError::PublisherIdentity)
        ));
        assert_eq!(copies(), 0);
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn every_new_copy_failure_and_panic_retires_payload_before_guards() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    reset(None, None);
    let input = build(&root).unwrap();
    let count = copies();
    drop(input);
    for ordinal in 0..count {
        reset(Some(ordinal), None);
        assert!(build(&root).is_err());
        assert_eq!(root.budget().used_bytes(), initial);
        reset(None, Some(ordinal));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(&root))).is_err());
        assert_eq!(root.budget().used_bytes(), initial);
    }
    reset(None, None);
}

#[test]
fn late_shared_pin_validation_failure_cleans_new_and_original_owners() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    let mut p = declaration();
    p.selection.immutable_revision = "unversioned".into();
    reset(None, None);
    assert!(matches!(
        prepare(
            owner_plan(&root),
            owner_publisher(&root, p),
            source_work(&root),
            inventory_work(&root),
            WORK
        ),
        Err(InputError::Source(_))
    ));
    assert!(copies() > 0);
    assert_eq!(root.budget().used_bytes(), initial);
}

#[test]
fn bundle_survives_root_drop_without_counter_or_capacity_replacement() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let input = build(&root).unwrap();
    let budget = input.semantic.budget() as *const _;
    let before = input.semantic.budget().used_bytes();
    drop(root);
    assert_eq!(input.semantic.budget() as *const _, budget);
    assert!(input.semantic.budget().used_bytes() < before);
    assert!(input._charge.is_for(input.semantic.budget()));
    assert!(std::ptr::eq(
        input.publisher.budget(),
        input.semantic.budget()
    ));
    assert_eq!(
        input.semantic.inventory().semantic_plan_identity(),
        input.publisher.payload().expected_semantic_plan.as_sha256()
    );
}

#[test]
fn oversized_original_working_guard_is_never_shrunk_or_reacquired() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let publisher = owner_publisher(&root, declaration());
    let required = quote(&publisher).unwrap().payload_bytes;
    let original = root.charge_external(required + 113).unwrap();
    let input = prepare(
        owner_plan(&root),
        publisher,
        original,
        inventory_work(&root),
        WORK,
    )
    .unwrap();
    assert_eq!(input._charge.payload_bytes(), required + 113);
    assert!(input._charge.is_for(root.budget()));
}

#[test]
fn genuine_plan_guard_is_recensused_before_metadata_copy() {
    let root = OwnedBudget::new(LIMIT).unwrap();
    let initial = root.budget().used_bytes();
    let unpriced = root
        .charge_external(0)
        .unwrap()
        .try_build(|_| Ok(plan()))
        .unwrap();
    reset(None, None);
    assert!(matches!(
        prepare(
            unpriced,
            owner_publisher(&root, declaration()),
            source_work(&root),
            inventory_work(&root),
            WORK
        ),
        Err(InputError::Inventory(NativeError::Resource))
    ));
    assert_eq!(copies(), 0);
    assert_eq!(root.budget().used_bytes(), initial);
}

fn operation(r: SigmaLinearActivationRequest<'_>) -> Operation<'_> {
    Operation {
        mode: r.mode,
        linear_operation: r.linear_operation,
        activation_operation: r.activation_operation,
        next_operation: r.next_operation,
        source: r.source,
        input: r.input,
        private_weights: r.private_weights,
        complete_sum: r.complete_sum,
        rescaled: r.rescaled,
        activation_output: r.activation_output,
        rescale_operation: r.rescale_operation,
        next_use: r.next_use,
        input_domain: r.input_domain,
        source_coefficients: r.source_coefficients,
        encoded_private_coefficients: r.encoded_private_coefficients,
    }
}

#[test]
fn real_owned_facade_source_check_and_d27_split_remain_authoritative() {
    for corrupt in [false, true] {
        let root = OwnedBudget::new(LIMIT).unwrap();
        let reference = plan();
        let values = [0.25; 128];
        let encoded = [1024; 128];
        let r = fixture_request(&reference, &values, &encoded);
        let selection = r.selection.clone();
        let mut dealer =
            typed::OwnedLocalDealer::new(r.source, r.input.identity().plan(), [72; 32], &root)
                .unwrap();
        let input = build(&root).unwrap();
        let mut bytes = raw();
        if corrupt {
            bytes[0] ^= 1;
        }
        let offsets: Vec<_> = (0..128).map(|i| 128 + 4 * i as u64).collect();
        let packet = SigmaSourceBytes {
            artifact_name: "generated-complete-f32",
            selection: &selection,
            dtype: SourceDtype::F32,
            endian: SourceEndian::Little,
            tensor_offset: 128,
            scalar_artifact_offsets: &offsets,
            bytes: &bytes,
        };
        let result = input.execute(packet, operation(r), &root, &mut dealer, &[-1; 8]);
        if corrupt {
            assert!(matches!(
                result,
                Err(InputError::Execution(SigmaExecutionError::Source(_)))
            ));
        } else {
            let (mask, run) = result.unwrap();
            let roles = run.into_roles(&root).unwrap();
            drop((mask, roles));
        }
    }
}
