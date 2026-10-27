use super::*;

// Proposed physical embedding only. This test interpreter uses exact integer
// numerators, NOT ciphertexts or a CKKS numerical simulation. Expected values
// come from separately indexed token-major X, input-major W and output bias;
// expected contraction never calls candidate packing, decoding or staging.
fn physical_layouts(i: usize, o: usize, tokens: usize, capacity: usize) -> (Layout, Layout) {
    (
        Layout::new(2, 2, 2, tokens, i / 2, capacity, Diagonal::Lower, LIMITS).unwrap(),
        Layout::new(2, 2, 2, tokens, o / 2, capacity, Diagonal::Lower, LIMITS).unwrap(),
    )
}

fn physical_oracle_case(iw: usize, ow: usize, tokens: usize, x: &[i64], w: &[i64], bias: &[i64]) {
    let (input, out) = physical_layouts(iw, ow, tokens, 1);
    let plan = block_pc_physical_plan(input, out, 4096, LIMITS).unwrap();
    physical_oracle_with_plan(plan, iw, ow, tokens, x, w, bias);
}

fn physical_oracle_with_plan(
    plan: BlockPcPhysicalPlan, iw: usize, ow: usize, tokens: usize,
    x: &[i64], w: &[i64], bias: &[i64],
) {
    let mut expected = vec![0_i64; 2 * ow];
    for token in 0..tokens {
        for feature in 0..ow {
            let mut numerator = 16 * bias[feature];
            for inner in 0..iw {
                numerator += x[token * iw + inner] * w[inner * ow + feature];
            }
            expected[token * ow + feature] = numerator;
        }
    }
    let mut dense_x: Vec<f64> = x.iter().map(|v| *v as f64 / 16.0).collect();
    dense_x[tokens * iw..].fill(f64::NAN);
    let weights: Vec<f64> = w.iter().map(|v| *v as f64 / 16.0).collect();
    let biases: Vec<f64> = bias.iter().map(|v| *v as f64 / 16.0).collect();
    let mut packed = vec![SENTINEL; 8192 + 3];
    plan.materialize_input(&dense_x, &mut packed, LIMITS).unwrap();
    assert!(packed[8192..].iter().all(|v| *v == SENTINEL));
    for q in 0..2 {
        for s in 0..4096 {
            let t = s / 4;
            let k = s % 4;
            let want = if t < tokens && k < iw / 2 {
                x[t * iw + 2 * k + (t + q) % 2] as f64 / 16.0
            } else { 0.0 };
            assert_eq!(packed[q * 4096 + s], want);
        }
    }
    for t in 0..tokens {
        for f in 0..iw {
            let (q, s) = plan.input_address(t, f).unwrap();
            assert_eq!(packed[q * 4096 + s], x[t * iw + f] as f64 / 16.0);
        }
    }
    let mut partial = vec![0_i64; 4096];
    let mut masked = partial.clone();
    let mut remainder = partial.clone();
    let mut aligned = partial.clone();
    let mut result = vec![0_i64; 8192];
    let mut plain = vec![SENTINEL; 4099];
    for stage in plan.stages() {
        let q = stage.output_frame;
        let a = stage.displacement;
        match stage.op {
            PhysicalOp::WeightProduct { input_frame: j } => {
                if j == 0 { partial.fill(0); }
                plan.materialize_weight(q, a, j, &weights, &mut plain, LIMITS).unwrap();
                assert!(plain[4096..].iter().all(|v| *v == SENTINEL));
                for s in 0..4096 {
                    partial[s] += (packed[j * 4096 + s] * 16.0) as i64
                        * (plain[s] * 16.0) as i64;
                }
            }
            PhysicalOp::ProductSumBoundary => {} // No numerical rescale claim.
            PhysicalOp::Mask => {
                plan.materialize_mask(a, &mut plain, LIMITS).unwrap();
                for s in 0..4096 {
                    masked[s] = partial[s] * plain[s] as i64;
                }
                if a == 0 { aligned.copy_from_slice(&masked); }
            }
            PhysicalOp::SupportMask { branch } => {
                plan.materialize_branch_mask(a, branch, &mut plain, LIMITS).unwrap();
                // Each branch reads the original P. In particular, Remainder
                // must NOT read masked (which would make every lane zero).
                let target = if branch == PhysicalMaskBranch::Remainder { &mut remainder } else { &mut masked };
                for s in 0..4096 { target[s] = partial[s] * plain[s] as i64; }
                if branch == PhysicalMaskBranch::Support { aligned.copy_from_slice(&masked); }
            }
            PhysicalOp::MaskRescale { .. } => {} // Exact ideal identity only.
            PhysicalOp::SubtractMasked => {
                for s in 0..4096 { remainder[s] = partial[s] - masked[s]; }
            }
            PhysicalOp::RotateMasked { shift } => {
                masked = (0..4096).map(|s| masked[(s as i32 + shift).rem_euclid(4096) as usize]).collect();
            }
            PhysicalOp::RotateRemainder { shift } => {
                remainder = (0..4096).map(|s| remainder[(s as i32 + shift).rem_euclid(4096) as usize]).collect();
            }
            PhysicalOp::AddAligned => {
                for s in 0..4096 { aligned[s] = masked[s] + remainder[s]; }
            }
            PhysicalOp::Accumulate => {
                for s in 0..4096 { result[q * 4096 + s] += aligned[s]; }
            }
        }
    }
    let mut bias_slots = vec![SENTINEL; 8195];
    plan.materialize_bias(&biases, &mut bias_slots, LIMITS).unwrap();
    assert!(bias_slots[8192..].iter().all(|v| *v == SENTINEL));
    for q in 0..2 {
        // Check EVERY physical lane, independently of candidate output decode.
        for s in 0..4096 {
            let t = s / 4;
            let block = s % 4;
            let feature = 2 * block + (t + q) % 2;
            let want = if t < tokens && block < ow / 2 { expected[t * ow + feature] } else { 0 };
            let actual = result[q * 4096 + s] + (bias_slots[q * 4096 + s] * 256.0) as i64;
            assert_eq!(actual, want, "q={q}, slot={s}, {iw}->{ow}, tokens={tokens}");
            let decoded = plan.output_coordinate(q, s).unwrap();
            if t < tokens && block < ow / 2 {
                assert_eq!(decoded, Some((t, feature)));
                let (token, output) = decoded.unwrap();
                assert_eq!(actual, expected[token * ow + output]);
            } else { assert_eq!(decoded, None); }
        }
    }
}

#[test]
fn physical_pc_exact_integer_asymmetric_zero_and_basis_oracles() {
    for (iw, ow) in [(4, 8), (8, 4)] {
        for tokens in [0, 1, 2] {
            let x = rect_integers(2 * iw, 7, 33);
            let w = rect_integers(iw * ow, 11, 33);
            let bias = rect_integers(ow, 13, 33);
            physical_oracle_case(iw, ow, tokens, &x, &w, &bias);
            physical_oracle_case(iw, ow, tokens, &vec![0; 2 * iw], &vec![0; iw * ow], &bias);
            // Each input coordinate and both token rows; distinctive signed
            // weights/bias catch transpose, alignment and post-alignment bias bugs.
            for t in 0..2 {
                for f in 0..iw {
                    let mut basis = vec![0; 2 * iw];
                    basis[t * iw + f] = if f % 2 == 0 { 16 } else { -16 };
                    physical_oracle_case(iw, ow, tokens, &basis, &w, &bias);
                }
            }
        }
    }
}

#[test]
fn physical_pc_order_masks_signed_wrap_counts_and_unqualified_contract() {
    let (input, out) = physical_layouts(4, 8, 2, 1);
    let plan = block_pc_physical_plan(input, out, 4096, LIMITS).unwrap();
    assert_eq!(plan.scale_status(), PhysicalScaleStatus::Unqualified);
    assert_eq!(BlockPcPhysicalPlan::LOGICAL_REFERENCE_WORK, 224);
    assert_eq!(BlockPcPhysicalPlan::CONSTRUCTION_WORK, 64);
    assert_eq!(plan.input_materialization_work(), Ok(8200));
    assert_eq!(plan.weight_materialization_work(), Ok(4128));
    assert_eq!(plan.bias_materialization_work(), Ok(8200));
    assert_eq!(plan.schedule(), Schedule {
        plaintext_products: 24, output_rotations: 12,
        rescale_after_product_sums: 8, ..Schedule::default()
    });
    assert_eq!(BlockPcPhysicalPlan::SIGNED_KEYS, [-3, -2, -1, 1, 2, 3]);
    assert_eq!(BlockPcPhysicalPlan::KEY_RESIDUES, [4093, 4094, 4095, 1, 2, 3]);
    let mut expected = Vec::new();
    for q in 0..2 {
        for a in 0..4 {
            let mut ops = vec![PhysicalOp::WeightProduct { input_frame: 0 },
                PhysicalOp::WeightProduct { input_frame: 1 }, PhysicalOp::ProductSumBoundary,
                PhysicalOp::Mask];
            if a > 0 {
                ops.extend([PhysicalOp::SubtractMasked,
                    PhysicalOp::RotateMasked { shift: (4 - a) as i32 },
                    PhysicalOp::RotateRemainder { shift: -(a as i32) }, PhysicalOp::AddAligned]);
            }
            ops.push(PhysicalOp::Accumulate);
            expected.extend(ops.into_iter().map(|op| PhysicalStage { output_frame: q, displacement: a, op }));
            let mut mask = vec![SENTINEL; 4097];
            plan.materialize_mask(a, &mut mask, LIMITS).unwrap();
            for s in 0..4096 {
                assert_eq!(mask[s], if s < 8 && (a == 0 || s % 4 >= 4 - a) { 1.0 } else { 0.0 });
            }
            assert_eq!(mask[4096], SENTINEL);
            if a > 0 {
                // One-hot probes include signed cyclic wrap and physical tail;
                // rotation convention is source=destination+signed shift.
                for source in [0_usize, 3, 4, 7, 8, 4095] {
                    for shift in [-(a as i32), (4 - a) as i32] {
                        let destination = (source as i32 - shift).rem_euclid(4096) as usize;
                        assert_eq!((destination as i32 + shift).rem_euclid(4096) as usize, source);
                    }
                }
            }
        }
    }
    assert_eq!(plan.stages(), expected.as_slice());
}

#[test]
fn physical_pc_geometry_caps_invalid_addresses_and_overflow() {
    for (iw, ow) in [(4, 8), (8, 4)] {
        let (input, out) = physical_layouts(iw, ow, 2, 2);
        // Logical capacity two stays valid and retains its full virtual count.
        assert_eq!(rectangular_block_pc_schedule(input, out).unwrap().plaintext_products, 12);
        assert!(matches!(block_pc_physical_plan(input, out, 4096, LIMITS), Err(Error::Geometry)));
        let (input, out) = physical_layouts(iw, ow, 2, 1);
        for slots in [0, 8, 2048, 8192, usize::MAX] {
            assert!(matches!(block_pc_physical_plan(input, out, slots, LIMITS), Err(Error::Geometry)));
        }
        let plan = block_pc_physical_plan(input, out, 4096, LIMITS).unwrap();
        assert_eq!(plan.input_address(2, 0), Err(Error::Geometry));
        assert_eq!(plan.input_address(0, iw), Err(Error::Geometry));
        assert_eq!(plan.output_coordinate(2, 0), Err(Error::Geometry));
        assert_eq!(plan.output_coordinate(0, 4096), Err(Error::Geometry));
        assert_eq!(plan.weight_coordinate(0, 4, 0, 0), Err(Error::Geometry));
        assert_eq!(plan.weight_coordinate(0, 0, 2, 0), Err(Error::Geometry));
        assert_eq!(plan.mask_value(usize::MAX, 0), Err(Error::Geometry));
        assert!(matches!(block_pc_physical_plan(input, out, 4096,
            Limits { max_work: 63, ..LIMITS }), Err(Error::WorkLimit)));
        assert!(matches!(block_pc_physical_plan(input, out, 4096,
            Limits { max_buffer_elements: 8191, ..LIMITS }), Err(Error::Buffer)));
    }
    let huge = Limits { max_buffer_elements: usize::MAX, max_work: usize::MAX };
    assert_eq!(Layout::new(2, 2, 2, 2, usize::MAX, 1, Diagonal::Lower, huge), Err(Error::Overflow));
    assert_eq!(mul(usize::MAX, 2), Err(Error::Overflow));
    assert_eq!(add(usize::MAX, 1), Err(Error::Overflow));
    let (input, _) = physical_layouts(4, 8, 2, 1);
    assert!(matches!(block_pc_physical_plan(input, input, 4096, LIMITS), Err(Error::Geometry)));
}

#[test]
fn physical_pc_logical_capacity_one_and_two_keep_independent_dense_identity() {
    for (iw, ow) in [(4, 8), (8, 4)] {
        for capacity in [1, 2] {
            for tokens in [0, 1, 2] {
                let (input, out) = physical_layouts(iw, ow, tokens, capacity);
                let x = rect_integers(2 * iw, 7, 33);
                let w = rect_integers(iw * ow, 11, 33);
                let mut transposed_x = vec![0; 2 * iw];
                let mut transposed_w = vec![0.0; iw * ow];
                for f in 0..iw {
                    for t in 0..2 { transposed_x[f * 2 + t] = x[t * iw + f]; }
                    for o in 0..ow { transposed_w[o * iw + f] = w[f * ow + o] as f64; }
                }
                let mut output = vec![SENTINEL; out.elements() + 3];
                rectangular_block_pc_mm(input, &transposed_w,
                    &rect_slots(2, iw / 2, capacity, tokens, &transposed_x),
                    out, &mut output, LIMITS).unwrap();
                let mut expected = vec![0_i64; ow * 2];
                for t in 0..tokens {
                    for o in 0..ow {
                        for f in 0..iw { expected[o * 2 + t] += x[t * iw + f] * w[f * ow + o]; }
                    }
                }
                rect_assert_slots(2, ow / 2, capacity, &output, &expected);
                if capacity == 2 {
                    assert!(matches!(block_pc_physical_plan(input, out, 4096, LIMITS), Err(Error::Geometry)));
                }
            }
        }
    }
}

#[test]
fn physical_pc_materialization_prewrite_atomicity_and_work_one_short() {
    let (input, out) = physical_layouts(4, 8, 2, 1);
    let plan = block_pc_physical_plan(input, out, 4096, LIMITS).unwrap();
    let mut destination = vec![SENTINEL; 8195];
    let unchanged = destination.clone();
    let x = vec![0.0; 8];
    let w = vec![0.0; 32];
    let b = vec![0.0; 8];
    for short in [0, 8191] {
        assert_eq!(plan.materialize_input(&x, &mut destination[..short], LIMITS), Err(Error::Buffer));
        assert_eq!(plan.materialize_bias(&b, &mut destination[..short], LIMITS), Err(Error::Buffer));
    }
    assert_eq!(plan.materialize_input(&x[..7], &mut destination, LIMITS), Err(Error::Buffer));
    assert_eq!(plan.materialize_weight(0, 0, 0, &w[..31], &mut destination, LIMITS), Err(Error::Buffer));
    assert_eq!(plan.materialize_bias(&b[..7], &mut destination, LIMITS), Err(Error::Buffer));
    assert_eq!(plan.materialize_weight(2, 0, 0, &w, &mut destination, LIMITS), Err(Error::Geometry));
    assert_eq!(plan.materialize_mask(4, &mut destination, LIMITS), Err(Error::Geometry));
    assert_eq!(plan.materialize_mask(0, &mut destination[..4095], LIMITS), Err(Error::Buffer));
    assert_eq!(plan.materialize_weight(0, 0, 0, &w, &mut destination[..4095], LIMITS), Err(Error::Buffer));
    assert_eq!(plan.materialize_input(&x, &mut destination,
        Limits { max_work: plan.input_materialization_work().unwrap() - 1, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_weight(0, 0, 0, &w, &mut destination,
        Limits { max_work: plan.weight_materialization_work().unwrap() - 1, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_bias(&b, &mut destination,
        Limits { max_work: plan.bias_materialization_work().unwrap() - 1, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_mask(0, &mut destination,
        Limits { max_work: 4095, ..LIMITS }), Err(Error::WorkLimit));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad_x = x.clone(); bad_x[7] = bad;
        let mut bad_w = w.clone(); bad_w[31] = bad;
        let mut bad_b = b.clone(); bad_b[7] = bad;
        assert_eq!(plan.materialize_input(&bad_x, &mut destination, LIMITS), Err(Error::NonFinite));
        assert_eq!(plan.materialize_weight(0, 0, 0, &bad_w, &mut destination, LIMITS), Err(Error::NonFinite));
        assert_eq!(plan.materialize_bias(&bad_b, &mut destination, LIMITS), Err(Error::NonFinite));
    }
    assert_eq!(plan.materialize_mask(0, &mut destination,
        Limits { max_buffer_elements: 8194, ..LIMITS }), Err(Error::Buffer));
    assert_eq!(destination, unchanged);
    // Exact budgets succeed; no hidden use of the logical work=224 quote.
    plan.materialize_input(&x, &mut destination,
        Limits { max_work: plan.input_materialization_work().unwrap(), ..LIMITS }).unwrap();
    plan.materialize_weight(0, 0, 0, &w, &mut destination,
        Limits { max_work: plan.weight_materialization_work().unwrap(), ..LIMITS }).unwrap();
    plan.materialize_bias(&b, &mut destination,
        Limits { max_work: plan.bias_materialization_work().unwrap(), ..LIMITS }).unwrap();
    plan.materialize_mask(0, &mut destination, Limits { max_work: 4096, ..LIMITS }).unwrap();
    assert!(destination[8192..].iter().all(|v| *v == SENTINEL));
}

#[test]
fn physical_pc_scaled_support_partition_all_slots() {
    for (iw, ow) in [(4, 8), (8, 4)] {
        let (input, out) = physical_layouts(iw, ow, 2, 1);
        let plan = block_pc_physical_plan_scaled_support(input, out, 4096, LIMITS).unwrap();
        let mut support = vec![SENTINEL; 4099];
        plan.materialize_branch_mask(0, PhysicalMaskBranch::Support, &mut support, LIMITS).unwrap();
        for a in 1..4 {
            let mut wrap = vec![SENTINEL; 4099];
            let mut rest = vec![SENTINEL; 4099];
            plan.materialize_branch_mask(a, PhysicalMaskBranch::Wrap, &mut wrap, LIMITS).unwrap();
            plan.materialize_branch_mask(a, PhysicalMaskBranch::Remainder, &mut rest, LIMITS).unwrap();
            for s in 0..4096 {
                let f = s < 8;
                let m = f && s % 4 >= 4 - a;
                assert_eq!(support[s], if f { 1.0 } else { 0.0 });
                assert_eq!(wrap[s], if m { 1.0 } else { 0.0 });
                assert_eq!(rest[s], if f && !m { 1.0 } else { 0.0 });
                assert_eq!(wrap[s] + rest[s], support[s]);
                assert_eq!(wrap[s] * rest[s], 0.0);
                assert_eq!(plan.branch_mask_value(a, PhysicalMaskBranch::Wrap, s), Ok(m));
                assert_eq!(plan.branch_mask_value(a, PhysicalMaskBranch::Remainder, s), Ok(f && !m));
                // Global 1-M would incorrectly leave 4088 unit tail lanes.
                if s >= 8 { assert_eq!((wrap[s], rest[s]), (0.0, 0.0)); }
            }
            for values in [&support, &wrap, &rest] {
                assert!(values[4096..].iter().all(|v| *v == SENTINEL));
            }
        }
    }
}

#[test]
fn scaled_support_integer_oracle() {
    for (iw, ow) in [(4, 8), (8, 4)] {
        for tokens in [0, 1, 2] {
            let (input, out) = physical_layouts(iw, ow, tokens, 1);
            let plan = block_pc_physical_plan_scaled_support(input, out, 4096, LIMITS).unwrap();
            let x = rect_integers(2 * iw, 7, 33);
            let w = rect_integers(iw * ow, 11, 33);
            let bias = rect_integers(ow, 13, 33);
            // Shared exact test interpreter, separately indexed dense oracle:
            // sum(X*W)+16*b over 256. No CKKS rescale/rounding simulation.
            physical_oracle_with_plan(plan, iw, ow, tokens, &x, &w, &bias);
            physical_oracle_with_plan(plan, iw, ow, tokens, &vec![0; 2 * iw], &vec![0; iw * ow], &bias);
            physical_oracle_with_plan(plan, iw, ow, tokens, &vec![0; 2 * iw], &vec![0; iw * ow], &vec![0; ow]);
            for t in 0..2 {
                for f in 0..iw {
                    let mut basis = vec![0; 2 * iw];
                    basis[t * iw + f] = if f % 2 == 0 { 16 } else { -16 };
                    physical_oracle_with_plan(plan, iw, ow, tokens, &basis, &w, &bias);
                }
            }
        }
    }
}

#[test]
fn scaled_support_order_counts_and_scale_states() {
    let (input, out) = physical_layouts(4, 8, 2, 1);
    let legacy = block_pc_physical_plan(input, out, 4096, LIMITS).unwrap();
    let plan = block_pc_physical_plan_scaled_support(input, out, 4096, LIMITS).unwrap();
    assert_eq!(legacy.stages().len(), 64);
    assert_eq!(legacy.construction_work(), 64);
    assert_eq!(legacy.schedule().plaintext_products, 24);
    assert_eq!(legacy.schedule().rescale_after_product_sums, 8);
    assert_eq!(legacy.scale_contract(), PhysicalScaleContract::LegacyUnresolved);
    assert_eq!(legacy.scale_contract().state(PhysicalScalePoint::Partial), None);
    assert_eq!(legacy.scale_status(), PhysicalScaleStatus::Unqualified);
    assert_eq!(plan.scale_status(), PhysicalScaleStatus::Unqualified);
    assert_eq!(plan.stages().len(), 78);
    assert_eq!(plan.construction_work(), 78);
    assert_eq!(BlockPcPhysicalPlan::LOGICAL_REFERENCE_WORK, 224);
    assert_eq!(plan.operation_counts(), PhysicalOperationCounts {
        weight_products: 16, mask_products: 14, product_sum_rescales: 8,
        mask_rescales: 14, total_rescales: 22, input_rotations: 0,
        output_rotations: 12, multiplicative_depth: Some(2),
    });
    assert_eq!(plan.schedule().plaintext_products, 30);
    assert_eq!(plan.schedule().rescale_after_product_sums, 8);
    assert_eq!(plan.scheduled_mask_materialization_work(), Ok(57344));
    assert_eq!(legacy.scheduled_mask_materialization_work(), Ok(32768));
    assert_eq!(BlockPcPhysicalPlan::SIGNED_KEYS, [-3, -2, -1, 1, 2, 3]);
    assert_eq!(BlockPcPhysicalPlan::KEY_RESIDUES, [4093, 4094, 4095, 1, 2, 3]);
    assert!(std::mem::size_of::<PhysicalStages>() <=
        std::mem::size_of::<[PhysicalStage; 78]>() + std::mem::align_of::<PhysicalStages>());
    let contract = plan.scale_contract();
    assert_eq!(contract, PhysicalScaleContract::ScaledSupport);
    let input_scale = contract.state(PhysicalScalePoint::Input).unwrap();
    assert_eq!(input_scale.expression(), PhysicalScaleExpression::Delta);
    assert_eq!(input_scale.dropped_primes(), PhysicalDroppedPrimes::None);
    assert_eq!(contract.state(PhysicalScalePoint::WeightPlaintext), Some(input_scale));
    assert_eq!(contract.state(PhysicalScalePoint::MaskPlaintext), Some(input_scale));
    let weight = contract.state(PhysicalScalePoint::WeightProduct).unwrap();
    assert_eq!(weight.expression(), PhysicalScaleExpression::DeltaSquared);
    assert_eq!(weight.dropped_primes(), PhysicalDroppedPrimes::None);
    let partial = contract.state(PhysicalScalePoint::Partial).unwrap();
    assert_eq!(partial.expression(), PhysicalScaleExpression::Alpha);
    assert_eq!(partial.dropped_primes(), PhysicalDroppedPrimes::P1);
    let beta = contract.state(PhysicalScalePoint::Aligned).unwrap();
    assert_eq!(beta.expression(), PhysicalScaleExpression::Beta);
    assert_eq!(beta.dropped_primes(), PhysicalDroppedPrimes::P1ThenP2);
    assert_eq!(contract.state(PhysicalScalePoint::Accumulator), Some(beta));
    assert_eq!(contract.state(PhysicalScalePoint::Bias), Some(beta));
    for branch in [PhysicalMaskBranch::Support, PhysicalMaskBranch::Wrap, PhysicalMaskBranch::Remainder] {
        let product = contract.state(PhysicalScalePoint::MaskProduct { branch }).unwrap();
        assert_eq!(product.expression(), PhysicalScaleExpression::AlphaDelta);
        assert_eq!(product.dropped_primes(), PhysicalDroppedPrimes::P1);
        assert_eq!(contract.state(PhysicalScalePoint::MaskRescaled { branch }), Some(beta));
        assert_ne!(product, partial);
        assert_ne!(product, beta);
    }
    // Internal negative equality witness; private fields have no public
    // constructor/retag API. Equal expression alone is insufficient.
    let different_sequence = PhysicalSymbolicScale {
        expression: PhysicalScaleExpression::Beta, dropped_primes: PhysicalDroppedPrimes::P1,
    };
    assert_ne!(beta, different_sequence);
    let mut expected = Vec::new();
    for q in 0..2 {
        for a in 0..4 {
            let first = if a == 0 { PhysicalMaskBranch::Support } else { PhysicalMaskBranch::Wrap };
            let mut ops = vec![PhysicalOp::WeightProduct { input_frame: 0 },
                PhysicalOp::WeightProduct { input_frame: 1 }, PhysicalOp::ProductSumBoundary,
                PhysicalOp::SupportMask { branch: first }, PhysicalOp::MaskRescale { branch: first }];
            if a > 0 {
                // Remainder is another original-P product, NOT P-U.
                ops.extend([PhysicalOp::SupportMask { branch: PhysicalMaskBranch::Remainder },
                    PhysicalOp::MaskRescale { branch: PhysicalMaskBranch::Remainder },
                    PhysicalOp::RotateMasked { shift: (4 - a) as i32 },
                    PhysicalOp::RotateRemainder { shift: -(a as i32) }, PhysicalOp::AddAligned]);
            }
            ops.push(PhysicalOp::Accumulate);
            expected.extend(ops.into_iter().map(|op| PhysicalStage { output_frame: q, displacement: a, op }));
        }
    }
    assert_eq!(plan.stages(), expected.as_slice());
    assert!(!plan.stages().iter().any(|s| matches!(s.op, PhysicalOp::Mask | PhysicalOp::SubtractMasked)));
    assert_eq!(plan.stages().iter().filter(|s| matches!(s.op, PhysicalOp::SupportMask { .. })).count(), 14);
    assert_eq!(plan.stages().iter().filter(|s| matches!(s.op, PhysicalOp::MaskRescale { .. })).count(), 14);
}

#[test]
fn physical_pc_actual_scale_not_retaggable() {
    // Exact rational ALGEBRA fixture only. Deliberately composite denominators,
    // not backend primes/parameters; never instantiate or relabel a ciphertext.
    let delta = 16_u128;
    let p1 = 15_u128;
    let p2 = 9_u128;
    let alpha = (delta * delta, p1);
    let mask_product = (delta * delta * delta, p1);
    let beta = (delta * delta * delta, p1 * p2);
    let input = (delta, 1_u128);
    let equal_rational = |left: (u128, u128), right: (u128, u128)|
        left.0 * right.1 == right.0 * left.1;
    assert_ne!(p1, delta);
    assert!(!equal_rational(alpha, input)); // Delta^2/p1 != Delta.
    assert!(!equal_rational(mask_product, alpha)); // Legacy P-U incompatible.
    assert!(!equal_rational(beta, input)); // No scale overwrite to Delta.
    assert!(!equal_rational(beta, alpha));
    // Both independent branches apply exactly the SAME second denominator.
    let wrap = (mask_product.0, mask_product.1 * p2);
    let remainder = (mask_product.0, mask_product.1 * p2);
    assert!(equal_rational(wrap, beta));
    assert!(equal_rational(remainder, beta));
}

#[test]
fn physical_pc_scale_one_sparse_mask_rounds_zero() {
    // Ideal normalized inverse-transform coefficient bound by triangle
    // inequality: |coefficient| <= m/4096 for m unit slot entries. At scale
    // ONE all exact coefficients lie strictly inside (-1/2,1/2), hence round
    // to zero. No upstream encoder/FFT/noise claim or backend execution.
    for m in [2_u128, 4, 6, 8] { // Wrap, remainder and full support F.
        assert!(2 * m < 4096);
        let positive_nearest_integer = (2 * m + 4096) / (2 * 4096);
        assert_eq!(positive_nearest_integer, 0);
        // Negative coefficient bound has the same absolute rounding result.
        assert_eq!(-(positive_nearest_integer as i128), 0);
    }
}

#[test]
fn scaled_support_prewrite_atomicity_and_work_limits() {
    let (input, out) = physical_layouts(4, 8, 2, 1);
    assert!(matches!(block_pc_physical_plan_scaled_support(input, out, 4096,
        Limits { max_work: 77, ..LIMITS }), Err(Error::WorkLimit)));
    let plan = block_pc_physical_plan_scaled_support(input, out, 4096,
        Limits { max_work: 78, ..LIMITS }).unwrap();
    assert!(matches!(block_pc_physical_plan_scaled_support(input, out, 4096,
        Limits { max_buffer_elements: 8191, ..LIMITS }), Err(Error::Buffer)));
    assert!(matches!(block_pc_physical_plan_scaled_support(input, out, 8192, LIMITS), Err(Error::Geometry)));
    let (c2_input, c2_out) = physical_layouts(4, 8, 2, 2);
    assert!(matches!(block_pc_physical_plan_scaled_support(c2_input, c2_out, 4096, LIMITS), Err(Error::Geometry)));
    let mut output = vec![SENTINEL; 4099];
    let unchanged = output.clone();
    for (a, branch) in [(0, PhysicalMaskBranch::Wrap), (0, PhysicalMaskBranch::Remainder),
        (1, PhysicalMaskBranch::Support), (3, PhysicalMaskBranch::Support),
        (4, PhysicalMaskBranch::Wrap), (usize::MAX, PhysicalMaskBranch::Support)] {
        assert_eq!(plan.materialize_branch_mask(a, branch, &mut output, LIMITS), Err(Error::Geometry));
        assert_eq!(output, unchanged);
    }
    for slot in [4096, usize::MAX] {
        assert_eq!(plan.branch_mask_value(0, PhysicalMaskBranch::Support, slot), Err(Error::Geometry));
        assert_eq!(plan.branch_mask_value(1, PhysicalMaskBranch::Remainder, slot), Err(Error::Geometry));
    }
    for (a, branch) in [(0, PhysicalMaskBranch::Support), (1, PhysicalMaskBranch::Wrap),
        (3, PhysicalMaskBranch::Remainder)] {
        assert_eq!(plan.materialize_branch_mask(a, branch, &mut output[..4095], LIMITS), Err(Error::Buffer));
        assert_eq!(plan.materialize_branch_mask(a, branch, &mut output,
            Limits { max_work: 4095, ..LIMITS }), Err(Error::WorkLimit));
        assert_eq!(plan.materialize_branch_mask(a, branch, &mut output,
            Limits { max_buffer_elements: 4098, ..LIMITS }), Err(Error::Buffer));
        assert_eq!(output, unchanged);
    }
    for (a, branch) in [(0, PhysicalMaskBranch::Support), (1, PhysicalMaskBranch::Wrap),
        (3, PhysicalMaskBranch::Remainder)] {
        plan.materialize_branch_mask(a, branch, &mut output, Limits { max_work: 4096, ..LIMITS }).unwrap();
        assert!(output[4096..].iter().all(|v| *v == SENTINEL));
    }
    // Shared limits and all existing materializer validations remain in force.
    let caps = physical_limits(Limits { max_buffer_elements: usize::MAX, max_work: usize::MAX });
    assert_eq!(caps.max_buffer_elements, 2_000_000);
    assert_eq!(caps.max_work, 100_000_000);
    let mut physical_output = vec![SENTINEL; 8195];
    let physical_unchanged = physical_output.clone();
    assert_eq!(plan.materialize_input(&[0.0; 8], &mut physical_output,
        Limits { max_work: 8199, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_weight(0, 0, 0, &[0.0; 32], &mut physical_output,
        Limits { max_work: 4127, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_bias(&[0.0; 8], &mut physical_output,
        Limits { max_work: 8199, ..LIMITS }), Err(Error::WorkLimit));
    assert_eq!(plan.materialize_bias(&[f64::NAN; 8], &mut physical_output, LIMITS), Err(Error::NonFinite));
    assert_eq!(physical_output, physical_unchanged);
    plan.materialize_input(&[0.0; 8], &mut physical_output,
        Limits { max_work: 8200, ..LIMITS }).unwrap();
    plan.materialize_weight(0, 0, 0, &[0.0; 32], &mut physical_output,
        Limits { max_work: 4128, ..LIMITS }).unwrap();
    plan.materialize_bias(&[0.0; 8], &mut physical_output,
        Limits { max_work: 8200, ..LIMITS }).unwrap();
    assert!(physical_output[8192..].iter().all(|v| *v == SENTINEL));
}

// Rectangular oracle: signed modular integer fixtures, ordinary integer dense
// contraction (no modular reduction of results). No candidate/layout indexing.
fn rect_integers(n: usize, factor: usize, modulus: usize) -> Vec<i64> {
    (0..n)
        .map(|i| ((i * factor + i / 3) % modulus) as i64 - (modulus / 2) as i64)
        .collect()
}

fn rect_dense_oracle(
    weights: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    n: usize,
    valid_columns: usize,
) -> Vec<i64> {
    let mut result = vec![0; rows * n];
    for row in 0..rows {
        for column in 0..valid_columns {
            for k in 0..inner {
                result[row * n + column] += weights[row * inner + k] * b[k * n + column];
            }
        }
    }
    result
}

// Independent Eq. (7) encoder/checker. Padding and suffix inputs are poisoned;
// expected output comes solely from dense integer contraction, not pack/unpack.
fn rect_slots(n: usize, heads: usize, capacity: usize, vc: usize, b: &[i64]) -> Vec<f64> {
    let mut slots = vec![f64::NAN; n * n * heads + 3];
    for d in 0..n {
        for t in 0..vc {
            for h in 0..heads {
                let x =
                    (d / capacity) * capacity * n * heads + ((d % capacity) * n + t) * heads + h;
                slots[x] = b[(h * n + (d + t) % n) * n + t] as f64;
            }
        }
    }
    slots
}

fn rect_assert_slots(n: usize, heads: usize, capacity: usize, values: &[f64], expected: &[i64]) {
    for d in 0..n {
        for t in 0..n {
            for h in 0..heads {
                let x =
                    (d / capacity) * capacity * n * heads + ((d % capacity) * n + t) * heads + h;
                assert_eq!(values[x], expected[(h * n + (d + t) % n) * n + t] as f64);
            }
        }
    }
    assert!(values[n * n * heads..].iter().all(|v| *v == SENTINEL));
}

#[test]
fn rectangular_pc_asymmetric_4_to_8_to_4_has_independent_integer_oracle() {
    for n in [1, 2, 4] {
        for capacity in (1..=n).filter(|c| n % c == 0) {
            for vc in [0, 1, n - 1, n] {
                let input =
                    Layout::new(n, n, n, vc, 4 / n, capacity, Diagonal::Lower, LIMITS).unwrap();
                let expanded =
                    Layout::new(n, n, n, vc, 8 / n, capacity, Diagonal::Lower, LIMITS).unwrap();
                let b = rect_integers(4 * n, 7, 19);
                let up = rect_integers(8 * 4, 11, 23);
                let down = rect_integers(4 * 8, 13, 29);
                let expected_up = rect_dense_oracle(&up, &b, 8, 4, n, vc);
                // Second oracle consumes independent first oracle, never candidate.
                let expected_down = rect_dense_oracle(&down, &expected_up, 4, 8, n, vc);
                let mut first = vec![SENTINEL; expanded.elements() + 3];
                rectangular_block_pc_mm(
                    input,
                    &up.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &rect_slots(n, 4 / n, capacity, vc, &b),
                    expanded,
                    &mut first,
                    LIMITS,
                )
                .unwrap();
                rect_assert_slots(n, 8 / n, capacity, &first, &expected_up);
                let mut second = vec![SENTINEL; input.elements() + 3];
                rectangular_block_pc_mm(
                    expanded,
                    &down.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &first,
                    input,
                    &mut second,
                    LIMITS,
                )
                .unwrap();
                rect_assert_slots(n, 4 / n, capacity, &second, &expected_down);
            }
        }
    }
}

#[test]
fn rectangular_pc_both_orientations_virtual_tail_blocks_and_counts() {
    // Unequal and non-multiple block-row counts exercise zero virtual tail blocks.
    // Physical blocks stay full; nondividing diagonal capacity remains rejected.
    for n in [1_usize, 2, 4, 6] {
        for (ih, oh) in [(1, 3), (3, 1), (2, 3), (3, 2)] {
            for capacity in (1..=n).filter(|c| n % c == 0) {
                let vc = n.saturating_sub(1);
                let input =
                    Layout::new(n, n, n, vc, ih, capacity, Diagonal::Lower, LIMITS).unwrap();
                let out = Layout::new(n, n, n, vc, oh, capacity, Diagonal::Lower, LIMITS).unwrap();
                let weights = rect_integers(n * oh * n * ih, 17, 31);
                let b = rect_integers(n * ih * n, 7, 19);
                let expected = rect_dense_oracle(&weights, &b, n * oh, n * ih, n, vc);
                let mut output = vec![SENTINEL; out.elements() + 3];
                rectangular_block_pc_mm(
                    input,
                    &weights.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &rect_slots(n, ih, capacity, vc, &b),
                    out,
                    &mut output,
                    LIMITS,
                )
                .unwrap();
                rect_assert_slots(n, oh, capacity, &output, &expected);
                let h = ih.max(oh);
                let frames = n / capacity;
                assert_eq!(
                    rectangular_block_pc_schedule(input, out).unwrap(),
                    Schedule {
                        plaintext_products: h * frames * n + h * frames,
                        input_rotations: frames * (capacity - 1),
                        output_rotations: 2 * (h - 1) * frames,
                        rescale_after_product_sums: h * frames,
                        ..Schedule::default()
                    }
                );
            }
        }
    }
}

#[test]
fn rectangular_pc_preflight_exact_limits_and_atomic_rejections() {
    let input = layout(2, 2, 2, 1, Diagonal::Lower);
    let out = layout(2, 2, 4, 1, Diagonal::Lower);
    let weights = vec![1.0; 32];
    let b = vec![2.0; input.elements()];
    let mut output = vec![SENTINEL; out.elements()];
    let before = output.clone();
    // n=2,H=4: 64 virtual weights + 2*16 virtual lanes + 128 terms.
    let required_work = 224;
    let exact = Limits {
        max_buffer_elements: 64,
        max_work: required_work,
    };
    let mut bad_weights = weights.clone();
    bad_weights[31] = f64::NAN;
    let mut bad_b = b.clone();
    bad_b[input.elements() - 1] = f64::INFINITY;
    for (w, bv, olen, limits, error) in [
        (
            &weights[..31],
            &bad_b[..],
            out.elements(),
            exact,
            Error::Buffer,
        ),
        (
            &bad_weights[..],
            &b[..b.len() - 1],
            out.elements(),
            exact,
            Error::Buffer,
        ),
        (
            &bad_weights[..],
            &b[..],
            out.elements() - 1,
            exact,
            Error::Buffer,
        ),
        (
            &bad_weights[..],
            &b[..],
            out.elements(),
            Limits {
                max_work: required_work - 1,
                ..exact
            },
            Error::WorkLimit,
        ),
        (
            &bad_weights[..],
            &b[..],
            out.elements(),
            Limits {
                max_buffer_elements: 63,
                ..exact
            },
            Error::Buffer,
        ),
        (
            &bad_weights[..],
            &b[..],
            out.elements(),
            exact,
            Error::NonFinite,
        ),
        (
            &weights[..],
            &bad_b[..],
            out.elements(),
            exact,
            Error::NonFinite,
        ),
    ] {
        assert_eq!(
            rectangular_block_pc_mm(input, w, bv, out, &mut output[..olen], limits),
            Err(error)
        );
        assert_eq!(output, before);
    }
    // Oversized supplied slices are charged in full, even unused suffixes.
    let oversized = vec![1.0; 65];
    for (w, bv) in [(&oversized[..], &b[..]), (&weights[..], &oversized[..])] {
        assert_eq!(
            rectangular_block_pc_mm(input, w, bv, out, &mut output, exact),
            Err(Error::Buffer)
        );
        assert_eq!(output, before);
    }
    let mut oversized_output = vec![SENTINEL; 65];
    assert_eq!(
        rectangular_block_pc_mm(input, &weights, &b, out, &mut oversized_output, exact),
        Err(Error::Buffer)
    );
    assert!(oversized_output.iter().all(|v| *v == SENTINEL));
    rectangular_block_pc_mm(input, &weights, &b, out, &mut output, exact).unwrap();
    assert_eq!(output, vec![8.0; out.elements()]);
    // Reverse rectangle pays the SAME full virtual traversal, not real extents.
    let reverse_b = vec![2.0; out.elements()];
    let mut reverse_output = vec![SENTINEL; input.elements() + 3];
    let reverse_before = reverse_output.clone();
    assert_eq!(
        rectangular_block_pc_mm(
            out,
            &weights,
            &reverse_b,
            input,
            &mut reverse_output,
            Limits {
                max_work: 223,
                ..exact
            }
        ),
        Err(Error::WorkLimit)
    );
    assert_eq!(reverse_output, reverse_before);
    rectangular_block_pc_mm(out, &weights, &reverse_b, input, &mut reverse_output, exact).unwrap();
    assert!(
        reverse_output[..input.elements()]
            .iter()
            .all(|v| *v == 16.0)
    );
    assert!(
        reverse_output[input.elements()..]
            .iter()
            .all(|v| *v == SENTINEL)
    );

    let layout_exact = Limits {
        max_buffer_elements: 16,
        max_work: 16,
    };
    assert_eq!(
        Layout::new(2, 2, 2, 2, 4, 1, Diagonal::Lower, layout_exact),
        Ok(out)
    );
    assert_eq!(
        Layout::new(
            2,
            2,
            2,
            2,
            4,
            1,
            Diagonal::Lower,
            Limits {
                max_buffer_elements: 15,
                ..layout_exact
            }
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        Layout::new(
            2,
            2,
            2,
            2,
            4,
            1,
            Diagonal::Lower,
            Limits {
                max_work: 15,
                ..layout_exact
            }
        ),
        Err(Error::WorkLimit)
    );

    // Exactly full capacity and one-short nondividing capacity; no widening.
    let full = layout(4, 4, 2, 4, Diagonal::Lower);
    assert!(rectangular_block_pc_schedule(full, full).is_ok());
    let partial_frame = layout(4, 4, 2, 3, Diagonal::Lower);
    let before = output.clone();
    for invalid in [
        partial_frame,
        layout(4, 4, 2, 4, Diagonal::Upper),
        Layout::new(4, 4, 3, 4, 2, 4, Diagonal::Lower, LIMITS).unwrap(),
    ] {
        assert_eq!(
            rectangular_block_pc_schedule(invalid, full),
            Err(Error::Geometry)
        );
        assert_eq!(
            rectangular_block_pc_mm(invalid, &[], &[], full, &mut output, LIMITS),
            Err(Error::Geometry)
        );
        assert_eq!(
            rectangular_block_pc_schedule(full, invalid),
            Err(Error::Geometry)
        );
        assert_eq!(
            rectangular_block_pc_mm(full, &[], &[], invalid, &mut output, LIMITS),
            Err(Error::Geometry)
        );
        assert_eq!(output, before);
    }
    assert_eq!(
        rectangular_block_pc_schedule(input, full),
        Err(Error::Geometry)
    );
    let wrong_columns = Layout::new(2, 2, 2, 1, 4, 1, Diagonal::Lower, LIMITS).unwrap();
    assert_eq!(
        rectangular_block_pc_schedule(input, wrong_columns),
        Err(Error::Geometry)
    );
    let wrong_capacity = layout(2, 2, 4, 2, Diagonal::Lower);
    assert_eq!(
        rectangular_block_pc_schedule(input, wrong_capacity),
        Err(Error::Geometry)
    );
    let unlimited = Limits {
        max_buffer_elements: usize::MAX,
        max_work: usize::MAX,
    };
    let huge = Layout::new(1, 1, 1, 1, usize::MAX / 4, 1, Diagonal::Lower, unlimited).unwrap();
    assert_eq!(
        rectangular_block_pc_schedule(huge, huge),
        Err(Error::Overflow)
    );
    let before = output.clone();
    assert_eq!(
        rectangular_block_pc_mm(huge, &[], &[], huge, &mut output, unlimited),
        Err(Error::Overflow)
    );
    assert_eq!(output, before);

    // Caller cannot widen the rectangular owner's 2M virtual-storage ceiling.
    let small = layout(1, 1, 1, 1, Diagonal::Lower);
    let virtual_over_cap = Layout::new(1, 1, 1, 1, 1415, 1, Diagonal::Lower, unlimited).unwrap();
    let mut cap_output = vec![SENTINEL; 1415];
    assert_eq!(
        rectangular_block_pc_mm(
            small,
            &[f64::NAN; 1415],
            &[f64::NAN],
            virtual_over_cap,
            &mut cap_output,
            unlimited
        ),
        Err(Error::Buffer)
    );
    assert!(cap_output.iter().all(|v| *v == SENTINEL));

    // Virtual extents fit 2M but 135,528,448 work exceeds the fixed 100M cap.
    // Real slices stay bounded; rejection must precede poisoned operand scans.
    let work_input = layout(128, 128, 1, 128, Diagonal::Lower);
    let work_out = layout(128, 128, 8, 128, Diagonal::Lower);
    let work_weights = vec![f64::NAN; 128 * 128 * 8];
    let work_b = vec![f64::NAN; work_input.elements()];
    let mut work_output = vec![SENTINEL; work_out.elements()];
    assert_eq!(
        rectangular_block_pc_mm(
            work_input,
            &work_weights,
            &work_b,
            work_out,
            &mut work_output,
            unlimited
        ),
        Err(Error::WorkLimit)
    );
    assert!(work_output.iter().all(|v| *v == SENTINEL));
}

fn rect_vector_rotate(values: &[f64], shift: usize) -> Vec<f64> {
    (0..values.len())
        .map(|x| values[(x + shift) % values.len()])
        .collect()
}

// Independent test-only Algorithm 1 simulation. Materialize packed vectors,
// rotated B, plaintext products, COMPLETED block products, then head masks and
// two alignment rotations. No production Layout/address/modular helpers.
#[allow(clippy::too_many_arguments)]
fn rect_algorithm1_vector_oracle(
    n: usize,
    ih: usize,
    oh: usize,
    c: usize,
    vc: usize,
    weights: &[f64],
    dense_b: &[f64],
) -> Vec<f64> {
    let h = ih.max(oh);
    let frames = n / c;
    let extent = c * n * h;
    let mut rotated = vec![vec![Vec::new(); c]; frames];
    for (j, rotations) in rotated.iter_mut().enumerate() {
        let mut frame = vec![0.0; extent];
        for (x, value) in frame.iter_mut().enumerate() {
            let k = x % h;
            let t = x / h % n;
            let diagonal = c * j + x / (n * h);
            if k < ih && t < vc {
                *value = dense_b[(k * n + (diagonal + t) % n) * n + t];
            }
        }
        for (ell, rotation) in rotations.iter_mut().enumerate() {
            *rotation = rect_vector_rotate(&frame, n * h * ell);
        }
    }
    let mut completed = vec![vec![vec![0.0; extent]; frames]; h];
    for (a, partial_frames) in completed.iter_mut().enumerate() {
        for (r, partial) in partial_frames.iter_mut().enumerate() {
            for (j, rotations) in rotated.iter().enumerate() {
                for (ell, operand) in rotations.iter().enumerate() {
                    let plaintext: Vec<_> = (0..extent)
                        .map(|x| {
                            let k = x % h;
                            let i = (a + k) % h;
                            let t = x / h % n;
                            let local_diagonal = x / (n * h);
                            let q = c * r + local_diagonal;
                            let rotated_diagonal = c * j + (local_diagonal + ell) % c;
                            if i < oh && k < ih {
                                weights[(i * n + (q + t) % n) * (n * ih)
                                    + k * n
                                    + (rotated_diagonal + t) % n]
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let products: Vec<_> =
                        plaintext.iter().zip(operand).map(|(a, b)| a * b).collect();
                    for (acc, product) in partial.iter_mut().zip(products) {
                        *acc += product;
                    }
                }
            }
        }
    }
    let mut aligned = vec![vec![0.0; extent]; frames];
    for (a, partial_frames) in completed.iter().enumerate() {
        for (partial, acc) in partial_frames.iter().zip(&mut aligned) {
            let right: Vec<_> = partial
                .iter()
                .enumerate()
                .map(|(x, v)| if x % h >= h - a { *v } else { 0.0 })
                .collect();
            let left: Vec<_> = partial
                .iter()
                .zip(&right)
                .map(|(v, masked)| v - masked)
                .collect();
            let right_aligned = rect_vector_rotate(&right, h - a);
            let left_aligned = rect_vector_rotate(&left, (extent - a) % extent);
            for ((acc, right), left) in acc.iter_mut().zip(right_aligned).zip(left_aligned) {
                *acc += right + left;
            }
        }
    }
    let mut result = vec![SENTINEL; n * n * oh + 3];
    for (r, frame) in aligned.iter().enumerate() {
        for local in 0..c {
            for t in 0..n {
                for i in 0..oh {
                    result[r * c * n * oh + (local * n + t) * oh + i] = if t < vc {
                        frame[(local * n + t) * h + i]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    result
}

#[test]
fn rectangular_pc_algorithm1_stages_preserve_grouping_and_alignment() {
    for (n, ih, oh, c, vc) in [
        (2, 2, 1, 1, 2),
        (2, 1, 2, 2, 1),
        (2, 2, 2, 2, 2),
        (4, 2, 3, 2, 3),
        (4, 3, 2, 4, 4),
        (4, 3, 3, 2, 4),
    ] {
        let input = Layout::new(n, n, n, vc, ih, c, Diagonal::Lower, LIMITS).unwrap();
        let out = Layout::new(n, n, n, vc, oh, c, Diagonal::Lower, LIMITS).unwrap();
        let weights: Vec<_> = rect_integers(n * oh * n * ih, 11, 23)
            .iter()
            .map(|v| *v as f64 / 7.0)
            .collect();
        let b = rect_integers(n * n * ih, 7, 19);
        let dense_b: Vec<_> = b.iter().map(|v| *v as f64 / 11.0).collect();
        let mut slots = rect_slots(n, ih, c, vc, &b);
        for value in &mut slots[..input.elements()] {
            if value.is_finite() {
                *value /= 11.0;
            }
        }
        let expected = rect_algorithm1_vector_oracle(n, ih, oh, c, vc, &weights, &dense_b);
        let mut output = vec![SENTINEL; out.elements() + 3];
        rectangular_block_pc_mm(input, &weights, &slots, out, &mut output, LIMITS).unwrap();
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let input = layout(2, 2, 2, 1, Diagonal::Lower);
    let out = layout(2, 2, 1, 1, Diagonal::Lower);
    let weights = [1e16, 0.0, -1e16, 1.0, 0.0, 0.0, 0.0, 0.0];
    let expected = rect_algorithm1_vector_oracle(2, 2, 1, 1, 2, &weights, &[1.0; 8]);
    let mut output = vec![SENTINEL; out.elements() + 3];
    rectangular_block_pc_mm(input, &weights, &[1.0; 8], out, &mut output, LIMITS).unwrap();
    assert_eq!(expected[0], 0.0);
    assert_eq!(output[0], 0.0);
    let flattened = weights[..4].iter().fold(0.0, |sum, weight| sum + weight);
    assert_eq!(flattened, 1.0);
    assert_eq!(output, expected);
}

const LIMITS: Limits = Limits {
    max_buffer_elements: 2_000_000,
    max_work: 100_000_000,
};
fn layout(r: usize, c: usize, h: usize, p: usize, d: Diagonal) -> Layout {
    Layout::new(r, c, r, c, h, p, d, LIMITS).unwrap()
}
fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7 % 13) as i32 - 6) as f64).collect()
}
fn packed(l: Layout, a: &[f64]) -> Vec<f64> {
    let mut v = vec![777.0; l.elements()];
    l.pack(a, &mut v, LIMITS).unwrap();
    v
}
fn unpacked(l: Layout, a: &[f64]) -> Vec<f64> {
    let mut v = vec![777.0; l.dense_elements()];
    l.unpack(a, &mut v, LIMITS).unwrap();
    v
}
// Independent row-major contraction, never calls diagonal/address helpers.
fn dense_mm(a: &[f64], b: &[f64], m: usize, n: usize, h: usize) -> Vec<f64> {
    let mut c = vec![0.0; h * m * n];
    for z in 0..h {
        for i in 0..m {
            for j in 0..n {
                for k in 0..n {
                    c[(z * m + i) * n + j] += a[(z * m + i) * n + k] * b[(z * n + k) * n + j];
                }
            }
        }
    }
    c
}

// Independent dense coordinates only: no layout, diagonal, rotation, pack or
// unpack helpers. Pair inner coordinates k and k+m/2 in the cyclic column order
// starting at j. Im(X) = (A[i,k]*B[k+m/2,j]-A[i,k+m/2]*B[k,j])/2.
#[allow(clippy::too_many_arguments)] // Keep independent dense geometry axes explicit.
fn dense_paired_oracle(
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    heads: usize,
    vr: usize,
    vk: usize,
    vc: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut real = vec![0.0; heads * n * n];
    let mut imaginary = real.clone();
    for h in 0..heads {
        for i in 0..vr {
            for j in 0..vc {
                let av = |k| if k < vk { a[(h * n + i) * m + k] } else { 0.0 };
                let bv = |k| if k < vk { b[(h * m + k) * n + j] } else { 0.0 };
                for k in 0..m {
                    real[(h * n + i) * n + j] += av(k) * bv(k);
                }
                for l in 0..m / 2 {
                    let k = (j + l) % m;
                    let paired = (k + m / 2) % m;
                    imaginary[(h * n + i) * n + j] +=
                        (av(k) * bv(paired) - av(paired) * bv(k)) * 0.5;
                }
            }
        }
    }
    (real, imaginary)
}

const SENTINEL: f64 = 12345.0;
// Coordinate-indexed checks retain the original lane, suffix and panic behavior.
#[allow(clippy::needless_range_loop)]
fn assert_padding_and_suffix(l: Layout, values: &[f64]) {
    for i in 0..l.elements() {
        if l.lane(i).unwrap().is_none() {
            assert_eq!(values[i], 0.0, "padding lane {i}");
        }
    }
    assert!(values[l.elements()..].iter().all(|v| *v == SENTINEL));
}
// Visit exactly the layout's coordinates, not a potentially shorter input slice.
#[allow(clippy::needless_range_loop)]
fn poison_padding(l: Layout, values: &mut [f64]) {
    for i in 0..l.elements() {
        if l.lane(i).unwrap().is_none() {
            values[i] = f64::NAN;
        }
    }
}

#[test]
// Explicit dense/packed coordinates are the independent bijection oracle.
#[allow(clippy::needless_range_loop)]
fn all_addresses_heads_diagonals_and_partial_frames_are_bijective() {
    for (r, c) in [(1, 1), (1, 8), (2, 8), (8, 2), (4, 4), (8, 8)] {
        for h in [1, 3, 12] {
            for p in 1..=r.min(c) {
                for d in [Diagonal::Upper, Diagonal::Lower] {
                    let l = layout(r, c, h, p, d);
                    let a = data(l.dense_elements());
                    let v = packed(l, &a);
                    let mut seen = vec![false; l.elements()];
                    for z in 0..h {
                        for i in 0..r {
                            for j in 0..c {
                                let x = l.address(z, i, j).unwrap();
                                assert!(!seen[x]);
                                seen[x] = true;
                                assert_eq!(l.lane(x), Ok(Some((z, i, j))));
                                assert_eq!(v[x], a[(z * r + i) * c + j]);
                            }
                        }
                    }
                    for x in 0..l.elements() {
                        assert_eq!(seen[x], l.lane(x).unwrap().is_some());
                    }
                    assert_eq!(unpacked(l, &v), a);
                }
            }
        }
    }
}

#[test]
fn every_basis_vector_transposes_across_rectangles_and_orientations() {
    for (r, c) in [(2, 8), (8, 2), (4, 4)] {
        for d in [Diagonal::Upper, Diagonal::Lower] {
            let a = layout(r, c, 3, 2, d);
            let b = layout(c, r, 3, 1, d);
            for basis in 0..a.dense_elements() {
                let mut dense = vec![0.0; a.dense_elements()];
                dense[basis] = 1.0;
                let mut out = vec![9.0; b.elements()];
                transpose(a, &packed(a, &dense), b, &mut out, LIMITS).unwrap();
                let mut oracle = vec![0.0; dense.len()];
                for h in 0..3 {
                    for i in 0..r {
                        for j in 0..c {
                            oracle[(h * c + j) * r + i] = dense[(h * r + i) * c + j];
                        }
                    }
                }
                assert_eq!(unpacked(b, &out), oracle);
            }
        }
    }
}

#[test]
// Retain coordinate indexing and the original masked-rotation oracle order.
#[allow(clippy::needless_range_loop)]
fn replication_and_masked_rotations_preserve_heads_and_padding() {
    for d in [Diagonal::Upper, Diagonal::Lower] {
        let l = Layout::new(4, 8, 3, 5, 3, 3, d, LIMITS).unwrap();
        let dense = data(l.dense_elements());
        let mut a = packed(l, &dense);
        for i in 0..a.len() {
            if l.lane(i).unwrap().is_none() {
                a[i] = f64::NAN;
            }
        }
        let mut replicated = vec![9.0; l.diagonals * l.frame_slots];
        replicate(l, &a, &mut replicated, LIMITS).unwrap();
        for diag in 0..l.diagonals {
            for copy in 0..l.capacity {
                for t in 0..l.length {
                    for h in 0..l.heads {
                        let (r, c) = l.coordinates(diag, t);
                        let expected = if r < 3 && c < 5 {
                            dense[(h * 4 + r) * 8 + c]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            replicated[diag * l.frame_slots + (copy * l.length + t) * l.heads + h],
                            expected
                        );
                    }
                }
            }
        }
        for shift in [0, 1, 7, 8, 9, usize::MAX] {
            let mut out = vec![9.0; l.elements()];
            internal_rotate(l, &a, shift, &mut out, LIMITS).unwrap();
            for i in 0..out.len() {
                if let Some((h, _, _)) = l.lane(i).unwrap() {
                    let diag = (i / l.frame_slots) * l.capacity
                        + (i % l.frame_slots) / (l.length * l.heads);
                    let t = (i / l.heads) % l.length;
                    let (r, c) = l.coordinates(diag, (t + shift % l.length) % l.length);
                    assert_eq!(
                        out[i],
                        if r < 3 && c < 5 {
                            dense[(h * 4 + r) * 8 + c]
                        } else {
                            0.0
                        }
                    );
                } else {
                    assert_eq!(out[i], 0.0);
                }
            }
        }
    }
}

#[test]
fn lower_products_match_dense_oracle_for_all_small_geometry_and_valid_rectangles() {
    for m in [1, 2, 4] {
        for n in [m, 2 * m] {
            for h in [1, 3] {
                for p in 1..=m {
                    for (vr, vk, vc) in [(m, n, n), (m - 1, n - 1, n - 1), (1, 1, 1), (0, 0, 0)] {
                        let a = Layout::new(m, n, vr, vk, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let b = Layout::new(n, n, vk, vc, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let c = Layout::new(m, n, vr, vc, h, p, Diagonal::Lower, LIMITS).unwrap();
                        let pa = packed(a, &data(a.dense_elements()));
                        let pb = packed(b, &data(b.dense_elements()));
                        let da = unpacked(a, &pa);
                        let db = unpacked(b, &pb);
                        let expected = dense_mm(&da, &db, m, n, h);
                        let mut out = vec![999.0; c.elements()];
                        lower_mm(a, &pa, b, &pb, c, &mut out, LIMITS).unwrap();
                        assert_eq!(unpacked(c, &out), expected);
                        if m % p == 0 && a == c {
                            let mut scratch = vec![999.0; 2 * c.elements()];
                            let mut optimized = vec![999.0; c.elements()];
                            optimized_lower_mm(
                                a,
                                &pa,
                                b,
                                &pb,
                                c,
                                &mut scratch,
                                &mut optimized,
                                LIMITS,
                            )
                            .unwrap();
                            assert_eq!(optimized, out);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_lower_basis_pair_checks_all_head_and_diagonal_boundaries() {
    let a = layout(2, 4, 2, 1, Diagonal::Lower);
    let optimized_a = layout(2, 4, 2, 2, Diagonal::Lower);
    let b = layout(4, 4, 2, 2, Diagonal::Lower);
    for i in 0..a.dense_elements() {
        for j in 0..b.dense_elements() {
            let mut da = vec![0.0; a.dense_elements()];
            da[i] = 1.0;
            let mut db = vec![0.0; b.dense_elements()];
            db[j] = 1.0;
            let mut out = vec![9.0; a.elements()];
            lower_mm(a, &packed(a, &da), b, &packed(b, &db), a, &mut out, LIMITS).unwrap();
            assert_eq!(unpacked(a, &out), dense_mm(&da, &db, 2, 4, 2));
            let mut optimized = vec![9.0; optimized_a.elements()];
            let mut scratch = vec![9.0; 2 * optimized_a.elements()];
            optimized_lower_mm(
                optimized_a,
                &packed(optimized_a, &da),
                b,
                &packed(b, &db),
                optimized_a,
                &mut scratch,
                &mut optimized,
                LIMITS,
            )
            .unwrap();
            assert_eq!(
                unpacked(optimized_a, &optimized),
                dense_mm(&da, &db, 2, 4, 2)
            );
        }
    }
}

#[test]
fn block_pc_full_cross_block_weights_match_independent_dense_oracle() {
    for n in [1, 2, 4] {
        for h in [1, 2, 3] {
            for p in 1..=n {
                let l = layout(n, n, h, p, Diagonal::Lower);
                // Algorithm 1 requires capacity dividing n, not arbitrary final frames.
                if n % p != 0 {
                    continue;
                }
                let d = n * h;
                let weights = data(d * d);
                let b = data(l.dense_elements());
                let mut out = vec![9.0; l.elements()];
                block_pc_mm(l, &weights, &packed(l, &b), &mut out, LIMITS).unwrap();
                let mut expected = vec![0.0; b.len()];
                for i in 0..d {
                    for t in 0..n {
                        for k in 0..d {
                            expected[i * n + t] += weights[i * d + k] * b[k * n + t];
                        }
                    }
                }
                assert_eq!(unpacked(l, &out), expected);
            }
        }
    }
}

#[test]
fn source_shapes_enumerate_without_he_objects_and_counts_have_separate_stages() {
    let l = layout(64, 128, 12, 16, Diagonal::Lower);
    assert_eq!(
        (l.frames(), l.frame_slots(), l.elements()),
        (4, 24576, 98304)
    );
    let mut seen = vec![false; l.elements()];
    for h in 0..12 {
        for r in 0..64 {
            for c in 0..128 {
                let i = l.address(h, r, c).unwrap();
                assert!(!seen[i]);
                seen[i] = true;
            }
        }
    }
    assert!(seen.iter().all(|x| *x));
    let s = lower_schedule(l).unwrap();
    assert_eq!(s.tensor_products, 512);
    assert_eq!(s.plaintext_products, 4 * 4 * 127);
    assert_eq!(s.input_rotations, 2 * 4 * 127);
    assert_eq!(s.relinearize_after_product_sums, 4);
    assert_eq!(s.conjugations, 0);
    let optimized = optimized_lower_schedule(l).unwrap();
    assert_eq!(optimized.tensor_products, 512);
    assert_eq!(optimized.plaintext_products, 3 * 4 * 127);
    assert_eq!(optimized.input_rotations, 4 * 127);
    assert_eq!(optimized.output_rotations, 4);
    assert_eq!(optimized.rescale_after_product_sums, 8);
    assert_eq!(optimized.relinearize_after_product_sums, 4);
    assert_eq!(optimized.relinearize_before_output_rotation, 4);
    assert_eq!(optimized.conjugations, 0);
    let pc = block_pc_schedule(layout(128, 128, 6, 32, Diagonal::Lower)).unwrap();
    assert_eq!(pc.plaintext_products, 6 * 4 * 128 + 6 * 4);
    assert_eq!(pc.input_rotations, 4 * 31);
    assert_eq!(pc.output_rotations, 2 * 5 * 4);
    assert_eq!(pc.tensor_products, 0);
    assert_eq!(pc.relinearize_after_product_sums, 0);
    assert_eq!(pc.rescale_after_product_sums, 24);
}

#[test]
fn printed_multiblock_rotation_has_explicit_counterexample() {
    // m=n=4,c=2,j=0,l=2,r=t=0. Eq.(10) requires L_2(A)[2]
    // = A[0,2]. Algorithm 2 line 7 rotates frame 0 by +2 and exposes
    // L_0(A)[2] = A[2,2]; its j-1 branch is empty for l mod c=0.
    let l = layout(4, 4, 1, 2, Diagonal::Lower);
    let a: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let slots = packed(l, &a);
    assert_eq!(slots[2], 10.0);
    assert_eq!(slots[l.address(0, 0, 2).unwrap()], 2.0);
    // Preserve the literal printed selection above as a negative control.
    // The approved paper-derived correction selects frame 1, not frame 0.
    let mut b = vec![0.0; 16];
    b[2 * 4] = 1.0;
    let mut scratch = vec![SENTINEL; 2 * l.elements()];
    let mut output = vec![SENTINEL; l.elements()];
    optimized_lower_mm(
        l,
        &slots,
        l,
        &packed(l, &b),
        l,
        &mut scratch,
        &mut output,
        LIMITS,
    )
    .unwrap();
    assert_eq!(output[l.address(0, 0, 0).unwrap()], 2.0);
    assert_ne!(output[l.address(0, 0, 0).unwrap()], slots[2]);
}

#[test]
fn optimized_corrected_algorithm_matches_240_audited_geometries_and_operand_coordinates() {
    let mut geometries = 0;
    let mut coordinates = 0;
    for m in 1..=8 {
        for n in (m..=16).step_by(m) {
            for c in (1..=m).filter(|c| m % c == 0) {
                for heads in [1, 2, 3] {
                    let left = layout(m, n, heads, c, Diagonal::Lower);
                    let right = layout(n, n, heads, c, Diagonal::Lower);
                    let frames = m / c;
                    let s = c * n * heads;
                    // Independent labeled operand coordinates: check each
                    // summand, including both mask branches and q>=M. Do not
                    // use Layout address/lane or the kernel's modular helpers.
                    for j in 0..frames {
                        for r in 0..c {
                            for t in 0..n {
                                for h in 0..heads {
                                    for l in 0..n {
                                        let q = l / c;
                                        let b = l % c;
                                        let x = (r * n + t) * heads + h;
                                        let y = if t >= n - l {
                                            (x + s - n * heads) % s
                                        } else {
                                            x
                                        };
                                        let src = (y + s + l * heads - n * b * heads) % s;
                                        let f = (j + frames - q % frames + frames
                                            - usize::from(r < b))
                                            % frames;
                                        let sd = src / (n * heads);
                                        let st = (src / heads) % n;
                                        let sh = src % heads;
                                        assert_eq!(
                                            (sh, (c * f + sd + st) % m, st),
                                            (h, (c * j + r + t) % m, (t + l) % n)
                                        );
                                        coordinates += 1;
                                    }
                                }
                            }
                        }
                    }
                    // Independent dense contraction, distinct head-sensitive
                    // signed integer operands (exactly representable in f64).
                    let a: Vec<_> = (0..heads * m * n)
                        .map(|i| ((i * 7 % 19) as i32 - 9) as f64)
                        .collect();
                    let b: Vec<_> = (0..heads * n * n)
                        .map(|i| ((i * 11 % 23) as i32 - 11) as f64)
                        .collect();
                    let expected = dense_mm(&a, &b, m, n, heads);
                    let mut pa = packed(left, &a);
                    let mut pb = packed(right, &b);
                    // Unused input suffix is never validated or consumed.
                    pa.extend([f64::NAN; 3]);
                    pb.extend([f64::INFINITY; 3]);
                    let mut output = vec![SENTINEL; left.elements() + 3];
                    let mut scratch = vec![SENTINEL; 2 * left.elements() + 3];
                    optimized_lower_mm(
                        left,
                        &pa,
                        right,
                        &pb,
                        left,
                        &mut scratch,
                        &mut output,
                        LIMITS,
                    )
                    .unwrap();
                    // Inspect output by independently derived frame map rather
                    // than unpacking through the same layout implementation.
                    for j in 0..frames {
                        for r in 0..c {
                            for t in 0..n {
                                for h in 0..heads {
                                    assert_eq!(
                                        output[j * s + (r * n + t) * heads + h],
                                        expected[(h * m + (c * j + r + t) % m) * n + t],
                                        "m={m}, n={n}, c={c}, H={heads}"
                                    );
                                }
                            }
                        }
                    }
                    assert_padding_and_suffix(left, &output);
                    assert!(
                        scratch[2 * left.elements()..]
                            .iter()
                            .all(|v| *v == SENTINEL)
                    );
                    let delayed = frames * usize::from(n > 1);
                    assert_eq!(
                        optimized_lower_schedule(left).unwrap(),
                        Schedule {
                            tensor_products: frames * n,
                            plaintext_products: 3 * frames * (n - 1),
                            input_rotations: frames * (n - 1),
                            output_rotations: delayed,
                            conjugations: 0,
                            rescale_after_product_sums: frames + delayed,
                            relinearize_after_product_sums: frames,
                            relinearize_before_output_rotation: delayed,
                        }
                    );
                    geometries += 1;
                }
            }
        }
    }
    assert_eq!(geometries, 240);
    assert_eq!(coordinates, 209880);
}

#[test]
fn optimized_single_frame_preserves_original_output_and_scratch_bits() {
    for (m, n, heads) in [(1, 1, 1), (1, 8, 3), (2, 8, 2), (4, 12, 3)] {
        let left = layout(m, n, heads, m, Diagonal::Lower);
        let right = layout(n, n, heads, m, Diagonal::Lower);
        let da: Vec<_> = (0..left.dense_elements())
            .map(|i| ((i * 7 % 19) as f64 - 9.0) / 7.0)
            .collect();
        let db: Vec<_> = (0..right.dense_elements())
            .map(|i| ((i * 11 % 23) as f64 - 11.0) / 11.0)
            .collect();
        let a = packed(left, &da);
        let b = packed(right, &db);
        let s = left.elements();
        let block = n * heads;
        let mut original_scratch = vec![0.0; 2 * s];
        // Literal former single-frame loop, including original summation order.
        for r in 0..m {
            for t in 0..n {
                for h in 0..heads {
                    let x = left.raw(r, t, h);
                    for l in 0..n {
                        let wrapped = t >= n - l;
                        let y = if wrapped { minus(x, block, s) } else { x };
                        let shift = minus(l * heads, (l % m) * block, s);
                        let product = a[plus(y, shift, s)] * right.read(&b, l, t, h);
                        if wrapped {
                            original_scratch[s + y] += product;
                        } else {
                            original_scratch[x] += product;
                        }
                    }
                }
            }
        }
        let original_output: Vec<_> = (0..s)
            .map(|x| original_scratch[x] + original_scratch[s + minus(x, block, s)])
            .collect();
        let mut output = vec![SENTINEL; s];
        let mut scratch = vec![SENTINEL; 2 * s];
        optimized_lower_mm(left, &a, right, &b, left, &mut scratch, &mut output, LIMITS).unwrap();
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            original_output
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            scratch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            original_scratch
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn optimized_multiblock_valid_rectangles_ignore_poison_and_preserve_suffixes() {
    for (m, n, c) in [(2, 2, 1), (4, 12, 1), (4, 12, 2), (6, 12, 3), (4, 8, 4)] {
        for (vr, vk) in [(m, n), (m - 1, n - 1), (1, 1), (0, n), (m, 0), (0, 0)] {
            let left = Layout::new(m, n, vr, vk, 3, c, Diagonal::Lower, LIMITS).unwrap();
            let right = Layout::new(n, n, vk, vk, 3, c, Diagonal::Lower, LIMITS).unwrap();
            let mut da = data(left.dense_elements());
            let mut db = data(right.dense_elements());
            // Zero extension constructed in row-major space, independent of
            // diagonal layout and poison detection helpers.
            for h in 0..3 {
                for i in 0..m {
                    for k in 0..n {
                        if i >= vr || k >= vk {
                            da[(h * m + i) * n + k] = 0.0;
                        }
                    }
                }
                for k in 0..n {
                    for t in 0..n {
                        if k >= vk || t >= vk {
                            db[(h * n + k) * n + t] = 0.0;
                        }
                    }
                }
            }
            let expected = dense_mm(&da, &db, m, n, 3);
            let mut a = packed(left, &da);
            let mut b = packed(right, &db);
            poison_padding(left, &mut a);
            poison_padding(right, &mut b);
            let mut output = vec![SENTINEL; left.elements() + 3];
            let mut scratch = vec![SENTINEL; 2 * left.elements() + 3];
            optimized_lower_mm(left, &a, right, &b, left, &mut scratch, &mut output, LIMITS)
                .unwrap();
            assert_eq!(unpacked(left, &output), expected);
            assert_padding_and_suffix(left, &output);
            assert!(scratch[..2 * left.elements()].iter().all(|v| v.is_finite()));
            assert!(
                scratch[2 * left.elements()..]
                    .iter()
                    .all(|v| *v == SENTINEL)
            );
        }
    }
}

#[test]
fn optimized_multiblock_preflight_is_atomic_and_work_bound_is_exact() {
    let left = layout(4, 12, 3, 2, Diagonal::Lower);
    // RHS storage capacity need not match left: replication is a fused read.
    let right = layout(12, 12, 3, 5, Diagonal::Lower);
    let a = packed(left, &data(left.dense_elements()));
    let b = packed(right, &data(right.dense_elements()));
    let mut bad_a = a.clone();
    *bad_a.last_mut().unwrap() = f64::NAN;
    let mut bad_b = b.clone();
    bad_b[right.address(2, 11, 11).unwrap()] = f64::INFINITY;
    let mut output = vec![SENTINEL; left.elements() + 3];
    let mut scratch = vec![SENTINEL; 2 * left.elements() + 3];
    let before = output.clone();
    let scratch_before = scratch.clone();
    let required_work = left.elements() + right.elements() + 4 * left.elements() * left.columns();
    let denied = Limits {
        max_work: required_work - 1,
        ..LIMITS
    };
    // Every extent is checked before scans; work denial precedes late NaNs.
    for (av, bv, slen, olen, limits, error) in [
        (
            &bad_a[..],
            &b[..],
            scratch.len(),
            output.len(),
            denied,
            Error::WorkLimit,
        ),
        (
            &a[..],
            &bad_b[..],
            scratch.len(),
            output.len(),
            denied,
            Error::WorkLimit,
        ),
        (
            &bad_a[..],
            &b[..],
            2 * left.elements() - 1,
            output.len(),
            LIMITS,
            Error::Buffer,
        ),
        (
            &bad_a[..],
            &b[..],
            scratch.len(),
            left.elements() - 1,
            LIMITS,
            Error::Buffer,
        ),
        (
            &a[..a.len() - 1],
            &bad_b[..],
            scratch.len(),
            output.len(),
            LIMITS,
            Error::Buffer,
        ),
        (
            &bad_a[..],
            &b[..b.len() - 1],
            scratch.len(),
            output.len(),
            LIMITS,
            Error::Buffer,
        ),
        (
            &bad_a[..],
            &b[..],
            scratch.len(),
            output.len(),
            LIMITS,
            Error::NonFinite,
        ),
        (
            &a[..],
            &bad_b[..],
            scratch.len(),
            output.len(),
            LIMITS,
            Error::NonFinite,
        ),
        (
            &a[..],
            &b[..],
            scratch.len(),
            output.len(),
            Limits {
                max_buffer_elements: scratch.len() - 1,
                ..LIMITS
            },
            Error::Buffer,
        ),
    ] {
        assert_eq!(
            optimized_lower_mm(
                left,
                av,
                right,
                bv,
                left,
                &mut scratch[..slen],
                &mut output[..olen],
                limits
            ),
            Err(error)
        );
        assert_eq!(output, before);
        assert_eq!(scratch, scratch_before);
    }
    // Unsupported partial logical frames and orientation remain geometry errors.
    for unsupported in [
        layout(4, 12, 3, 3, Diagonal::Lower),
        layout(4, 12, 3, 2, Diagonal::Upper),
    ] {
        assert_eq!(optimized_lower_schedule(unsupported), Err(Error::Geometry));
        assert_eq!(
            optimized_lower_mm(
                unsupported,
                &bad_a,
                right,
                &bad_b,
                unsupported,
                &mut scratch,
                &mut output,
                LIMITS
            ),
            Err(Error::Geometry)
        );
        assert_eq!(output, before);
        assert_eq!(scratch, scratch_before);
    }
    optimized_lower_mm(
        left,
        &a,
        right,
        &b,
        left,
        &mut scratch,
        &mut output,
        Limits {
            max_work: required_work,
            ..LIMITS
        },
    )
    .unwrap();
    assert_eq!(
        unpacked(left, &output),
        dense_mm(&unpacked(left, &a), &unpacked(right, &b), 4, 12, 3)
    );
    assert_padding_and_suffix(left, &output);
    assert!(
        scratch[2 * left.elements()..]
            .iter()
            .all(|v| *v == SENTINEL)
    );
}

#[test]
fn specified_complexification_has_nonzero_imaginary_control_and_dense_real_result() {
    let left = layout(8, 4, 2, 1, Diagonal::Upper);
    let right = layout(4, 8, 2, 1, Diagonal::Lower);
    let out = layout(8, 8, 2, 1, Diagonal::Lower);
    let a = data(left.dense_elements());
    let b = data(right.dense_elements());
    let mut c = vec![99.0; out.elements()];
    let mut im = c.clone();
    paired_upper_lower_mm(
        left,
        &packed(left, &a),
        right,
        &packed(right, &b),
        out,
        &mut im,
        &mut c,
        LIMITS,
    )
    .unwrap();
    let (expected, expected_imaginary) = dense_paired_oracle(&a, &b, 8, 4, 2, 8, 4, 8);
    assert_eq!(unpacked(out, &c), expected);
    assert_eq!(unpacked(out, &im), expected_imaginary);
    assert!(im.iter().any(|x| *x != 0.0));
    let counts = paired_upper_lower_schedule(left).unwrap();
    assert_eq!(counts.tensor_products, 16);
    assert_eq!(counts.conjugations, 8);
    assert_eq!(counts.input_rotations, 14);
}

#[test]
fn overflow_short_buffers_invalid_inputs_and_work_reject_before_mutation() {
    assert_eq!(
        Layout::new(usize::MAX, usize::MAX, 0, 0, 2, 1, Diagonal::Lower, LIMITS),
        Err(Error::Overflow)
    );
    assert_eq!(
        Layout::new(3, 4, 3, 4, 1, 1, Diagonal::Lower, LIMITS),
        Err(Error::Geometry)
    );
    let l = layout(2, 2, 1, 2, Diagonal::Lower);
    let a = packed(l, &[1.0, 2.0, 3.0, 4.0]);
    let mut out = vec![999.0; 5];
    let sentinel = out.clone();
    assert_eq!(l.pack(&[1.0], &mut out, LIMITS), Err(Error::Buffer));
    assert_eq!(out, sentinel);
    assert_eq!(
        l.pack(&[1.0, 2.0, 3.0, f64::NAN], &mut out, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(out, sentinel);
    let tiny = Limits {
        max_work: 0,
        ..LIMITS
    };
    assert_eq!(
        lower_mm(l, &a, l, &a, l, &mut out, tiny),
        Err(Error::WorkLimit)
    );
    assert_eq!(out, sentinel);
    assert_eq!(
        lower_mm(l, &a[..3], l, &a, l, &mut out, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    assert_eq!(replicate(l, &a, &mut out, LIMITS), Err(Error::Buffer));
    assert_eq!(out, sentinel);
    assert_eq!(
        transpose(l, &a, l, &mut out[..3], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    let mut scratch = vec![888.0; 7];
    let before = scratch.clone();
    assert_eq!(
        optimized_lower_mm(l, &a, l, &a, l, &mut scratch, &mut out, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(out, sentinel);
    assert_eq!(scratch, before);
    let mut bad = a.clone();
    bad[3] = f64::INFINITY;
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut out, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(out, sentinel);
    let mut with_tail = vec![999.0; 5];
    l.pack(&[1.0; 4], &mut with_tail, LIMITS).unwrap();
    assert_eq!(with_tail[4], 999.0);
    assert_eq!(l.address(1, 0, 0), Err(Error::Geometry));
    assert_eq!(l.lane(l.elements()), Err(Error::Buffer));
}

#[test]
fn all_buffer_apis_check_extents_and_work_before_late_nan_scans() {
    // Construct with generous limits, then deny work at call time. Last active
    // element is NaN: budget must win over the otherwise late NonFinite error.
    let l = layout(4, 4, 2, 4, Diagonal::Lower);
    let upper = layout(4, 4, 2, 1, Diagonal::Upper);
    let single = layout(4, 4, 2, 1, Diagonal::Lower);
    let mut bad = vec![1.0; l.elements()];
    *bad.last_mut().unwrap() = f64::NAN;
    let good = vec![1.0; l.elements()];
    let mut weights = vec![1.0; 64];
    *weights.last_mut().unwrap() = f64::NAN;
    let mut output = vec![SENTINEL; l.diagonals * l.frame_slots + 3];
    let mut scratch = vec![SENTINEL; 2 * l.elements() + 3];
    let before = output.clone();
    let scratch_before = scratch.clone();
    let denied = Limits {
        max_work: 0,
        ..LIMITS
    };
    assert_eq!(l.pack(&bad, &mut output, denied), Err(Error::WorkLimit));
    assert_eq!(l.unpack(&bad, &mut output, denied), Err(Error::WorkLimit));
    assert_eq!(
        transpose(l, &bad, l, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        replicate(l, &bad, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        lower_mm(l, &bad, l, &bad, l, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        block_pc_mm(l, &weights, &bad, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        optimized_lower_mm(l, &bad, l, &bad, l, &mut scratch, &mut output, denied),
        Err(Error::WorkLimit)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &bad,
            single,
            &bad,
            single,
            &mut scratch,
            &mut output,
            denied
        ),
        Err(Error::WorkLimit)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);

    // Once work is admitted, the same late active NaN must be diagnosed without
    // changing the destination (the budget rejection did not mask a bad fixture).
    assert_eq!(l.unpack(&bad, &mut output, LIMITS), Err(Error::NonFinite));
    assert_eq!(
        transpose(l, &bad, l, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(
        replicate(l, &bad, &mut output, LIMITS),
        Err(Error::NonFinite)
    );
    assert_eq!(output, before);

    // Extent checks must also precede *all* scans, including earlier operands.
    let short = l.elements() - 1;
    assert_eq!(
        l.unpack(&bad, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        transpose(l, &bad, l, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        internal_rotate(l, &bad, 1, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        replicate(l, &bad, &mut output[..short], LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        lower_mm(l, &bad, l, &good[..short], l, &mut output, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        block_pc_mm(l, &weights, &good[..short], &mut output, LIMITS),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &bad,
            l,
            &good[..short],
            l,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &bad,
            single,
            &good[..short],
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);
}

#[test]
fn optimized_and_complex_errors_preserve_both_outputs_including_suffixes() {
    let l = layout(4, 4, 2, 4, Diagonal::Lower);
    let upper = layout(4, 4, 2, 1, Diagonal::Upper);
    let single = layout(4, 4, 2, 1, Diagonal::Lower);
    let good = vec![1.0; l.elements()];
    let mut bad = good.clone();
    *bad.last_mut().unwrap() = f64::NAN;
    let mut output = vec![SENTINEL; l.elements() + 3];
    let mut scratch = vec![SENTINEL; 2 * l.elements() + 3];
    let before = output.clone();
    let scratch_before = scratch.clone();
    for (a, b) in [(&bad, &good), (&good, &bad)] {
        assert_eq!(
            optimized_lower_mm(l, a, l, b, l, &mut scratch, &mut output, LIMITS),
            Err(Error::NonFinite)
        );
        assert_eq!(
            paired_upper_lower_mm(
                upper,
                a,
                single,
                b,
                single,
                &mut scratch,
                &mut output,
                LIMITS
            ),
            Err(Error::NonFinite)
        );
        assert_eq!(output, before);
        assert_eq!(scratch, scratch_before);
    }
    let short = l.elements() - 1;
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            l,
            &mut scratch[..2 * l.elements() - 1],
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            l,
            &mut scratch,
            &mut output[..short],
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &good,
            single,
            &good,
            single,
            &mut scratch[..short],
            &mut output,
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        paired_upper_lower_mm(
            upper,
            &good,
            single,
            &good,
            single,
            &mut scratch,
            &mut output[..short],
            LIMITS
        ),
        Err(Error::Buffer)
    );
    assert_eq!(
        optimized_lower_mm(
            l,
            &good,
            l,
            &good,
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Geometry)
    );
    assert_eq!(
        paired_upper_lower_mm(
            l,
            &good,
            single,
            &good,
            single,
            &mut scratch,
            &mut output,
            LIMITS
        ),
        Err(Error::Geometry)
    );
    assert_eq!(output, before);
    assert_eq!(scratch, scratch_before);
}

#[test]
fn transforms_zero_padding_ignore_poison_and_preserve_output_suffixes() {
    for d in [Diagonal::Upper, Diagonal::Lower] {
        let l = Layout::new(4, 8, 3, 5, 2, 3, d, LIMITS).unwrap();
        let target = Layout::new(8, 4, 5, 3, 2, 3, d, LIMITS).unwrap();
        let dense = data(l.dense_elements());
        let mut slots = vec![SENTINEL; l.elements() + 3];
        l.pack(&dense, &mut slots, LIMITS).unwrap();
        assert_padding_and_suffix(l, &slots);
        poison_padding(l, &mut slots);
        let mut restored = vec![SENTINEL; l.dense_elements() + 3];
        l.unpack(&slots, &mut restored, LIMITS).unwrap();
        for h in 0..2 {
            for i in 0..4 {
                for j in 0..8 {
                    assert_eq!(
                        restored[(h * 4 + i) * 8 + j],
                        if i < 3 && j < 5 {
                            dense[(h * 4 + i) * 8 + j]
                        } else {
                            0.0
                        }
                    );
                }
            }
        }
        assert!(
            restored[l.dense_elements()..]
                .iter()
                .all(|v| *v == SENTINEL)
        );
        let mut transposed = vec![SENTINEL; target.elements() + 3];
        transpose(l, &slots, target, &mut transposed, LIMITS).unwrap();
        assert_padding_and_suffix(target, &transposed);
        for h in 0..2 {
            for i in 0..3 {
                for j in 0..5 {
                    assert_eq!(
                        transposed[target.address(h, j, i).unwrap()],
                        dense[(h * 4 + i) * 8 + j]
                    );
                }
            }
        }
        let mut rotated = vec![SENTINEL; l.elements() + 3];
        internal_rotate(l, &slots, 1, &mut rotated, LIMITS).unwrap();
        assert_padding_and_suffix(l, &rotated);
        let required = l.diagonals * l.frame_slots;
        let mut replicated = vec![SENTINEL; required + 3];
        replicate(l, &slots, &mut replicated, LIMITS).unwrap();
        for diag in 0..l.diagonals {
            for copy in 0..l.capacity {
                for t in 0..l.length {
                    for h in 0..l.heads {
                        let (r, c) = l.coordinates(diag, t);
                        assert_eq!(
                            replicated[diag * l.frame_slots + (copy * l.length + t) * l.heads + h],
                            if r < 3 && c < 5 {
                                dense[(h * 4 + r) * 8 + c]
                            } else {
                                0.0
                            }
                        );
                    }
                }
            }
        }
        assert!(replicated[required..].iter().all(|v| *v == SENTINEL));
    }
}

#[test]
fn lower_and_pc_products_zero_padding_and_preserve_suffixes() {
    // Nondividing capacity also exercises padded final frame in the basic path.
    let left = Layout::new(4, 8, 3, 5, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let right = Layout::new(8, 8, 5, 6, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let out = Layout::new(4, 8, 3, 6, 2, 3, Diagonal::Lower, LIMITS).unwrap();
    let mut a = packed(left, &data(left.dense_elements()));
    let mut b = packed(right, &data(right.dense_elements()));
    let expected = dense_mm(&unpacked(left, &a), &unpacked(right, &b), 4, 8, 2);
    poison_padding(left, &mut a);
    poison_padding(right, &mut b);
    let mut c = vec![SENTINEL; out.elements() + 3];
    lower_mm(left, &a, right, &b, out, &mut c, LIMITS).unwrap();
    assert_eq!(unpacked(out, &c), expected);
    assert_padding_and_suffix(out, &c);

    let pc = Layout::new(4, 4, 4, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let raw = data(pc.dense_elements());
    let weights = data(64);
    let mut b = packed(pc, &raw);
    poison_padding(pc, &mut b);
    let mut c = vec![SENTINEL; pc.elements() + 3];
    block_pc_mm(pc, &weights, &b, &mut c, LIMITS).unwrap();
    assert_padding_and_suffix(pc, &c);
    for i in 0..8 {
        for j in 0..3 {
            let expected: f64 = (0..8).map(|k| weights[i * 8 + k] * raw[k * 4 + j]).sum();
            assert_eq!(c[pc.address(i / 4, i % 4, j).unwrap()], expected);
        }
    }
}

#[test]
fn optimized_partial_rectangle_zeroes_output_padding_and_preserves_scratch_suffix() {
    let left = Layout::new(2, 4, 1, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let right = Layout::new(4, 4, 3, 3, 2, 2, Diagonal::Lower, LIMITS).unwrap();
    let mut a = packed(left, &data(left.dense_elements()));
    let mut b = packed(right, &data(right.dense_elements()));
    let expected = dense_mm(&unpacked(left, &a), &unpacked(right, &b), 2, 4, 2);
    poison_padding(left, &mut a);
    poison_padding(right, &mut b);
    let mut c = vec![SENTINEL; left.elements() + 3];
    let mut scratch = vec![SENTINEL; 2 * left.elements() + 3];
    optimized_lower_mm(left, &a, right, &b, left, &mut scratch, &mut c, LIMITS).unwrap();
    assert_eq!(unpacked(left, &c), expected);
    assert_padding_and_suffix(left, &c);
    // Scratch holds unmasked intermediate partial sums, not a matrix layout.
    assert!(
        scratch[2 * left.elements()..]
            .iter()
            .all(|v| *v == SENTINEL)
    );
}

#[test]
fn complex_basis_pairs_and_partial_rectangles_match_both_dense_oracles() {
    for (n, m) in [(4, 2), (4, 4)] {
        for (vr, vk, vc) in [(n, m, n), (n - 1, m - 1, n - 1)] {
            let left = Layout::new(n, m, vr, vk, 2, 1, Diagonal::Upper, LIMITS).unwrap();
            let right = Layout::new(m, n, vk, vc, 2, 1, Diagonal::Lower, LIMITS).unwrap();
            let out = Layout::new(n, n, vr, vc, 2, 1, Diagonal::Lower, LIMITS).unwrap();
            // Includes basis entries outside valid rectangles: these must never
            // contribute to either component, even with poisoned packed padding.
            for ai in 0..left.dense_elements() {
                for bi in 0..right.dense_elements() {
                    let mut da = vec![0.0; left.dense_elements()];
                    da[ai] = 1.0;
                    let mut db = vec![0.0; right.dense_elements()];
                    db[bi] = 1.0;
                    let (expected, expected_im) =
                        dense_paired_oracle(&da, &db, n, m, 2, vr, vk, vc);
                    let mut a = packed(left, &da);
                    let mut b = packed(right, &db);
                    poison_padding(left, &mut a);
                    poison_padding(right, &mut b);
                    let mut c = vec![SENTINEL; out.elements() + 3];
                    let mut im = c.clone();
                    paired_upper_lower_mm(left, &a, right, &b, out, &mut im, &mut c, LIMITS)
                        .unwrap();
                    assert_eq!(unpacked(out, &c), expected);
                    assert_eq!(unpacked(out, &im), expected_im);
                    assert_padding_and_suffix(out, &c);
                    assert_padding_and_suffix(out, &im);
                }
            }
        }
    }
}
