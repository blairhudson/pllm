//! Approved D20/D22 integer regression only; Native HOLD, authoring UNRUN.
//!
//! These are LOCAL clear-fixture dealer inputs, not authenticated private source
//! ingress. SourceIdentity below is a declaration, not a checkpoint certificate.
//! Neither D27 role separation nor this test decoder authenticates a publisher,
//! proves checkpoint-wide ranges, or admits Qwen's full 896-wide projection.
//! No product composition, inherited activation ingress or share-to-mask bridge.
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Shape};
use sha2::{Digest, Sha256};

const LIMIT: usize = 64 << 20;
const FROZEN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
));

// Independent equation fixture, NOT fixed_activation_tables or a production
// activation result. Existing reference.py:138-174 certifies every entry using
// rational exp bounds L <= exp(i/64) <= U and equal endpoint floors of
// 64*i/(1+U) and 64*i/(1+L). Reuse that certification, never execute a generator
// or introduce a second certification framework here.
fn corrections() -> Vec<i128> {
    let hex = |bytes: &[u8]| {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
    };
    assert_eq!(
        hex(&Sha256::digest(FROZEN)),
        "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
    );
    let fixture: serde_json::Value = serde_json::from_slice(FROZEN).unwrap();
    let table: Vec<i128> = fixture["tables"]["silu"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| i128::from(value.as_u64().unwrap()))
        .collect();
    assert_eq!(table.len(), 1024);
    let mut digest = Sha256::new();
    for &entry in &table {
        digest.update(u64::try_from(entry).unwrap().to_le_bytes());
    }
    assert_eq!(
        hex(&digest.finalize()),
        "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f"
    );
    assert_eq!([table[0], table[1], table[2], table[64], table[1023]], [0, 31, 63, 1101, 0]);
    table
}

fn source() -> SourceIdentity {
    SourceIdentity::new([11; 32], [12; 32], approved_implementation())
}

fn dealer(budget: &Budget) -> LocalDealer<'_> {
    LocalDealer::new(source(), [1; 32], [2; 32], budget).unwrap()
}

fn declaration(
    rows: usize,
    inner: usize,
    columns: usize,
    layout: matrix::MatrixLayout,
    domains: [(i128, i128); 2],
) -> Result<LinearActivationSpec, Error> {
    let edges = std::array::from_fn(|index| {
        let dimensions = match index {
            0 => [rows, inner],
            1 => match layout {
                matrix::MatrixLayout::InputOutput => [inner, columns],
                matrix::MatrixLayout::OutputInput => [columns, inner],
            },
            _ => [rows, columns],
        };
        EdgeDescriptor::new(
            EdgeIdentity::new(
                source(), [1; 32], [20 + index as u8; 32], [3 + index as u8; 32], 0,
            ),
            Shape::new(&dimensions).unwrap(),
            Numeric::new(50, if index == 2 { 24 } else { 12 }, Interpretation::Signed).unwrap(),
            Representation::CommonAdditiveMasked,
            (index == 1).then_some(WeightVisibility::Private),
        )
        .unwrap()
    });
    LinearActivationSpec::new(
        matrix::Spec { bits: 50, rows, inner, columns, layout },
        edges,
        [[31; 32], [32; 32], [33; 32], [34; 32]],
        ActivationKind::Silu,
        domains,
    )
}

const LAYOUTS: [matrix::MatrixLayout; 2] = [
    matrix::MatrixLayout::InputOutput,
    matrix::MatrixLayout::OutputInput,
];

// Fixture weights arrive in mathematical [inner, columns] order. Transpose
// their storage, not their meaning, for the second genuinely rectangular layout.
fn weights(layout: matrix::MatrixLayout, inner: usize, columns: usize, io: &[i64]) -> Vec<i64> {
    assert_eq!(io.len(), inner * columns);
    match layout {
        matrix::MatrixLayout::InputOutput => io.to_vec(),
        matrix::MatrixLayout::OutputInput => (0..columns)
            .flat_map(|column| (0..inner).map(move |k| io[k * columns + column]))
            .collect(),
    }
}

// shared-profile-resolution.md section 2.1, entirely i128 arithmetic:
// S_rc = sum_k x_rk*w_kc; Z = floor(S/4096); Y = floor(Z/64);
// J = min(abs(Y),1023); A = [Y>=0]*Z - T[J]. No per-tile floor,
// production interval/floor/activation helper, floating point or final shift.
fn silu(z: i128, table: &[i128]) -> i128 {
    let y = z.div_euclid(64);
    let index = y.abs().min(1023) as usize;
    (if y >= 0 { z } else { 0 }) - table[index]
}

fn expected(s: LinearActivationSpec, x: &[i64], w: &[i64], table: &[i128]) -> Vec<i128> {
    let mut result = Vec::new();
    for row in 0..s.matrix.rows {
        for column in 0..s.matrix.columns {
            let mut sum = 0i128;
            for k in 0..s.matrix.inner {
                let wi = match s.matrix.layout {
                    matrix::MatrixLayout::InputOutput => k * s.matrix.columns + column,
                    matrix::MatrixLayout::OutputInput => column * s.matrix.inner + k,
                };
                sum += i128::from(x[row * s.matrix.inner + k]) * i128::from(w[wi]);
            }
            assert!((-(1i128 << 48)..(1i128 << 48)).contains(&sum));
            result.push(silu(sum.div_euclid(4096), table));
        }
    }
    result
}

fn finish<'b>(run: &mut LinearActivationRun<'b>) -> [CommonMasked<'b, Ring>; 2] {
    let mut frames = run.begin(&mut || false).unwrap();
    for exchange in 0..11 {
        let step = run
            .advance([frames[1].as_bytes(), frames[0].as_bytes()], &mut || false)
            .unwrap();
        match step.into_frames() {
            Ok(next) => {
                assert!(exchange < 10);
                frames = next;
            }
            Err(step) => {
                assert_eq!(exchange, 10);
                return step.into_output().ok().unwrap();
            }
        }
    }
    panic!("missing eleven-exchange D22 output")
}

// Private unit-test observation of ACTUAL outputs only. Expected values above
// never inspect these masks/backings; no decoder or authority becomes public.
fn assert_output(mask: &DealerMask<'_, Ring>, output: &[CommonMasked<'_, Ring>; 2], want: &[i128]) {
    for (role, party) in output.iter().enumerate() {
        assert_eq!(party.role, role as u8);
        assert_eq!(party.binding, mask.binding);
        assert_eq!(party.backing.values().len(), want.len());
        let actual: Vec<i128> = party.backing.values().iter()
            .zip(mask.backing.values())
            .map(|(&common, &mask)| {
                let word = i128::from(common.wrapping_sub(mask) & ((1u64 << 50) - 1));
                if word >= 1i128 << 49 { word - (1i128 << 50) } else { word }
            })
            .collect();
        assert_eq!(actual, want);
    }
}

fn check(s: LinearActivationSpec, x: &[i64], w: &[i64], table: &[i128]) {
    let want = expected(s, x, w, table);
    // Independent fresh native material, including both full-ring mask endpoints.
    // Deliberately local fixtures, NOT production entropy or a privacy argument.
    for endpoint in [0, 255] {
        let budget = Budget::new(LIMIT).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let (mask, mut run) = d.linear_activation_from_values_with(s, x, w, &mut |bytes| {
            bytes.fill(endpoint);
            Ok(())
        }).unwrap();
        let output = finish(&mut run);
        assert!(matches!(run.begin(&mut || false), Err(Error::Consumed)));
        assert!(matches!(run.advance([&[], &[]], &mut || false), Err(Error::Consumed)));
        assert_output(&mask, &output, &want);
        drop((run, mask));
        assert!(budget.used_bytes() > baseline); // real output owners survive run
        drop(output);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn silu_complete_nonmultiple_dot_not_per_tile_floor_and_zero_both_layouts() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let table = corrections();
    assert_eq!((-262142i128).div_euclid(4096), -64);
    assert_eq!((-262141i128).div_euclid(4096) + (-1i128).div_euclid(4096), -65);
    assert_eq!((silu(-64, &table), silu(-65, &table)), (-31, -63));
    assert_eq!((silu(64, &table), silu(63, &table)), (33, 63));
    let x = [-1, -1, 0, 0, 1, 1];
    let io = [262141, 262143, -262141, 1, 1, -1];
    for layout in LAYOUTS {
        let s = declaration(3, 2, 3, layout, [(-1, 2), (-262143, 262144)]).unwrap();
        let w = weights(layout, 2, 3, &io);
        assert_eq!(expected(s, &x, &w, &table), [-31, -31, 63, 0, 0, 0, 63, 33, -31]);
        check(s, &x, &w, &table);
    }
}

#[test]
fn silu_tr6_sign_original_z_and_clipping_transitions_after_nonmultiple_dot() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let table = corrections();
    let z = [
        -65537, -65536, -65535, -65473, -65472, -65471,
        -4097, -4096, -4095, -129, -128, -127, -65, -64, -63, -1,
        0, 1, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097,
        65471, 65472, 65473, 65535, 65536, 65537,
    ];
    assert_eq!((silu(-1, &table), silu(0, &table), silu(1, &table)), (-31, 0, 1));
    assert_eq!((silu(4096, &table), silu(-4096, &table)), (2995, -1101));
    // abs(floor(Z/64)), NOT floor(abs(Z)/64), at the negative edge.
    assert_eq!((-65473i128).div_euclid(64).abs(), 1024);
    assert_eq!((-65472i128).div_euclid(64).abs(), 1023);
    assert_eq!(65535i128.div_euclid(64), 1023);
    assert_eq!(65536i128.div_euclid(64), 1024);
    let x: Vec<i64> = z.iter().flat_map(|&value| [value, 1]).collect();
    for layout in LAYOUTS {
        let s = declaration(z.len(), 2, 1, layout, [(-65538, 65538), (0, 4097)]).unwrap();
        let w = weights(layout, 2, 1, &[4096, 1]);
        // Every complete dot is 4096*Z+1, including negative nonmultiples.
        let want: Vec<i128> = z.iter().map(|&value| silu(i128::from(value), &table)).collect();
        assert_eq!(expected(s, &x, &w, &table), want);
        check(s, &x, &w, &table);
    }
}

#[test]
fn silu_current_256_element_caps_maximum_complete_work_and_inner_both_layouts() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let table = corrections();
    // All three pairwise products <=256 imply (rows*inner*columns)^2
    // <=256^3, hence at most 4096 mathematical MACs; 16^3 attains it.
    // Native complete work is 6x that (dealer + both parties), or 24576.
    // Its current 2^20 ceiling is consequently slack, not an adjustable grant.
    // 1x256x1
    // separately attains the largest possible complete inner dimension.
    for (rows, inner, columns) in [(16, 16, 16), (1, 256, 1)] {
        let x: Vec<i64> = (0..rows).flat_map(|row| {
            (0..inner).map(move |k| (((row * 5 + k * 3) % 17) as i64 - 8) * 31)
        }).collect();
        let io: Vec<i64> = (0..inner).flat_map(|k| {
            (0..columns).map(move |column| (((k * 7 + column * 11) % 19) as i64 - 9) * 127)
        }).collect();
        for layout in LAYOUTS {
            let s = declaration(rows, inner, columns, layout, [(-256, 257), (-8192, 8193)]).unwrap();
            let q = matrix::party::resources(s.matrix).unwrap();
            assert_eq!(q.dealer_macs, rows * inner * columns);
            assert_eq!(q.party_macs, [2 * rows * inner * columns, 3 * rows * inner * columns]);
            assert_eq!(q.complete_macs, 6 * rows * inner * columns);
            assert_eq!(q.input_elements, rows * inner);
            assert_eq!(q.weight_elements, inner * columns);
            assert_eq!(q.output_elements, rows * columns);
            check(s, &x, &weights(layout, inner, columns, &io), &table);
        }
    }
    for layout in LAYOUTS {
        for (rows, inner, columns) in [(1, 257, 1), (257, 1, 1), (1, 1, 257)] {
            assert!(matches!(
                declaration(rows, inner, columns, layout, [(-1, 2); 2]),
                Err(Error::Resource)
            ));
        }
    }
    // No caller-controlled work grant exists on this current local owner API.
    // Do not invent one: byte grants below are not claims of work admission.
}

#[test]
fn silu_exact_one_short_original_budget_precedes_entropy_and_preserves_outputs() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let table = corrections();
    let s = declaration(1, 2, 1, LAYOUTS[0], [(-1, 2), (0, 262144)]).unwrap();
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    let want = expected(s, &[-1, -1], &[262141, 1], &table);
    for short in [true, false] {
        let budget = Budget::new(LIMIT).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let filler = budget.reserve(LIMIT - baseline - native + usize::from(short)).unwrap();
        let before = budget.used_bytes();
        let counter = d.counter;
        let mut draws = 0;
        let result = d.linear_activation_from_values_with(s, &[-1, -1], &[262141, 1], &mut |bytes| {
            draws += 1;
            bytes.fill(173);
            Ok(())
        });
        if short {
            assert!(matches!(result, Err(Error::Resource)));
            assert_eq!(draws, 0);
            assert_eq!(d.counter, counter);
            assert_eq!(budget.used_bytes(), before);
        } else {
            let (mask, mut run) = result.unwrap();
            assert!(draws > 0);
            assert_eq!(budget.admitted_peak_bytes(), LIMIT);
            let output = finish(&mut run);
            drop(run);
            assert_output(&mask, &output, &want);
            drop((mask, output));
            assert_eq!(budget.used_bytes(), before);
        }
        drop(filler);
        assert_eq!(budget.used_bytes(), baseline);
    }
}

#[test]
fn silu_reserved_exact_short_foreign_and_post_transfer_fault_ownership() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let table = corrections();
    let s = declaration(1, 2, 1, LAYOUTS[0], [(-1, 2), (0, 262144)]).unwrap();
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    for case in 0..4 {
        let budget = Budget::new(LIMIT).unwrap();
        let foreign = Budget::new(LIMIT).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let owner = if case == 1 { &foreign } else { &budget };
        let payload = native - usize::from(case == 0);
        let mut charge = owner.charge_external(payload).unwrap();
        let before = owner.used_bytes();
        let counter = d.counter;
        // With no free capacity, exact success must move existing charge slices,
        // not reserve a second operation peak or refund/reacquire the grant.
        let filler = if case == 2 { Some(budget.reserve(LIMIT - budget.used_bytes()).unwrap()) } else { None };
        let mut draws = 0;
        let result = d.linear_activation_from_reserved_values_with(
            s, &[-1, -1], &[262141, 1], &mut charge, &mut |bytes| {
                draws += 1;
                if case == 3 { return Err(Error::Randomness); }
                bytes.fill(173);
                Ok(())
            },
        );
        match case {
            0 | 1 => {
                assert!(matches!(result, Err(Error::Resource)));
                assert_eq!(draws, 0);
                assert_eq!(d.counter, counter);
                assert_eq!(charge.payload_bytes(), payload);
                assert_eq!(owner.used_bytes(), before);
            }
            2 => {
                let (mask, mut run) = result.unwrap();
                assert_eq!(charge.payload_bytes(), 0);
                let output = finish(&mut run);
                assert_output(&mask, &output, &expected(s, &[-1, -1], &[262141, 1], &table));
                drop((run, mask, output));
            }
            3 => {
                assert!(matches!(result, Err(Error::Randomness)));
                assert_eq!(draws, 1);
                assert_eq!(charge.payload_bytes(), 0); // transferred, not refunded
                assert_eq!(budget.used_bytes(), baseline + charge.charged_bytes());
                drop(matrix::party::IssuanceTicket::acquire().unwrap());
            }
            _ => unreachable!(),
        }
        drop((filler, charge));
        assert_eq!(budget.used_bytes(), baseline);
        assert_eq!(foreign.used_bytes(), 0);
    }
}

#[test]
fn silu_peer_failure_and_cancellation_burn_both_keep_original_frame_until_last_drop() {
    let _serial = matrix::TEST_SERIAL.lock().unwrap();
    let s = declaration(1, 2, 1, LAYOUTS[0], [(-1, 2), (0, 262144)]).unwrap();
    let native = linear_activation_resources(s).unwrap().reservation_bytes();
    for cancel in [false, true] {
        let budget = Budget::new(LIMIT).unwrap();
        let mut d = dealer(&budget);
        let baseline = budget.used_bytes();
        let mut charge = budget.charge_external(native + 13).unwrap();
        let (mask, mut run) = d.linear_activation_from_reserved_values_with(
            s, &[-1, -1], &[262141, 1], &mut charge, &mut |bytes| {
                bytes.fill(173);
                Ok(())
            },
        ).unwrap();
        let [retained, peer] = run.begin(&mut || false).unwrap();
        if cancel {
            run.cancel();
        } else {
            // Left party advances successfully before right's malformed peer.
            assert!(run.advance([peer.as_bytes(), &[]], &mut || false).is_err());
        }
        assert!(matches!(run.begin(&mut || false), Err(Error::Consumed)));
        assert!(matches!(run.advance([&[], &[]], &mut || false), Err(Error::Consumed)));
        assert_eq!(charge.payload_bytes(), 13);
        drop((run, mask, peer));
        assert!(budget.used_bytes() > baseline + charge.charged_bytes());
        assert!(matches!(matrix::party::IssuanceTicket::acquire(), Err(Error::Resource)));
        drop(retained);
        assert_eq!(budget.used_bytes(), baseline + charge.charged_bytes());
        drop(matrix::party::IssuanceTicket::acquire().unwrap());
        drop(charge);
        assert_eq!(budget.used_bytes(), baseline);
    }
}
