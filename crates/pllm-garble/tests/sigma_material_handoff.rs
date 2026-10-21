//! External typed-ownership checks, not transport or checkpoint qualification.
//! Intermediate values remain opaque here; native unit tests check integer parity.
use pllm_core::edge_contract::{
    EdgeDescriptor, EdgeIdentity, Interpretation, Numeric, Representation, Shape, SourceIdentity,
    WeightVisibility,
};
use pllm_garble::masked_wire_reference::{
    matrix::{MatrixLayout, Spec},
    typed::{
        approved_implementation, ArithmeticParty, BoundRange, Budget, CommonMasked, DealerMask,
        Error, LocalDealer, MatrixParty, MatrixRange, Ring, Step,
    },
};

fn source() -> SourceIdentity {
    SourceIdentity::new([1; 32], [2; 32], approved_implementation())
}

fn edge(id: u8, shape: &[usize], bits: u8, fractional: u8, private: bool) -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [3; 32], [id + 64; 32], [id; 32], 0),
        Shape::new(shape).unwrap(),
        Numeric::new(bits, fractional, Interpretation::Signed).unwrap(),
        Representation::CommonAdditiveMasked,
        private.then_some(WeightVisibility::Private),
    )
    .unwrap()
}

fn predicate() -> EdgeDescriptor {
    EdgeDescriptor::new(
        EdgeIdentity::new(source(), [3; 32], [76; 32], [12; 32], 0),
        Shape::new(&[2, 2]).unwrap(),
        Numeric::new(1, 0, Interpretation::Predicate).unwrap(),
        Representation::CommonXorMasked,
        None,
    )
    .unwrap()
}

struct Fixture<'b> {
    mask: DealerMask<'b, Ring>,
    parties: [MatrixParty<'b>; 2],
    x: [CommonMasked<'b, Ring>; 2],
    w: [CommonMasked<'b, Ring>; 2],
}

fn fixture<'b>(dealer: &mut LocalDealer<'b>, layout: MatrixLayout) -> Fixture<'b> {
    let (x_mask, x) = dealer
        .prepare_root(
            edge(5, &[2, 3], 16, 2, false),
            BoundRange::new(-3, 4).unwrap(),
            [7; 32],
        )
        .unwrap()
        .encode(&[-3, 2, 1, 1, -2, 3])
        .unwrap();
    let (shape, values) = match layout {
        MatrixLayout::InputOutput => ([3, 2], [2, -1, -3, 2, 1, 1]),
        MatrixLayout::OutputInput => ([2, 3], [2, -3, 1, -1, 2, 1]),
    };
    let (w_mask, w) = dealer
        .prepare_root(
            edge(6, &shape, 16, 2, true),
            BoundRange::new(-3, 4).unwrap(),
            [7; 32],
        )
        .unwrap()
        .encode(&values)
        .unwrap();
    let (mask, parties) = dealer
        .matrix(
            [7; 32],
            Spec {
                bits: 16,
                rows: 2,
                inner: 3,
                columns: 2,
                layout,
            },
            x_mask,
            w_mask,
            edge(8, &[2, 2], 16, 4, false),
            [9; 32],
            MatrixRange::RequireSignedNoWrap,
        )
        .unwrap();
    Fixture {
        mask,
        parties,
        x,
        w,
    }
}

fn arithmetic<'b, K>(
    mut parties: [ArithmeticParty<'b, K>; 2],
    [x0, x1]: [CommonMasked<'b, Ring>; 2],
) -> [CommonMasked<'b, K>; 2] {
    let [left, right] = &mut parties;
    let mut a = left.begin(x0).unwrap();
    let mut b = right.begin(x1).unwrap();
    loop {
        match (
            left.advance(b.as_bytes()).unwrap(),
            right.advance(a.as_bytes()).unwrap(),
        ) {
            (Step::Output(x), Step::Output(y)) => return [x, y],
            (Step::Frame(x), Step::Frame(y)) => {
                a = x;
                b = y;
            }
            _ => panic!("party schedules diverged"),
        }
    }
}

// Single test owns this integration binary's process-wide matrix singleton.
#[test]
fn consuming_handoff_preserves_original_typed_chain_and_rejects_substitution() {
    for layout in [MatrixLayout::InputOutput, MatrixLayout::OutputInput] {
        let budget = Budget::new(64 << 20).unwrap();
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [left, right],
                x: [x0, x1],
                w: [w0, w1],
            } = fixture(&mut dealer, layout);
            let a = left.export_material().unwrap();
            let b = right.export_material().unwrap();
            assert_ne!(a.descriptor(), b.descriptor());
            let (permit0, packet0) = a.into_parts();
            let (permit1, packet1) = b.into_parts();
            assert_eq!(&packet0.as_bytes()[..8], b"PLLMSM01");
            assert_eq!(&packet1.as_bytes()[..8], b"PLLMSM01");
            let mut left = permit0.import(packet0).unwrap();
            let mut right = permit1.import(packet1).unwrap();
            // These are the original capabilities, not owners recreated from metadata.
            let a = left.begin(x0, w0, &mut || false).unwrap();
            let b = right.begin(x1, w1, &mut || false).unwrap();
            let outputs = [
                left.finish(b.as_bytes()).unwrap(),
                right.finish(a.as_bytes()).unwrap(),
            ];
            drop((a, b, left, right));
            let (mask, parties) = dealer
                .tr([9; 32], mask, 2, edge(10, &[2, 2], 14, 2, false), [11; 32])
                .unwrap();
            let outputs = arithmetic(parties, outputs);
            let (mask, parties) = dealer
                .gap_ars([11; 32], mask, 1, edge(13, &[2, 2], 14, 1, false), [14; 32])
                .unwrap();
            let outputs = arithmetic(parties, outputs);
            let (mask, parties) = dealer.drelu([14; 32], mask, predicate(), [15; 32]).unwrap();
            let outputs = arithmetic(parties, outputs);
            for output in &outputs {
                assert_eq!(output.descriptor(), predicate());
            }
            drop((mask, outputs));
        }
        assert_eq!(budget.used_bytes(), 0);
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [left, right],
                x,
                w,
            } = fixture(&mut dealer, layout);
            let (permit0, packet0) = left.export_material().unwrap().into_parts();
            let (permit1, packet1) = right.export_material().unwrap().into_parts();
            // Both attempts consume their permits. Sibling retirement is explicit.
            assert_eq!(permit0.import(packet1).err(), Some(Error::Context));
            assert_eq!(permit1.import(packet0).err(), Some(Error::Context));
            drop((mask, x, w));
        }
        assert_eq!(budget.used_bytes(), 0);
        {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [left, mut right],
                x,
                w,
            } = fixture(&mut dealer, layout);
            let (permit, packet) = left.export_material().unwrap().into_parts();
            let imported = permit.import(packet).unwrap();
            assert_eq!(imported.export_material().err(), Some(Error::Consumed));
            right.cancel();
            drop((right, mask, x, w));
        }
        assert_eq!(budget.used_bytes(), 0);
        for wrong_mask in [false, true] {
            let mut dealer = LocalDealer::new(source(), [3; 32], [4; 32], &budget).unwrap();
            let Fixture {
                mask,
                parties: [left, mut right],
                x: [x0, x1],
                w: [w0, w1],
            } = fixture(&mut dealer, layout);
            let (permit, packet) = left.export_material().unwrap().into_parts();
            let mut left = permit.import(packet).unwrap();
            if wrong_mask {
                // Identical edge/shape/value declaration, independently issued mask.
                let (other_mask, [other_x0, other_x1]) = dealer
                    .prepare_root(
                        edge(5, &[2, 3], 16, 2, false),
                        BoundRange::new(-3, 4).unwrap(),
                        [7; 32],
                    )
                    .unwrap()
                    .encode(&[-3, 2, 1, 1, -2, 3])
                    .unwrap();
                assert_eq!(
                    left.begin(other_x0, w0, &mut || false).err(),
                    Some(Error::Context)
                );
                drop((other_mask, other_x1, x0, x1, w1));
            } else {
                assert_eq!(
                    left.begin(x1, w0, &mut || false).err(),
                    Some(Error::Context)
                );
                drop((x0, w1));
            }
            assert_eq!(left.finish(&[]).err(), Some(Error::Consumed));
            right.cancel();
            drop((left, right, mask, dealer));
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}
