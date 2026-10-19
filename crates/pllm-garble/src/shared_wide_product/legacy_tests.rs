use super::*;

fn context() -> Context {
    Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        first_tensor_index: 7,
    }
}

#[test]
fn unchanged_public_sizes_wire_layout_and_role_checks() {
    assert_eq!(HEADER, 78);
    assert_eq!(FRAME_BYTES, 94);
    assert_eq!(KEY_BYTES, 24);
    let [mut a, mut b] = issue(context()).unwrap();
    assert_eq!(a.key_bytes(), 24);
    assert_eq!(b.key_bytes(), 24);
    let f = a.start(u64::MAX, 1u64 << 63).unwrap();
    let g = b.start(1, 1u64 << 63).unwrap();
    assert_eq!(f.len(), 94);
    assert_eq!(g.len(), 94);
    assert_eq!(&f[..8], b"PLLMMU64");
    assert_eq!(&f[8..40], &g[8..40]);
    assert_eq!(&f[40..72], &g[40..72]);
    assert_eq!(&f[72..78], &[0, 0, 1, 0, 0, 0]);
    assert_eq!(&g[72..78], &[0, 1, 1, 0, 0, 0]);
    assert!(matches!(a.finish(&f), Err(Error::Context)));
    assert_eq!(a.key_bytes(), 0);
    b.cancel();
    assert_eq!(b.key_bytes(), 0);
}

#[test]
fn malformed_foreign_context_nonce_order_replay_cancel_burns() {
    for field in [0, 8, 40, 72, 73, 74, 93, 94] {
        let [mut a, mut b] = issue(context()).unwrap();
        let _ = a.start(7, 11).unwrap();
        let mut frame = b.start(13, 17).unwrap();
        match field {
            93 => {
                frame.pop();
            }
            94 => frame.push(0),
            _ => frame[field] ^= 1,
        }
        assert!(matches!(a.finish(&frame), Err(Error::Context)));
        assert_eq!(a.key_bytes(), 0);
        assert!(matches!(a.finish(&frame), Err(Error::Consumed)));
    }
    let [mut a, mut b] = issue(context()).unwrap();
    let [_, mut foreign] = issue(context()).unwrap();
    let _ = a.start(1, 2).unwrap();
    let f = foreign.start(3, 4).unwrap();
    assert!(matches!(a.finish(&f), Err(Error::Context)));
    assert_eq!(a.key_bytes(), 0);
    assert!(matches!(b.finish(&[]), Err(Error::Context)));
    assert_eq!(b.key_bytes(), 0);
    let [mut a, mut b] = issue(context()).unwrap();
    let _ = a.start(1, 2).unwrap();
    assert!(matches!(a.start(1, 2), Err(Error::Context)));
    assert_eq!(a.key_bytes(), 0);
    let _ = b.start(1, 2).unwrap();
    b.cancel();
    assert_eq!(b.key_bytes(), 0);
    assert!(matches!(b.finish(&[]), Err(Error::Consumed)));
    let [mut a, mut b] = issue(context()).unwrap();
    let f = a.start(1, 2).unwrap();
    let g = b.start(3, 4).unwrap();
    assert_eq!(
        a.finish(&g).unwrap().wrapping_add(b.finish(&f).unwrap()),
        24
    );
    assert!(matches!(a.finish(&g), Err(Error::Consumed)));
    assert!(matches!(b.start(0, 0), Err(Error::Consumed)));
}
