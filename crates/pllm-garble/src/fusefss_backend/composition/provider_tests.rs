use super::*;
use std::sync::atomic::Ordering;

#[test]
fn byte_limit_refuses_before_factory_without_mutating_admitted_prefix() {
    let required = ReplyWindow::<usize>::reservation_bytes().unwrap();
    let budget = Budget::new(required).unwrap();
    let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
    window.push(0, MATERIAL_WINDOW_BYTES).unwrap();
    let pointer = window.replies[0].bytes.as_ptr();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert_eq!(window.push(1, 1), Err(Error::Resource));
    assert_eq!(window.requests(), &[0]);
    assert_eq!(window.replies[0].bytes.capacity(), MATERIAL_WINDOW_BYTES);
    assert_eq!(window.replies[0].bytes.as_ptr(), pointer);
    assert_eq!(budget.used_bytes(), required);
    drop(window);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn scalar_reply_preserves_one_write_bound_and_burn() {
    Reply::with_bound(4, |reply| {
        assert_eq!(reply.bound(), 4);
        assert_eq!(reply.write(&[1, 2, 3, 4, 5]), Err(Error::Resource));
        assert_eq!(reply.write(&[1]), Err(Error::Consumed));
        assert!(reply.bytes().is_err());
        Ok(())
    })
    .unwrap();
    Reply::with_bound(4, |reply| {
        reply.write(&[1])?;
        assert_eq!(reply.bytes()?, &[1]);
        assert_eq!(reply.write(&[2]), Err(Error::Consumed));
        assert!(reply.bytes().is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn complete_quote_precedes_factories_and_same_counter_never_refunds_source() {
    let required = ReplyWindow::<usize>::reservation_bytes().unwrap();
    let short = Budget::new(required - 1).unwrap();
    short.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        ReplyWindow::new(&short, 0usize),
        Err(Error::Resource)
    ));
    assert_eq!(short.used_bytes(), 0);
    let budget = Budget::new(required + 17).unwrap();
    let source = budget.reserve(17).unwrap();
    let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
    assert_eq!(budget.used_bytes(), required + 17);
    for index in 0..MATERIAL_WINDOW_RECORDS {
        window
            .push(index, MATERIAL_WINDOW_BYTES / MATERIAL_WINDOW_RECORDS)
            .unwrap();
        assert_eq!(budget.used_bytes(), required + 17);
    }
    assert!(window.push(9, 1).is_err());
    drop(window);
    assert_eq!(budget.used_bytes(), 17);
    let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
    for index in 0..MATERIAL_WINDOW_RECORDS {
        window
            .push(index, MATERIAL_WINDOW_BYTES / MATERIAL_WINDOW_RECORDS)
            .unwrap();
    }
    window
        .provide(|requests, replies| {
            assert_eq!(requests, &(0..MATERIAL_WINDOW_RECORDS).collect::<Vec<_>>());
            for reply in replies {
                reply.write(&[17])?;
            }
            assert_eq!(budget.used_bytes(), required + 17);
            Ok(())
        })
        .unwrap();
    assert_eq!(window.replies.capacity(), MATERIAL_WINDOW_RECORDS);
    assert_eq!(
        window
            .replies
            .iter()
            .map(|r| r.bytes.capacity())
            .sum::<usize>(),
        MATERIAL_WINDOW_BYTES
    );
    drop(window);
    assert_eq!(budget.used_bytes(), 17);
    drop(source);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn all_writes_validate_before_import_and_window_retains_original_pointers() {
    for count in [1, 3, MATERIAL_WINDOW_RECORDS] {
        let budget = Budget::new(ReplyWindow::<usize>::reservation_bytes().unwrap()).unwrap();
        let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
        for i in 0..count {
            window.push(i, 16).unwrap();
        }
        let pointers: Vec<_> = window.replies.iter().map(|r| r.bytes.as_ptr()).collect();
        window
            .provide(|_, replies| {
                for (i, reply) in replies.iter_mut().enumerate() {
                    reply.write(&[i as u8])?;
                }
                Ok(())
            })
            .unwrap();
        for (i, ptr) in pointers.into_iter().enumerate() {
            assert_eq!(window.bytes(i).unwrap().as_ptr(), ptr);
            assert_eq!(window.bytes(i).unwrap(), &[i as u8]);
        }
        drop(window);
        assert_eq!(budget.used_bytes(), 0);
    }
    let budget = Budget::new(ReplyWindow::<usize>::reservation_bytes().unwrap()).unwrap();
    let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
    window.push(0, 8).unwrap();
    window.push(1, 8).unwrap();
    assert_eq!(
        window.provide(|_, replies| { replies[0].write(&[1]) }),
        Err(Error::Consumed)
    );
    assert!(window.bytes(0).is_err());
    assert_eq!(window.push(2, 8), Err(Error::Consumed));
    assert_eq!(
        window.provide(|_, _| panic!("burned window must not call provider")),
        Err(Error::Consumed)
    );
    drop(window);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn provider_error_panic_and_every_allocation_failure_retire_complete_window() {
    let required = ReplyWindow::<usize>::reservation_bytes().unwrap();
    for panic in [false, true] {
        let budget = Budget::new(required).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut window = ReplyWindow::new(&budget, 0usize).unwrap();
            window.push(0, 32).unwrap();
            window.push(1, 32).unwrap();
            window.provide(|_, replies| {
                replies[0].write(&[1, 2, 3])?;
                assert_eq!(budget.used_bytes(), required);
                if panic {
                    std::panic::panic_any(12345u64);
                }
                Err(Error::Consumed)
            })
        }));
        if panic {
            assert_eq!(*result.unwrap_err().downcast::<u64>().unwrap(), 12345);
        } else {
            assert_eq!(result.unwrap(), Err(Error::Consumed));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
    // Vec backing + eight exact Reply factories + temporary guarded-view array.
    for fail in 0..=MATERIAL_WINDOW_RECORDS + 2 {
        let budget = Budget::new(required).unwrap();
        budget.fail_after.store(fail as isize, Ordering::SeqCst);
        let result = (|| {
            let mut window = ReplyWindow::new(&budget, 0usize)?;
            for i in 0..MATERIAL_WINDOW_RECORDS {
                window.push(i, 16)?;
            }
            window.provide(|_, replies| {
                for reply in replies {
                    reply.write(&[1])?;
                }
                Ok(())
            })
        })();
        if fail < MATERIAL_WINDOW_RECORDS + 2 {
            assert_eq!(result, Err(Error::Allocation));
        } else {
            assert_eq!(result, Ok(()));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}
