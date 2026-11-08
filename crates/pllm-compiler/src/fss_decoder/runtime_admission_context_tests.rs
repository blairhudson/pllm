use super::*;

#[test]
fn opaque_constructor_failure_runs_once_and_preserves_budget_and_counters() {
    let tensor = Tensor::from_ieee("weight", &[2, 2], WeightDtype::F32, &[0; 4]).unwrap();
    let program = Program {
        online: OnlineSchedule::TensorV1,
        source: Source::new(vec![tensor]).unwrap(),
        nodes: Vec::new(), initial: 1, selections: 1, sequence: 1, vocabulary: 2,
        digest: [71; 32], obligations: [0; 32],
        resources: Resources { metadata_bytes: 4096, live_array_bytes_per_party: 4096, ..Resources::default() },
        limits: Limits::default(), energy: None,
    };
    let budget = request_budget(&program).unwrap();
    let tables = request_tables(&program, &budget).unwrap();
    let mut state = Schedule::new(&program, Context {
        plan: program.digest, session: [1; 32], operation: [2; 32], first_tensor_index: 0,
    }, 0, None, &tables).unwrap();
    state.node = 3;
    let before = budget.used_bytes();
    let counters = format!("{:?}", state.resources);
    let calls = std::cell::Cell::new(0);
    let failure = state.kernel_context(&[0], 1, |_| {
        calls.set(calls.get() + 1);
        Err(Error::Resource)
    }).unwrap_err();
    assert_eq!(calls.get(), 1);
    assert_eq!(failure.error, Error::Resource);
    let context = failure.context.unwrap();
    assert_eq!(context.gate, AdmissionGate::KernelConstruction);
    assert_eq!(context.node, Some(3));
    assert_eq!((context.kernel, context.required, context.used, context.limit), (None, None, None, None));
    assert_eq!(budget.used_bytes(), before);
    assert_eq!(format!("{:?}", state.resources), counters);
    assert_eq!(state.kernel(&[0], 1, |_| {
        calls.set(calls.get() + 1);
        Err(Error::Resource)
    }).unwrap_err(), Error::Resource);
    assert_eq!(calls.get(), 2);
    drop(state);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}
