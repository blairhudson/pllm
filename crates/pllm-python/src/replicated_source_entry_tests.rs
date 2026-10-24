use super::*;
use pyo3::types::PyList;

#[test]
fn actual_private_native_source_entry_two_accepted_ids_and_original_feedback() {
    let config=br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let mut publisher = Publisher::new_with_policy(
        config,
        4,
        2,
        [31; 32],
        [32; 32],
        WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
    )
    .unwrap();
    let mut files = Vec::new();
    for (key, shape) in publisher.shapes() {
        let n: usize = shape.iter().product();
        let value = if key.contains("norm") { 1f32 } else { 0f32 };
        let bytes = value.to_le_bytes().repeat(n);
        files.push((key.clone(), bytes));
    }
    for (key, bytes) in &files {
        publisher.append(key, 0, bytes).unwrap();
    }
    publisher.seal(&[0, 1, 0, 1]).unwrap();
    let manifest = publisher.manifest().unwrap();
    let mut chunks = Vec::new();
    while let Some(chunk) = publisher.next_chunk().unwrap() {
        chunks.push(chunk);
    }
    Python::initialize();
    let frames = Python::attach(|py| {
        let module = PyModule::new(py, "_source_entry_native_test").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Worker", module.getattr("ReplicatedWorker").unwrap())
            .unwrap();
        locals
            .set_item("manifest", PyBytes::new(py, &manifest))
            .unwrap();
        let data = PyList::empty(py);
        for (key, bytes) in &files {
            data.append((key, PyBytes::new(py, bytes))).unwrap();
        }
        locals.set_item("files", data).unwrap();
        let inputs = PyList::empty(py);
        for chunk in &chunks {
            let roles = PyList::empty(py);
            for bytes in &chunk.shares {
                roles.append(PyBytes::new(py, bytes)).unwrap();
            }
            inputs.append((&chunk.key, chunk.offset, roles)).unwrap();
        }
        locals.set_item("inputs", inputs).unwrap();
        py.run(c"\
workers = [Worker._from_source_request(manifest, role) for role in range(3)]
for worker in workers:
    for key, data in files: worker.append_public_source(key, 0, data)
    worker.seal_public_source()
offers = [worker.call('offer', b'', role)[1][0][1] for role, worker in enumerate(workers)]
for sender, frame in enumerate(offers):
    for role, worker in enumerate(workers):
        if role != sender: worker.call('public', frame, sender)
seeds = [worker.call('seed', b'', role)[1][0] for role, worker in enumerate(workers)]
for sender, (to, frame) in enumerate(seeds): workers[to].call('receive_seed', frame, sender)
for worker in workers: worker.activate()
for key, offset, shares in inputs:
    for role, worker in enumerate(workers):
        k = key.encode()
        wire = len(k).to_bytes(2,'little') + k + offset.to_bytes(4,'little') + (len(shares[role])//16).to_bytes(4,'little') + shares[role]
        worker.call('import', wire, role)
done = [False]*3
queue = []
for step in range(100000):
    if all(done): break
    if queue:
        pending, queue = queue, []
        for sender, to, frame in pending:
            done[to], output = workers[to].call('peer', frame, sender)
            queue.extend((to, dest, response) for dest, response in output)
    else:
        for role, worker in enumerate(workers):
            if not done[role]:
                done[role], output = worker.call('advance', b'', role)
                queue.extend((role, to, frame) for to, frame in output)
assert all(done) and not queue
frames = [worker.finish() for worker in workers]
",Some(&locals),Some(&locals)).unwrap();
        locals
            .get_item("frames")
            .unwrap()
            .unwrap()
            .extract::<Vec<Vec<u8>>>()
            .unwrap()
    });
    assert_eq!(
        publisher
            .finish([&frames[0], &frames[1], &frames[2]])
            .unwrap(),
        vec![0, 0]
    );
}
