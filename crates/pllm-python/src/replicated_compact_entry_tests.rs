//! Private native envelope boundaries shared by source-owned storage routes.
//! These are NOT compact execution qualification. Positive compact execution
//! requires the compiler's real server dispatch and client publisher factory.
use super::*;

#[test]
fn actual_native_envelopes_remain_send_sync() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<NativePublisher>();
    require_send_sync::<NativeWorker>();
}

#[test]
fn native_source_entry_does_not_accept_client_storage_or_public_constructor_policy() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_source_storage_boundary_test").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Worker", module.getattr("ReplicatedWorker").unwrap())
            .unwrap();
        // Signature rejection precedes compiler parsing: no manifest, config,
        // source receipt, publisher or execution authority can be created here.
        py.run(
            c"
for keyword in ({'storage': 'compact'}, {'compact': True}, {'selection': 'compact'}):
    try:
        Worker._from_source_request(b'not a manifest', 0, **keyword)
    except TypeError:
        pass
    else:
        raise AssertionError('client storage keyword became native authority')
try:
    Worker(b'not a manifest', 0, server_policy=(b'{}', b'{}'))
except TypeError:
    pass
else:
    raise AssertionError('ordinary constructor acquired server-policy entry')
try:
    Worker._from_source_request(b'not a manifest', 3)
except ValueError as exc:
    assert 'party outside 0..3' in str(exc)
else:
    raise AssertionError('foreign role accepted before source issuance')
",
            None,
            Some(&locals),
        )
        .unwrap();
    });
}

struct SourceFixture {
    config: Vec<u8>,
    manifest: Vec<u8>,
    coefficients: Vec<(String, Vec<u8>)>,
}

fn expanded_source_fixture() -> SourceFixture {
    // Original expanded Publisher is ONLY a genuine bounded wire/config fixture
    // for the unchanged source entry. It does not issue compact authority or
    // supply a compact execution oracle.
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
    let mut publisher = Publisher::new_with_policy(
        config,
        4,
        2,
        [61; 32],
        [62; 32],
        WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
    )
    .unwrap();
    let coefficients = publisher
        .shapes()
        .iter()
        .map(|(key, shape)| {
            let n: usize = shape.iter().product();
            let value = if key.contains("norm") { 1f32 } else { 0f32 };
            (key.clone(), value.to_le_bytes().repeat(n))
        })
        .collect::<Vec<_>>();
    for (key, bytes) in &coefficients {
        publisher.append(key, 0, bytes).unwrap();
    }
    publisher.seal(&[0, 1, 0, 1]).unwrap();
    let manifest = publisher.manifest().unwrap();
    SourceFixture {
        config: config.to_vec(),
        manifest,
        coefficients,
    }
}

#[test]
fn native_server_policy_rejects_client_storage_foreign_config_and_unsupported_compact() {
    let fixture = expanded_source_fixture();
    // Same concrete admitted server document shape as E's source-header tests.
    // This is trusted test SERVER configuration, not a client Manifest selector.
    let experiment = serde_json::json!({
        "schema":"pllm.experiment.v1", "name":"native-server-storage-boundary",
        "pipeline":{"model":{"source":"original-tiny-source"},"components":{
            "linear":{"component":pllm_compiler::replicated_decoder::sdk::PROTOCOL,
            "params":{"weights":"public","range_policy":"checked",
            "max_abs_activation":1,"storage":"compact"}}
        }},
        "deployment":{"kind":"local","root":"."},
        "budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}
    });
    let mut attempts = Vec::new();
    for (key, value) in [
        ("storage", "unknown"),
        ("weights", "private"),
        ("range_policy", "static"),
        ("lookup", "tiled"),
        ("selection", "tiled"),
        ("peer_transport", "streams"),
    ] {
        let mut changed = experiment.clone();
        changed["pipeline"]["components"]["linear"]["params"][key] = value.into();
        attempts.push((
            fixture.manifest.clone(),
            serde_json::to_vec(&changed).unwrap(),
            fixture.config.clone(),
        ));
    }
    let experiment = serde_json::to_vec(&experiment).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&fixture.manifest).unwrap();
    manifest["storage"] = "compact".into();
    attempts.push((
        serde_json::to_vec(&manifest).unwrap(),
        experiment.clone(),
        fixture.config.clone(),
    ));
    let mut config: serde_json::Value = serde_json::from_slice(&fixture.config).unwrap();
    config["vocab_size"] = 4.into();
    attempts.push((
        fixture.manifest.clone(),
        experiment.clone(),
        serde_json::to_vec(&config).unwrap(),
    ));
    let mut manifest: serde_json::Value = serde_json::from_slice(&fixture.manifest).unwrap();
    manifest["outputs"] = 1.into();
    attempts.push((
        serde_json::to_vec(&manifest).unwrap(),
        experiment,
        fixture.config,
    ));
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_source_server_policy_boundary_test").unwrap();
        register(&module).unwrap();
        let worker = module.getattr("ReplicatedWorker").unwrap();
        for (manifest, experiment, config) in attempts {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "server_policy",
                    (PyBytes::new(py, &experiment), PyBytes::new(py, &config)),
                )
                .unwrap();
            let result = worker.call_method(
                "_from_source_request",
                (PyBytes::new(py, &manifest), 0),
                Some(&kwargs),
            );
            let error =
                result.expect_err("unsupported compact request must not become an expanded worker");
            assert!(
                error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "{error}"
            );
        }
    });
}

#[test]
fn source_import_argument_errors_consume_existing_native_envelope_before_conversion() {
    let SourceFixture { manifest, .. } = expanded_source_fixture();
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_source_envelope_conversion_test").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Worker", module.getattr("ReplicatedWorker").unwrap())
            .unwrap();
        locals
            .set_item("manifest", PyBytes::new(py, &manifest))
            .unwrap();
        py.run(
            c"
def assert_retired(worker):
    try:
        worker.append_public_source('unused', 0, b'')
    except ValueError as exc:
        assert 'worker retired' in str(exc)
    else:
        raise AssertionError('failed operation left native source owner reusable')

# Each attempt gets a fresh genuine import owner. Conversion errors must burn
# that owner even when no compiler append/seal operation has begun.
bad_appends = (
    (object(), 0, b''),
    ('unused', True, b''),
    ('unused', 0, bytearray()),
)
for args in bad_appends:
    worker = Worker._from_source_request(manifest, 0)
    try:
        worker.append_public_source(*args)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError('malformed append accepted')
    assert_retired(worker)
worker = Worker._from_source_request(manifest, 0)
try:
    worker.seal_public_source(untrusted=True)
except TypeError:
    pass
else:
    raise AssertionError('unexpected seal argument accepted')
assert_retired(worker)
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[test]
fn native_worker_storage_witness_is_unsealed_then_actual_expanded_then_retired() {
    let fixture = expanded_source_fixture();
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_source_storage_witness_test").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Worker", module.getattr("ReplicatedWorker").unwrap())
            .unwrap();
        locals
            .set_item("manifest", PyBytes::new(py, &fixture.manifest))
            .unwrap();
        let files = pyo3::types::PyList::empty(py);
        for (key, bytes) in &fixture.coefficients {
            files.append((key, PyBytes::new(py, bytes))).unwrap();
        }
        locals.set_item("files", files).unwrap();
        py.run(
            c"
worker = Worker._from_source_request(manifest, 0)
assert worker._storage_kind is None  # importer is not yet an executable owner
try:
    worker._storage_kind = 'compact'
except AttributeError:
    pass
else:
    raise AssertionError('storage observation is writable')
for key, data in files:
    worker.append_public_source(key, 0, data)
    assert worker._storage_kind is None
worker.seal_public_source()
# Genuine old source construction: this guards against a universal compact
# answer. Compact success needs the real R publisher/E server path separately.
assert worker._storage_kind == 'expanded'
assert worker._storage_kind == 'expanded'  # readonly observation is nonconsuming
worker.close()
try:
    worker._storage_kind
except ValueError as exc:
    assert 'worker retired' in str(exc)
else:
    raise AssertionError('retired owner echoed stale storage intent')
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

fn compact_experiment() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "schema":"pllm.experiment.v1","name":"native-compact-owner",
        "pipeline":{"model":{"source":"original-tiny-source"},"components":{
            "linear":{"component":pllm_compiler::replicated_decoder::sdk::PROTOCOL,
                "params":{"weights":"public","range_policy":"checked",
                    "max_abs_activation":1,"storage":"compact"}}
        }},"deployment":{"kind":"local","root":"."},
        "budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}
    }))
    .unwrap()
}

#[test]
fn compact_conversion_guard_preserves_primary_python_exception_identity() {
    let fixture = expanded_source_fixture();
    let mut owner = Publisher::prepare_compact_source_request(
        &compact_experiment(), &fixture.config, 4, 2, [61; 32], [62; 32],
    ).unwrap();
    Python::initialize();
    Python::attach(|py| {
        let primary = pyo3::exceptions::PyTypeError::new_err("original conversion failure");
        let original = primary.value(py).clone().unbind();
        let returned = compact_conversion::<()>(&mut owner, Err(primary)).unwrap_err();
        assert!(returned.value(py).is(original.bind(py)));
        assert!(returned.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        assert_eq!(returned.value(py).str().unwrap().to_str().unwrap(), "original conversion failure");
    });
    assert!(owner.has_retained_compact());
    assert_eq!(owner.compact_start().unwrap_err(), "compact acquisition already attempted");
    owner.compact_retire().unwrap();
}

#[test]
fn actual_compact_publisher_and_three_workers_witness_same_sealed_storage() {
    let fixture = expanded_source_fixture();
    let experiment = compact_experiment();
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_matched_compact_native_owners").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Publisher", module.getattr("ReplicatedPublisher").unwrap())
            .unwrap();
        locals
            .set_item("Worker", module.getattr("ReplicatedWorker").unwrap())
            .unwrap();
        locals
            .set_item("experiment", PyBytes::new(py, &experiment))
            .unwrap();
        locals
            .set_item("config", PyBytes::new(py, &fixture.config))
            .unwrap();
        py.run(c"
p = Publisher._from_compact_source_request(experiment, config, 4, 2, bytes([61])*32, bytes([62])*32)
assert p._storage_kind is None
disk = p.collection_disk_bytes
assert disk > 0  # caller MUST reserve this before append; this test is not disk admission
p._compact_start()  # adopt pending owner before acquiring real backing
try: p._storage_kind = 'compact'
except AttributeError: pass
else: raise AssertionError('storage observation is writable')
files = []
for key, shape in p.shapes:
    count = 1
    for dim in shape: count *= dim
    data = (bytes((0,0,128,63)) if 'norm' in key else bytes(4)) * count
    files.append((key, data))
    p.append(key, 0, data)
    assert p._storage_kind is None
manifest = p.seal(b''.join(i.to_bytes(4, 'little') for i in (0,1,0,1)))
assert p._storage_kind == 'compact'
assert p.collection_disk_bytes == disk
workers = [Worker._from_source_request(manifest, role, server_policy=(experiment,config)) for role in range(3)]
for worker in workers:
    assert worker._storage_kind is None
    for key, data in files: worker.append_public_source(key, 0, data)
    worker.seal_public_source()
    assert worker._storage_kind == 'compact'
# Actual same-owner input sharing is available only AFTER genuine compilation.
assert p.next_chunk() is not None
assert p._storage_kind == 'compact'
# Worker close consumes its envelope; publisher close retains its envelope.
for worker in workers:
    assert worker._storage_kind == 'compact'
    worker.close()
    try: worker._storage_kind
    except ValueError: pass
    else: raise AssertionError('retired worker retained storage observation')
p.close()
assert p._storage_kind is None
assert p.collection_disk_bytes == disk
try: p.collection_disk_directory
except ValueError as exc: assert str(exc) == 'compact directory unavailable after failed operation'
else: raise AssertionError('retired publisher retained directory observation')
p.close()
# The ordinary constructor is not redirected to compact by the new binding.
legacy = Publisher(config, 4, 2, bytes([61])*32, bytes([62])*32, weights='public', range_policy='checked', max_abs_activation=1)
assert legacy._storage_kind is None
for key, data in files: legacy.append(key, 0, data)
legacy.seal(b''.join(i.to_bytes(4, 'little') for i in (0,1,0,1)))
assert legacy._storage_kind == 'expanded'
legacy.close()
",Some(&locals),Some(&locals)).unwrap();
    });
}

#[test]
fn actual_compact_publisher_factory_refuses_unsupported_and_conversion_failures_burn_owner() {
    let fixture = expanded_source_fixture();
    let experiment = compact_experiment();
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_compact_native_publisher_errors").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Publisher", module.getattr("ReplicatedPublisher").unwrap())
            .unwrap();
        locals
            .set_item("experiment", PyBytes::new(py, &experiment))
            .unwrap();
        locals
            .set_item("config", PyBytes::new(py, &fixture.config))
            .unwrap();
        py.run(c"
base = (experiment, config, 4, 2, bytes([61])*32, bytes([62])*32)
for index, bad in ((0,bytearray(experiment)),(2,True),(3,1),(4,bytes(31))):
    args=list(base); args[index]=bad
    try: Publisher._from_compact_source_request(*args)
    except (TypeError,ValueError): pass
    else: raise AssertionError('invalid compact factory arguments accepted/fell back')
import os

class BytesSubclass(bytes): pass

def assert_error(call, kind, message=None):
    try: call()
    except Exception as exc:
        assert type(exc) is kind, (type(exc), str(exc))
        if message is not None: assert str(exc) == message, str(exc)
        return exc
    raise AssertionError('invalid operation accepted')

for started in (False, True):
    # Fresh genuine first tensor/key and full F32 frame BEFORE any failure.
    probe = Publisher._from_compact_source_request(*base)
    key, shape = probe.shapes[0]
    count = 1
    for dim in shape: count *= dim
    data = (bytes((0,0,128,63)) if 'norm' in key else bytes(4)) * count
    assert 0 < len(data) <= 16384
    probe.close()
    del probe
    cases = [
        ('append', (), {}, TypeError, 'replicated missing required argument'),
        ('append', (key,0,data,0), {}, TypeError, 'replicated argument count exceeds signature'),
        ('append', (key,0,data), {'key':key}, TypeError, 'replicated multiple values for argument'),
        ('append', (key,0,data), {'unknown':0}, TypeError, 'replicated unexpected keyword argument'),
        ('append', (object(),0,data), {}, TypeError, None),
        ('append', ('k'*513,0,data), {}, ValueError, 'replicated key too long'),
        ('append', (key,True,data), {}, ValueError, 'replicated integer cannot be bool'),
        ('append', (key,object(),data), {}, TypeError, None),
        ('append', (key,-1,data), {}, OverflowError, None),
        ('append', (key,1<<200,data), {}, OverflowError, None),
        ('append', (key,0,bytearray(data)), {}, TypeError, None),
        ('append', (key,0,BytesSubclass(data)), {}, TypeError, None),
        ('append', (key,0,bytes(16385)), {}, ValueError, 'replicated frame exceeds bound'),
        ('seal', (), {}, TypeError, 'replicated missing required argument'),
        ('seal', (bytes(16),0), {}, TypeError, 'replicated argument count exceeds signature'),
        ('seal', (), {'unknown':bytes(16)}, TypeError, 'replicated unexpected keyword argument'),
        ('seal', (bytes(16),), {'ids':bytes(16)}, TypeError, 'replicated multiple values for argument'),
        ('seal', (bytearray(16),), {}, TypeError, None),
        ('seal', (BytesSubclass(bytes(16)),), {}, TypeError, None),
        ('seal', (bytes(129),), {}, ValueError, 'replicated frame exceeds bound'),
        ('seal', (bytes(3),), {}, ValueError, 'replicated IDs require u32 bytes'),
        ('finish', (), {}, TypeError, 'replicated missing required argument'),
        ('finish', (b'',b'',b'',b''), {}, TypeError, 'replicated argument count exceeds signature'),
        ('finish', (b'',b''), {'unknown':b''}, TypeError, 'replicated unexpected keyword argument'),
        ('finish', (b'',b'',b''), {'first':b''}, TypeError, 'replicated multiple values for argument'),
    ]
    for method in ('_compact_start', 'next_chunk', 'close'):
        cases.extend((method,args,kwargs,TypeError,'replicated argument count exceeds signature')
                     for args,kwargs in (((object(),),{}),((),{'unexpected':True})))
    for index in range(3):
        for bad,kind,message in ((bytearray(),TypeError,None),
                                 (BytesSubclass(),TypeError,None),
                                 (bytes(146),ValueError,'replicated frame exceeds bound')):
            frames = [b'',b'',b'']; frames[index] = bad
            cases.append(('finish',tuple(frames),{},kind,message))
    for method,args,kwargs,kind,message in cases:
        p = Publisher._from_compact_source_request(*base)
        disk = p.collection_disk_bytes
        directory = identity = None
        if started:
            p._compact_start()
            directory = p.collection_disk_directory
            stat = os.stat(directory)
            identity = (stat.st_dev,stat.st_ino)
        assert_error(lambda: getattr(p,method)(*args,**kwargs), kind, message)
        assert p._storage_kind is None
        assert p.collection_disk_bytes == disk  # custody observation, NOT reuse proof
        assert_error(lambda: p.collection_disk_directory, ValueError,
                     'compact directory unavailable after failed operation')
        assert_error(lambda: p._compact_start(), ValueError, 'compact acquisition already attempted')
        assert_error(lambda: p.append(key,0,data), ValueError, 'compact acquisition not ready')
        # Exact bytes, bounded frames pass binding conversion; compiler checks
        # terminal acquisition BEFORE stage/frame parsing (not generic badframe).
        assert_error(lambda: p.finish(bytes(145),bytes(145),bytes(145)), ValueError,
                     'compact acquisition not ready')
        assert_error(lambda: p.seal(bytes(16)), ValueError, 'compact acquisition not ready')
        assert_error(lambda: p.next_chunk(), ValueError, 'compact acquisition not ready')
        if started:
            stat = os.stat(directory)
            assert (stat.st_dev,stat.st_ino) == identity
        p.close()  # only valid explicit retirement removes acquired backing
        assert p._storage_kind is None and p.collection_disk_bytes == disk
        assert_error(lambda: p.collection_disk_directory, ValueError,
                     'compact directory unavailable after failed operation')
        if started: assert not os.path.exists(directory)
        p.close()
        del p

# A failed sealed owner keeps its genuine storage witness until explicit retire.
p = Publisher._from_compact_source_request(*base)
p._compact_start()
directory = p.collection_disk_directory
disk = p.collection_disk_bytes
for key,shape in p.shapes:
    count = 1
    for dim in shape: count *= dim
    p.append(key,0,(bytes((0,0,128,63)) if 'norm' in key else bytes(4))*count)
p.seal(bytes(16))
assert p._storage_kind == 'compact'
assert_error(lambda: p.next_chunk(unexpected=True), TypeError,
             'replicated argument count exceeds signature')
assert p._storage_kind == 'compact'  # real retained sealed artifact, not reusable
assert_error(lambda: p.next_chunk(), ValueError, 'compact acquisition not ready')
assert os.path.isdir(directory) and p.collection_disk_bytes == disk
p.close()
assert p._storage_kind is None and p.collection_disk_bytes == disk
assert not os.path.exists(directory)
p.close()
del p
",Some(&locals),Some(&locals)).unwrap();
    });
}

#[test]
fn compact_snapshot_directory_is_actual_readonly_and_retained_through_seal() {
    let fixture = expanded_source_fixture();
    let experiment = compact_experiment();
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_compact_snapshot_location_test").unwrap();
        register(&module).unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("Publisher", module.getattr("ReplicatedPublisher").unwrap())
            .unwrap();
        locals
            .set_item("experiment", PyBytes::new(py, &experiment))
            .unwrap();
        locals
            .set_item("config", PyBytes::new(py, &fixture.config))
            .unwrap();
        py.run(
            c"
import os
p = Publisher._from_compact_source_request(experiment, config, 4, 2, bytes([61])*32, bytes([62])*32)
parent = p.collection_disk_directory  # parent-only observation, not acquired inode
p._compact_start()  # pending directory getter reports parent only
directory = p.collection_disk_directory
assert directory != parent and os.path.dirname(directory) == parent
assert isinstance(directory, str) and os.path.isabs(directory)
assert os.path.isdir(directory)  # actual native anchor BEFORE first append
identity = os.stat(directory)
assert p._storage_kind is None  # a real directory is not an execution seal
assert p.collection_disk_bytes > 0  # location is NOT a DiskLease
try: p.collection_disk_directory = directory
except AttributeError: pass
else: raise AssertionError('snapshot location observer is writable')
for key, shape in p.shapes:
    count = 1
    for dim in shape: count *= dim
    p.append(key, 0, (bytes((0,0,128,63)) if 'norm' in key else bytes(4))*count)
    assert p.collection_disk_directory == directory
p.seal(b''.join(i.to_bytes(4, 'little') for i in (0,1,0,1)))
assert p._storage_kind == 'compact'
assert p.collection_disk_directory == directory
current = os.stat(directory)
assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)
p.close()
try: p.collection_disk_directory
except ValueError as exc: assert str(exc) == 'compact directory unavailable after failed operation'
else: raise AssertionError('retired publisher echoed cached directory')
# Python strings/stat metadata are NOT native owner handles.
assert not os.path.exists(directory)
# Legacy anonymous files have no explicit compact owner. Do not guess a path.
legacy = Publisher(config, 4, 2, bytes([61])*32, bytes([62])*32)
try: legacy.collection_disk_directory
except ValueError: pass
else: raise AssertionError('anonymous default publisher fabricated a location')
legacy.close()
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}
