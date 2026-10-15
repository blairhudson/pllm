/** One audited cold cohort; complete worker peaks, with distinct role contracts. */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { clientMemoryEvidence } from './research-memory.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const digest = (value) => /^[a-f0-9]{64}$/.test(value ?? '');
const check = (condition, message) => { if (!condition) throw new Error(message); };
const same = (left, right, message) => check(isDeepStrictEqual(left, right), message);
const median = (values) => {
  check(values.length > 0 && values.every((value) => Number.isFinite(value) && value >= 0), 'Invalid tier observation');
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const axes = {
  clientPeak: ['process.peak_rss', 1e6], clientCpu: ['client.cold_cpu', 1],
  preparationCpu: ['preparation.cold_cpu', 1], preparationPeak: ['preparation.peak_rss', 1e6],
  ttft: ['response.ttft', 1], decodeTps: ['response.decode_tps', 1],
  totalNetwork: ['network.all_link_stream_bytes', 1e6], full: ['request.0.full', 1],
};
const evidenceContracts = {
  'private-boundary-tiers-v1': { plan: 'pllm.private_boundary_tiers_plan.v1',
    summary: 'pllm.private_boundary_tiers.v1', cases: 8 },
  'private-boundary-decode-v1': { plan: 'pllm.private_boundary_decode_plan.v1',
    summary: 'pllm.private_boundary_decode.v1', cases: 5 },
};

export function privateBoundaryTierCohorts(spec, document, plan, read, helpers) {
  const summaryRaw = read(spec.summary), summary = JSON.parse(summaryRaw);
  const contract = evidenceContracts[spec.adapter];
  check(hash(summaryRaw) === spec.summary_sha256 &&
    contract && plan?.schema === contract.plan && plan.tiny === false &&
    summary.schema === contract.summary && summary.complete === true &&
    summary.all_logits_active_kv_positions_selections_rng_and_outputs_match === true &&
    document.schema === 'pllm.benchmark_run.v1' && document.complete === true && digest(document.cohort),
  'Missing complete audited tier evidence');
  same(document.suite, plan.suite, 'Tier suite differs from frozen plan');
  const suite = document.suite, count = suite.sampling.processes;
  check(suite.schema === 'pllm.benchmark_suite.v1' && count >= 3 && count === summary.replicates &&
    suite.sampling.order === 'balanced' && suite.sampling.warmups === 0 && suite.sampling.repetitions === 1 &&
    suite.cases.length === contract.cases && document.results.length === count * suite.cases.length &&
    summary.clients === document.results.length && plan.preflight.length === suite.cases.length &&
    plan.preflight.every((cell) => cell.admission.admitted === true), 'Incomplete repeated tier cohort');
  for (const file of [spec.replay_file, spec.frozen_factory.file]) {
    const entry = Object.entries(plan.frozen_sources).find(([name]) => name === file || name.endsWith(`/${file}`));
    check(entry && hash(read(file)) === entry[1], 'Tier measurement source changed');
  }
  same(Object.keys(summary.groups).sort(), suite.cases.map((cell) => cell.name).sort(), 'Tier recipe set changed');
  const groups = new Map(), seen = new Set();
  let reference = null, states = null, contracts = null, checkedValues = 0, maximumSwap = 0;
  for (const result of document.results) {
    const e = result.execution, name = e.case.name;
    const selected = suite.cases.find((cell) => cell.name === name);
    same(e.case, selected, 'Tier case differs from frozen suite');
    check(result.schema_version === 'pllm.benchmark_result.v2' && result.status === 'completed' &&
      e.cohort === document.cohort && e.failure_code === null && e.admission.admitted === true &&
      e.oracle.outcome === 'passed' && e.isolation.fresh_process === true &&
      e.fixture.metadata.owned_client_cache === 'fresh' &&
      e.build.digest === selected.build.expected_digest && digest(e.build.digest) &&
      result.warmups === 0 && Number.isInteger(e.replicate) && e.replicate >= 0 && e.replicate < count,
    'Unchecked tier build, isolation or admission');
    same(e.sampling, suite.sampling, 'Tier sampling changed');
    const cell = `${name}/${e.replicate}`;
    check(!seen.has(cell), 'Duplicate tier replicate'); seen.add(cell);
    const parameters = selected.target.parameters, experiment = parameters.model.experiment;
    const report = e.details.report, run = report.runs[0], audit = e.details.response_state_audit;
    check(selected.target.kind === 'response' && parameters.state_audit === true &&
      parameters.scenario.kind === 'cold' && !parameters.concurrent_requests &&
      report.runs.length === 1 && report.warmup_runs.length === 0 &&
      report.checks.passed === true && Object.values(report.checks).every((value) => value === true) &&
      report.prepared_material_accounting.conserved === true && report.memory_guard.tripped === false &&
      report.configuration.provider_backend === 'native' && run.cold === true && run.warm === false &&
      run.tokens.input_tokens === 150 && run.tokens.output_tokens === 8 && run.max_output_tokens === 8 &&
      experiment.budget.requests === 1 && experiment.budget.max_input_tokens === 150 && experiment.budget.max_new_tokens === 8 &&
      audit.schema === 'pllm.response_state_audit.v1' &&
      audit.bindings.length === 1 && audit.bindings[0].failed === false &&
      audit.bindings[0].cryptographic_randomness_frozen === false && audit.bindings[0].selections.length === 8,
    'Tier response or complete-state audit failed');
    helpers.checkRun(run);
    same(audit.client_memory_owners, ['sdk', 'dashboard', 'native_wan_relay', 'audit', 'role_supervision'],
      'Tier client owner list changed');
    check(report.configuration.source_lock_digest === plan.source_lock_digest &&
      e.comparison.source_digest === report.configuration.source_lock_digest &&
      e.fixture.digest === e.comparison.source_digest && e.comparison.workload_digest === result.workload_digest &&
      Object.keys(e.comparison).length === 7 && Object.values(e.comparison).every(digest) &&
      digest(result.environment.digest) && digest(result.configuration_digest) && digest(e.suite_digest) &&
      digest(audit.bindings[0].numeric_body_fingerprint), 'Tier source or numeric commitment changed');
    const identity = { source: report.configuration.source_lock_digest, prompt: report.configuration.prompt_digest,
      body: audit.bindings[0].numeric_body_fingerprint, tokens: run.tokens,
      output: run.generation.output_text_digest, wan: report.configuration.wan_emulation_digest,
      sampling: report.configuration.sampling, build: e.build };
    same(identity, summary.identity, 'Tier summary identity changed');
    // Deliberately compare exact state and selected metric contracts across the
    // different role graphs. Ordinary privacy/body-coverage identities survive.
    const comparison = { identity, environment: result.environment.digest, suite: e.suite_digest,
      numeric: e.comparison.numeric_digest, state: e.comparison.state_digest,
      workload: e.comparison.workload_digest, output: e.comparison.output_digest };
    if (reference) same(comparison, reference, 'Tier source, environment or comparison changed');
    reference = comparison;
    if (states) same(audit.bindings, states, 'Tier logits, KV, positions, selections or RNG changed');
    states = audit.bindings;
    const records = new Map(result.metrics.map((metric) => [metric.id, metric]));
    check(records.size === result.metrics.length, 'Duplicate tier metric');
    const measurements = Object.fromEntries([...records].map(([id, metric]) => [id, metric.value]));
    if (spec.adapter === 'private-boundary-decode-v1') {
      helpers.checkedResponseBodies(report, measurements, name !== 'maintained');
    }
    const metricContracts = Object.values(axes).map(([id]) => {
      const { value, sample_count, unavailable_reason, ...contract } = records.get(id);
      check(Number.isFinite(value) && value >= 0 && sample_count === 1 && unavailable_reason === null,
        'Missing tier priority metric');
      return contract;
    });
    if (contracts) same(metricContracts, contracts, 'Tier metric owner, unit or window changed');
    contracts = metricContracts;
    const peak = measurements['process.peak_rss'], scenarioPeak = report.client_process_memory.after.lifetime_peak_rss_bytes;
    check(Number.isSafeInteger(peak) && peak >= scenarioPeak && scenarioPeak > 0 &&
      scenarioPeak === measurements['client.peak_rss'] &&
      records.get('process.peak_rss').window === 'process_birth_through_target' &&
      records.get('process.peak_rss').measurement_scope === 'process_lifetime_high_water' &&
      report.client_process_memory.scope.includes('client and in-process dashboard'), 'Whole-client tier owner missing');
    const wan = helpers.checkedWan(report.native_network_accounting.samples.runs[0].after.wan_emulation);
    check(report.wan_readiness.emulation.native_stream_rate_snapshots_checked === true &&
      wan.conditions_digest === identity.wan && wan.total_stream_bytes === measurements['network.all_link_stream_bytes'],
    'Tier stream measurement or rate scope changed');
    const swap = report.memory_guard.maximum_host_swap_growth_bytes;
    check(Number.isSafeInteger(swap) && swap >= 0, 'Unknown tier swap evidence');
    maximumSwap = Math.max(maximumSwap, swap);
    for (const selection of states[0].selections) {
      check(Number.isSafeInteger(selection.checked_values) && selection.checked_values > 0, 'Empty tier state check');
      checkedValues += selection.checked_values;
    }
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push({ result, report, experiment, measurements, peak, wan });
  }
  check(checkedValues === summary.checked_values && maximumSwap === summary.maximum_observed_swap_growth_bytes,
    'Tier state or swap totals changed');
  const rows = [...groups].map(([name, samples]) => {
    const first = samples[0], recorded = summary.groups[name];
    check(samples.length === count, 'Missing tier recipe replicate');
    same(recorded.experiment, first.experiment, 'Tier summarized recipe changed');
    for (const sample of samples) {
      same(sample.experiment, first.experiment, 'Tier recipe changed across replicates');
      check(sample.result.configuration_digest === first.result.configuration_digest &&
        sample.report.experiment.pipeline_digest === first.report.experiment.pipeline_digest,
      'Tier configuration binding changed');
    }
    const maxima = Math.max(...samples.map((sample) => sample.peak));
    check(recorded.maximum_client_peak_bytes === maxima, 'Tier maximum hid a replicate');
    same(recorded.observed_tiers, Object.fromEntries([['low', 256 * 2 ** 20], ['medium', 2 * 2 ** 30], ['high', 16 * 2 ** 30]]
      .map(([tier, maximum]) => [tier, maxima <= maximum])), 'Tier budget qualification changed');
    same(Object.keys(recorded.medians).sort(), [...new Set([...Object.values(axes).map(([id]) => id),
      'client.peak_rss', 'network.covered_body_bytes', 'target.latency'])].sort(), 'Tier summary metric set changed');
    for (const [id, cached] of Object.entries(recorded.medians)) {
      const values = samples.map((sample) => sample.measurements[id]);
      same(cached, values.some((value) => value === null) ? null : median(values), 'Tier summary median changed');
    }
    const pipeline = first.experiment.pipeline;
    const placement = helpers.executionPlacement(pipeline, first.report.configuration.roles, first.report.client_body_placement);
    const optional = (select, divisor = 1) => {
      const observations = samples.map(select);
      return observations.some((value) => value == null) ? null : median(observations) / divisor;
    };
    const values = { ...Object.fromEntries(helpers.metrics.map(({ id }) => [id, null])),
      ...Object.fromEntries(Object.entries(axes).map(([axis, [id, divisor]]) => [axis, recorded.medians[id] / divisor])),
      requestTps: median(samples.map(({ report }) => 8 / report.runs[0].durations.full_seconds)),
      clientNetwork: median(samples.map(({ wan }) => wan.client_stream_bytes)) / 1e6,
      networkPerToken: recorded.medians['network.all_link_stream_bytes'] / 1e6 / 8,
      aggregateCpu: optional(({ report }) => report.process_cpu_accounting?.aggregate_cold_first_response_cpu_seconds),
      providerPeak: optional(({ report }) => helpers.providerPeak(report.configuration.roles, report.runs[0].processes)),
      inferencePeak: optional(({ report }) => report.runs[0].processes?.inference?.rss_peak_bytes, 1e6),
      totalPeak: optional(({ report }) => report.native_process_memory?.complete === true
        ? report.native_process_memory.sampled_total_peak_rss_bytes : null, 1e6),
      bodyOffload: placement.clientMacPercent === null ? null : 100 - placement.clientMacPercent };
    return { id: name, label: name, file: spec.frozen_factory.file, fileSha256: spec.frozen_factory.sha256,
      configurationDigest: first.result.configuration_digest, sourceLock: summary.identity.source,
      pipeline, experiment: first.experiment, reference: name === 'maintained' ? 'prepared' : null,
      paper: null, publicPrefixTokens: 0, placement, metrics: values, sampleCount: count,
      clientMemory: clientMemoryEvidence(samples.map((sample) => sample.peak), 'process_birth_through_target'),
      provenance: { report: spec.report, reportSha256: spec.sha256, plan: spec.plan, planSha256: spec.plan_sha256 },
      buildDigest: first.result.execution.build.digest, frozenSources: plan.frozen_sources,
      comparisonCommitments: first.result.execution.comparison,
      recordedCase: first.result.execution.case, samplingContract: suite.sampling,
      measurementContracts: first.result.metrics.map(({ value, ...contract }) => contract),
      replay: { originalDriver: spec.replay_file, originalDriverSha256: spec.replay_sha256,
        source: spec.report.split('/').at(-1).replace(/\.json$/, ''), case: name,
        callable: 'pllm.metrics.BenchmarkCase.from_spec', argument: 'recordedCase',
        requiresOriginalBuildAndLifecycle: true },
      scopedTimings: { completeTargetSeconds: recorded.medians['target.latency'] } };
  });
  const first = document.results[0].execution.details.report, pipeline = rows[0].pipeline;
  return [{ ...spec, baseline: 'maintained', model: pipeline.model.source,
    identity: { model: pipeline.model.source, source: pipeline.model, body: summary.identity.body,
      prompt: summary.identity.prompt, quantization: pipeline.components.quantization,
      inputs: 150, outputs: 8, cap: 8, warm: false, output: summary.identity.output,
      nonlinear: null, state: null, verification: null, sampling: first.runs[0].sampling,
      backend: 'native', wan: summary.identity.wan, links: null, wanBackend: 'native-shared-tcp-pacer', clientIsolation: true },
    measurement: { requestScope: 'single audited cold request', phase: 'cold', sampleCount: count,
      clientRssLabel: 'Client RAM · complete worker lifetime', clientCpuLabel: 'Client CPU · cold first response',
      preparationCpuLabel: 'Prep CPU · birth to cold first response',
      cpu: 'SDK, dashboard and relay share the client process; cold first-response window',
      rss: 'Largest complete worker lifetime peak across replicates qualifies each tier; idle suite launcher excluded',
      timing: 'Online TTFT and N−1 decode; exact logits/KV audit charged in every arm; provider startup excluded from request latency',
      traffic: 'Source-present all-link TCP streams from relay startup through response; physical wire and source/tokenizer distribution unknown' }, rows }];
}

// Separate contract: never weaken the archived readers' fixed eight-output rules.
const confirmationNames = ['maintained', 'coupled-serial-u32', 'coupled-parallel-uniform_residues'];
const confirmationMetrics = [...new Set([...Object.values(axes).map(([id]) => id),
  'client.peak_rss', 'network.covered_body_bytes', 'target.latency'])];
const rowNames = ['issued', 'reserved', 'claimed', 'burned', 'discarded'];
const fixedFields = ['preparation_rows', 'inference_stage_calls', 'preparation_attributed_body_bytes',
  'preparation_prefill_material_body_bytes', 'preparation_decode_material_body_bytes',
  'preparation_prefill_rows_issued', 'preparation_decode_rows_issued', 'preparation_padding_rows_issued',
  ...rowNames.slice(1).map((name) => `prepared_stage_rows_${name}`), 'generation_executed_decode_steps', 'generation_output_tokens'];
const preparedFields = ['inference_upload_bytes', 'correction_push_bytes', 'correction_ack_bytes',
  'preparation_upload_bytes', 'preparation_download_bytes', 'prepared_stage_rows_issued'];
const boundaryFields = ['private_boundary_lookup_upload_bytes', 'private_boundary_lookup_download_bytes',
  'private_boundary_head_upload_bytes', 'private_boundary_lookup_rows', 'private_boundary_head_calls'];
const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const select = (value, fields) => Object.fromEntries(fields.map((key) => [key, value[key]]));
const canonical = (value, spaced = false, key = '') => {
  const comma = spaced ? ', ' : ',', colon = spaced ? ': ' : ':';
  if (Array.isArray(value)) return `[${value.map((item) => canonical(item, spaced)).join(comma)}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map((name) =>
    `${JSON.stringify(name)}${colon}${canonical(value[name], spaced, name)}`).join(comma)}}`;
  // SDK serializes these WAN fields as floats, even when integral. Other public
  // suite/build fields are integer/string/null; do not use JS's lossy 100 form.
  if (Number.isInteger(value) && ['download_mbps', 'upload_mbps'].includes(key)) return `${value}.0`;
  return JSON.stringify(value);
};
const sdkDigest = (value) => hash(canonical(value));
const equalNumbers = (actual, expected, message) => {
  if (typeof expected === 'number') check(Number.isFinite(actual) && Math.abs(actual - expected) <=
    Number.EPSILON * Math.max(1, Math.abs(actual), Math.abs(expected)) * 4, message);
  else if (expected && typeof expected === 'object') {
    same(Object.keys(actual ?? {}).sort(), Object.keys(expected).sort(), message);
    for (const key of Object.keys(expected)) equalNumbers(actual[key], expected[key], message);
  } else same(actual, expected, message);
};

function confirmationBodies(report, values, boundary, early, chunk, helpers) {
  helpers.checkedResponseBodies(report, values, boundary); // original raw owners and directed TCP windows
  const p = report.runs[0].privacy, startup = report.topology_accounting.startup.traffic_attribution.audit_counters;
  const inventory = Object.fromEntries(rowNames.map((name) => {
    const key = `prepared_stage_rows_${name}`;
    check(integer(p[key]) && integer(startup[key]), 'Confirmation raw inventory missing');
    return [name, p[key] + startup[key]];
  }));
  const material = report.prepared_material_accounting;
  same(material.counts, inventory, 'Confirmation sealed inventory differs from raw rows');
  check(material.schema === 'pllm.prepared_material_accounting.v1' && material.conserved === true &&
    material.remaining_stage_rows === 0 && inventory.issued === inventory.claimed + inventory.burned + inventory.discarded &&
    inventory.reserved === inventory.claimed + inventory.burned, 'Confirmation material not conserved');
  check([...fixedFields, ...preparedFields].every((key) => integer(p[key])) && integer(chunk) && chunk > 0 &&
    p.inference_stage_calls > 0 && p.inference_stage_calls <= p.prepared_stage_rows_claimed &&
    p.preparation_rows === p.prepared_stage_rows_issued && p.prepared_stage_rows_issued ===
      p.prepared_stage_rows_claimed + p.prepared_stage_rows_burned + p.prepared_stage_rows_discarded &&
    p.prepared_stage_rows_reserved === p.prepared_stage_rows_claimed + p.prepared_stage_rows_burned &&
    p.prepared_stage_rows_issued === p.preparation_prefill_rows_issued + p.preparation_decode_rows_issued + p.preparation_padding_rows_issued &&
    p.preparation_prefill_material_body_bytes + p.preparation_decode_material_body_bytes <= p.preparation_attributed_body_bytes &&
    p.preparation_attributed_body_bytes === p.preparation_upload_bytes + p.preparation_download_bytes + p.correction_push_bytes,
  'Confirmation issued/claimed/material work changed');
  if (!early) check(p.prepared_stage_rows_issued === p.prepared_stage_rows_reserved &&
    p.prepared_stage_rows_reserved === p.prepared_stage_rows_claimed && p.prepared_stage_rows_burned === 0 &&
    p.prepared_stage_rows_discarded === 0, 'Confirmation at-cap unused rows');
  const stages = report.topology_accounting.stages.runs;
  check(stages.length === 1, 'Confirmation stage windows changed');
  const stage = stages[0], measured = stage.body_bytes_by_stage_and_edge;
  const fields = { 'client->inference': 'inference_upload_bytes', 'inference->client': 'inference_download_bytes',
    'client->preparation': 'preparation_upload_bytes', 'preparation->client': 'preparation_download_bytes',
    'preparation->inference': 'correction_push_bytes' };
  check(Object.keys(measured).length > 0 && Object.keys(measured).length <= 4096, 'Missing confirmation stages');
  for (const edges of Object.values(measured)) {
    same(Object.keys(edges).sort(), Object.keys(fields).sort(), 'Confirmation stage links changed');
    check(Object.values(edges).every(integer), 'Invalid confirmation stage bytes');
  }
  const sums = Object.fromEntries(Object.keys(fields).map((edge) => [edge,
    Object.values(measured).reduce((total, row) => total + row[edge], 0)]));
  same(sums, Object.fromEntries(Object.entries(fields).map(([edge, field]) => [edge, p[field]])), 'Confirmation stage raw bytes differ');
  same(sums, stage.measured_body_bytes_by_edge, 'Confirmation stage sum differs');
  same(sums, stage.expected_stage_body_bytes_by_edge, 'Confirmation stage expected bytes differ');
  check(stage.reconciled_with_protocol_bodies === true, 'Unreconciled confirmation stages');
  const rows = early ? p.prepared_stage_rows_claimed : p.prepared_stage_rows_issued;
  if (boundary) {
    check(boundaryFields.every((key) => integer(p[key])) && integer(p.private_boundary_head_download_bytes) &&
      p.private_boundary_head_download_bytes > 0 && p.private_boundary_head_calls === p.generation_executed_decode_steps + 1 &&
      p.private_boundary_lookup_rows === report.runs[0].tokens.input_tokens + p.generation_executed_decode_steps,
    'Confirmation boundary work differs');
    const online = ['boundary_a', 'boundary_b'].flatMap((role) => ['upload', 'download'].map((direction) =>
      p[`role_link.${role}.online_${direction}_bytes`]));
    check(online.every(integer) && online.reduce((a, b) => a + b, 0) === p.private_boundary_head_download_bytes +
      boundaryFields.filter((key) => key.endsWith('_bytes')).reduce((sum, key) => sum + p[key], 0), 'Confirmation boundary bodies differ');
  }
  return { prepared: select(p, preparedFields), fixed: select(p, fixedFields), boundary: select(p, boundaryFields),
    stages: measured, reply: p.inference_download_bytes, head: p.private_boundary_head_download_bytes,
    total: values['network.covered_body_bytes'],
    controls: report.topology_accounting.startup.lifecycle_control_serialized_body_bytes + report.topology_accounting.runs[0].lifecycle_control_serialized_body_bytes,
    bound: 8 * (Math.ceil(rows / chunk) + p.inference_stage_calls - 1) };
}

function confirmationMeasurements(result) {
  const e = result.execution, report = e.details.report, run = report.runs[0], d = run.durations;
  const output = run.tokens.output_tokens, records = new Map(result.metrics.map((m) => [m.id, m]));
  const kinds = { seconds: ['pllm/latency', 'seconds'], cpu: ['pllm/cpu-time/v1', 'seconds'],
    memory: ['pllm/memory', 'bytes'], bytes: ['pllm/communication', 'bytes'], throughput: ['pllm/throughput', 'per_second'] };
  const add = (id, value, { kind = 'seconds', role = 'client', window = 'execution', scope = 'target',
    statistic = 'median', phase = 'full', denominator = null } = {}) => {
    const [component, unit] = kinds[kind];
    const parameters = { seconds: { statistic, phase }, cpu: {}, memory: { kind: 'peak_rss' },
      bytes: { direction: 'total', phase }, throughput: { basis: 'tokens' } }[kind];
    const expected = { id, component, parameters, unit, unit_detail: kind === 'throughput' ? 'tokens_per_second' : null,
      statistic, phase, role, origin: value === null ? 'not_available' : 'measured',
      unavailable_reason: value === null ? 'measurement unavailable' : null, sample_count: value === null ? 0 : 1,
      window, measurement_scope: scope, denominator, owners: [role] };
    const actual = records.get(id);
    check(actual !== undefined, 'Confirmation required metric absent');
    const { value: observed, ...metadata } = actual;
    same(metadata, expected, `Confirmation explicit measurement contract changed: ${id}`);
    equalNumbers(observed, value, `Confirmation metric differs from raw observation: ${id}`);
  };
  const wall = e.details.timings.wall_seconds;
  check(wall.length === 1 && Number.isFinite(wall[0]) && wall[0] >= 0, 'Confirmation raw target sample missing');
  check(result.samples.length === 1, 'Confirmation v2 sample count changed');
  same(Object.keys(result.samples[0]).sort(), ['index', 'kind', 'status', 'duration_seconds', 'artifact_digest'].sort(), 'Confirmation v2 sample fields changed');
  check(result.samples[0].index === 0 && result.samples[0].kind === 'measurement' && result.samples[0].status === 'completed' &&
    (result.samples[0].artifact_digest === null || digest(result.samples[0].artifact_digest)), 'Confirmation v2 sample status changed');
  equalNumbers(result.samples[0].duration_seconds, wall[0], 'Confirmation sample/execution duration mismatch');
  const cpu = report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role;
  add('client.peak_rss', report.client_process_memory.after.lifetime_peak_rss_bytes,
    { kind: 'memory', window: 'process_birth_through_scenario', scope: 'process_lifetime_high_water' });
  for (const role of ['client', 'preparation']) add(`${role}.cold_cpu`, cpu[role], { kind: 'cpu', role,
    window: role === 'client' ? 'benchmark_start_through_first_response' : 'role_birth_through_first_response', scope: 'process_cpu' });
  add('preparation.peak_rss', run.processes.preparation.rss_peak_bytes,
    { kind: 'memory', role: 'preparation', window: 'request_windows', scope: 'sampled_process_rss' });
  add('request.0.full', d.full_seconds, { window: 'request_setup_through_completion', scope: 'ordinary_response_lifecycle' });
  for (const id of ['request.0.ttft', 'response.ttft']) add(id, d.ttft_seconds, { window: 'online_to_first_token', phase: 'online' });
  let decode = null;
  if (output > 1) {
    check(Number.isFinite(d.generation_seconds) && d.generation_seconds > 0, 'Confirmation raw generation duration invalid');
    decode = (output - 1) / d.generation_seconds;
    equalNumbers(d.tokens_per_second, decode, 'Confirmation cached raw decode rate differs from N-1 duration');
  }
  for (const id of ['request.0.decode_tps', 'response.decode_tps']) add(id, decode,
    { kind: 'throughput', window: 'after_first_output', phase: 'online', denominator: { unit: 'generated_tokens', value: Math.max(output - 1, 0) } });
  add('network.all_link_stream_bytes', report.native_network_accounting.samples.runs[0].after.wan_emulation.total_stream_bytes,
    { kind: 'bytes', role: 'all-links', statistic: 'sum', window: 'relay_start_through_scenario', scope: 'tcp_stream_payload' });
  add('network.covered_body_bytes', report.topology_accounting.startup.all_link_serialized_body_bytes + report.topology_accounting.runs[0].all_link_serialized_body_bytes,
    { kind: 'bytes', role: 'all-links', statistic: 'sum', window: 'setup_through_scenario',
      scope: `covered_application_bodies:${report.topology_accounting.runs[0].body_counter_coverage}` });
  add('target.latency', wall[0]);
  // Finalized high-water is a raw measurement itself, not reconstructible from
  // scenario/child RSS. Validate its explicit owner/window; keep max-rep gating.
  add('process.peak_rss', records.get('process.peak_rss').value,
    { kind: 'memory', window: 'process_birth_through_target', scope: 'process_lifetime_high_water' });
}

/** New three-arm read-only confirmation adapter; no legacy contract relaxation. */
export function privateBoundaryDecodeConfirmationCohorts(spec, document, plan, read, helpers) {
  const summaryRaw = read(spec.summary), summary = JSON.parse(summaryRaw), suite = document.suite;
  const cap = summary.output_cap;
  check(spec.adapter === 'private-boundary-decode-confirmation-v1' && hash(summaryRaw) === spec.summary_sha256 &&
    plan.schema === 'pllm.private_boundary_decode_confirmation_plan.v1' && plan.tiny === false &&
    summary.schema === 'pllm.private_boundary_decode_confirmation.v1' && summary.tiny === false &&
    summary.complete === true && summary.all_logits_active_kv_positions_selections_rng_and_outputs_match === true &&
    summary.cohort === document.cohort && digest(document.cohort) && [8, 32, 64].includes(cap) && plan.output_cap === cap &&
    document.schema === 'pllm.benchmark_run.v1' && document.complete === true && document.results.length === 9 &&
    summary.clients === 9 && summary.replicates === 3, 'Incomplete confirmation schema/cohort');
  same(suite, plan.suite, 'Confirmation frozen suite changed');
  same(suite.cases.map(({ name }) => name), confirmationNames, 'Confirmation case membership changed');
  same(suite.sampling, { processes: 3, order: 'balanced', threads: 1, repetitions: 1, warmups: 0,
    timeout_seconds: 3600, memory_bytes: 8 * 2 ** 30 }, 'Confirmation sampling changed');
  check(suite.schema === 'pllm.benchmark_suite.v1' && suite.name === 'private-boundary-decode-confirmation-v1' &&
    plan.preflight.length === 3 && plan.preflight.every((row) => row.admission.admitted === true), 'Confirmation admission incomplete');
  same(plan.public_input_counts, { '150': 150 }, 'Confirmation public prompt bound changed');
  same(plan.preflight.map(({ case: name }) => name), confirmationNames, 'Confirmation preflight cells changed');
  same(Object.keys(summary.groups).sort(), [...confirmationNames].sort(), 'Confirmation summary groups changed');
  check(plan.frozen_sources && Object.values(plan.frozen_sources).every(digest), 'Confirmation frozen source hashes missing');
  // One explicitly registered historical exception; never discover snapshots,
  // rewrite plan paths, or fall back from a mismatching live helper implicitly.
  const factoryPath = 'benchmarks/research/private_boundary_decode_confirmation.py';
  const snapshotPath = 'docs/evidence/private-boundary-confirmation-initial-auditor.py';
  const initialHash = '2720640b8928f044cc65b6f2b361b5d88882202a323de4fe34a28f2fd007772c';
  const snapshots = spec.source_snapshots;
  const usedSnapshots = new Set();
  if (snapshots !== undefined) {
    check(snapshots !== null && typeof snapshots === 'object' && !Array.isArray(snapshots), 'Confirmation snapshot map invalid');
    same(Object.keys(snapshots), [factoryPath], 'Confirmation snapshot map contains missing/unlisted sources');
    const mapped = snapshots[factoryPath];
    check(mapped !== null && typeof mapped === 'object' && !Array.isArray(mapped), 'Confirmation snapshot entry invalid');
    same(Object.keys(mapped).sort(), ['file', 'sha256'], 'Confirmation snapshot entry fields changed');
    check(mapped.file === snapshotPath && mapped.sha256 === initialHash &&
      spec.frozen_factory.file === mapped.file && spec.frozen_factory.sha256 === mapped.sha256,
    'Confirmation unregistered snapshot or factory binding');
    const entries = Object.entries(plan.frozen_sources).filter(([name]) => name === factoryPath);
    check(entries.length === 1 && entries[0][1] === mapped.sha256, 'Confirmation snapshot absent/unused in original plan');
  } else {
    check(spec.frozen_factory.file === factoryPath, 'Confirmation archival factory requires explicit source snapshot map');
  }
  const readSource = (file) => {
    if (snapshots !== undefined && file === factoryPath) {
      usedSnapshots.add(file);
      return read(snapshots[file].file);
    }
    return read(file);
  };
  for (const file of [spec.replay_file, factoryPath]) {
    const entries = Object.entries(plan.frozen_sources).filter(([name]) => name === file || name.endsWith(`/${file}`));
    check(entries.length === 1 && hash(readSource(file)) === entries[0][1], 'Confirmation frozen measurement source changed');
  }
  // Public source-present artifact hashes stay pinned by the exact archived plan;
  // don't require an unshipped live checkpoint/tokenizer cache to read evidence.
  for (const [name, expected] of Object.entries(plan.frozen_sources)) {
    if (!name.endsWith('.py') && !name.startsWith('docs/evidence/')) continue;
    const path = name.startsWith('/') ? name.slice(name.lastIndexOf('/scripts/') + 1) : name;
    check(!path.startsWith('/') && hash(readSource(path)) === expected, 'Confirmation frozen helper changed');
  }
  same([...usedSnapshots], snapshots === undefined ? [] : [factoryPath], 'Confirmation snapshot map has unused entries');
  const seen = new Set(), groups = new Map(confirmationNames.map((name) => [name, []]));
  let common = null, states = null, contracts = null, checkedValues = 0;
  const ownerNames = ['sdk', 'dashboard', 'native_wan_relay', 'audit', 'role_supervision'];
  const sampleContract = { effective_temperature: 0, mode: 'greedy', requested_temperature: 0, top_p: null };
  for (const result of document.results) {
    const e = result.execution, selected = suite.cases.find(({ name }) => name === e.case.name), name = e.case.name;
    same(e.case, selected, 'Confirmation measured case changed');
    same(e.sampling, suite.sampling, 'Confirmation execution sampling changed');
    same(Object.keys(result).sort(), ['schema_version', 'id', 'created_at', 'status', 'scope', 'profile', 'component_ids', 'model',
      'plan_lock_digest', 'configuration_digest', 'workload_digest', 'privacy_cohort', 'numeric_cohort', 'environment',
      'warmups', 'repetitions', 'samples', 'metrics', 'limitations', 'execution'].sort(), 'Confirmation v2 result fields changed');
    same(Object.keys(e).sort(), ['case', 'case_digest', 'suite_digest', 'cohort', 'replicate', 'build', 'sampling', 'isolation',
      'fixture', 'oracle', 'admission', 'comparison', 'details', 'failure_code', 'budgets'].sort(), 'Confirmation v2 execution fields changed');
    same(e.budgets, [], 'Confirmation unexpected budget assessments');
    same(selected.budgets, [], 'Confirmation unexpected case budgets');
    same(Object.keys(e.build).sort(), ['attributes', 'digest'], 'Confirmation observed build fields changed');
    same(Object.keys(e.isolation).sort(), ['controller_included', 'fresh_process', 'sample_owners'], 'Confirmation isolation fields changed');
    check(Array.isArray(e.isolation.sample_owners) && e.isolation.sample_owners.length > 0 &&
      e.isolation.sample_owners.every((owner) => typeof owner === 'string' && owner.length > 0), 'Confirmation sample owners missing');
    check(e.fixture.kind === 'checkpoint' && ['required_bytes', 'limit_bytes', 'host_headroom_bytes'].every((key) => integer(e.admission[key])) &&
      e.admission.required_bytes <= Math.min(e.admission.limit_bytes, e.admission.host_headroom_bytes), 'Confirmation v2 fixture/resource admission invalid');
    check(selected && result.schema_version === 'pllm.benchmark_result.v2' && result.scope === 'deployment' &&
      result.status === 'completed' && result.warmups === 0 && result.repetitions === 1 &&
      Number.isInteger(e.replicate) && e.replicate >= 0 && e.replicate < 3 && !seen.has(`${name}/${e.replicate}`) &&
      e.cohort === document.cohort && e.suite_digest === sdkDigest(suite) && e.case_digest === sdkDigest(selected) &&
      e.failure_code === null && e.admission.admitted === true && e.details.topology_admission.admitted === true &&
      e.isolation.fresh_process === true && e.isolation.controller_included === false &&
      e.fixture.metadata.cached_checkpoint === true && e.fixture.metadata.owned_client_cache === 'fresh' &&
      digest(e.build.digest) && e.build.digest === sdkDigest(e.build.attributes) && e.build.digest === selected.build.expected_digest &&
      selected.build.expected_digest === suite.cases[0].build.expected_digest, 'Confirmation execution/build/cell identity changed');
    check(result.configuration_digest === hash('pllm.benchmark_target.v1\0' + canonical(selected.target)), 'Confirmation v2 target digest differs');
    seen.add(`${name}/${e.replicate}`);
    const p = selected.target.parameters, experiment = p.model.experiment, report = e.details.report;
    same(experiment.pipeline.model, suite.cases[0].target.parameters.model.experiment.pipeline.model, 'Confirmation recipe model/source differs');
    same(experiment.budget, { requests: 1, max_input_tokens: 150, max_new_tokens: cap }, 'Confirmation public bounds changed');
    same(p.wan, { download_mbps: 100, upload_mbps: 40, parties: [], role_parties: [], schema: 'pllm.wan_conditions.v1' }, 'Confirmation WAN caps changed');
    if (name === 'maintained') check(!experiment.pipeline.components.boundary &&
      experiment.pipeline.components.cache.params.fixed_input_tokens === 150, 'Confirmation maintained recipe changed');
    else same(experiment.pipeline.components.boundary, { component: 'pllm/private-lookup-full-head/v1', params: name === confirmationNames[1]
      ? {} : { head_dispatch: 'parallel', head_output_encoding: 'uniform_residues' } }, 'Confirmation head algorithm changed');
    check(selected.target.kind === 'response' && p.backend === 'native' && p.state_audit === true && p.scenario.kind === 'cold' &&
      p.scenario.count === 1 && p.scenario.max_output_tokens === cap && !p.concurrent_requests &&
      report.schema_version === 'pllm.loopback_benchmark.v1' && report.runs.length === 1 && report.warmup_runs.length === 0 &&
      report.memory_preflight.admitted === true && report.configuration.provider_backend === 'native' &&
      report.configuration.wan_emulation === true && report.configuration.warmups === 0 && report.configuration.repetitions === 1 &&
      report.configuration.max_output_tokens === cap && report.checks.passed === true &&
      Object.values(report.checks).every((value) => value === true), 'Confirmation target/report contract changed');
    for (const key of ['all_runs_completed', 'authoritative_token_usage', 'memory_guard_passed', 'model_fingerprint_present',
      'no_online_preparation_requests', 'no_plaintext_prompt_bytes_sent', 'no_plaintext_token_ids_sent', 'sampling_matches_request']) {
      check(report.checks[key] === true, 'Confirmation required report check missing');
    }
    const run = report.runs[0], audit = e.details.response_state_audit, binding = audit.bindings[0];
    helpers.checkRun(run); // permits early EOS, but intentionally rejects zero-output telemetry
    check(run.cold === true && run.warm === false && run.failure === null && run.max_output_tokens === cap &&
      run.model_id === experiment.pipeline.model.source && result.model.id === run.model_id &&
      digest(report.configuration.prompt_digest) && digest(report.experiment.pipeline_digest) && digest(report.experiment.configuration_digest) &&
      run.tokens.input_tokens === 150 && run.tokens.output_tokens > 0 && run.tokens.output_tokens <= cap &&
      audit.schema === 'pllm.response_state_audit.v1' && audit.bindings.length === 1 && binding.failed === false &&
      binding.sampler_seed === 42 && binding.temperature === 0 && binding.cryptographic_randomness_frozen === false &&
      digest(binding.numeric_body_fingerprint), 'Confirmation state/sampler scope changed');
    same(audit.client_memory_owners, ownerNames, 'Confirmation worker owners changed');
    same(report.configuration.sampling, sampleContract, 'Confirmation configured sampler changed');
    same(run.sampling, sampleContract, 'Confirmation requested sampler changed');
    const output = run.tokens.output_tokens, early = output < cap, selections = binding.selections.length;
    confirmationMeasurements(result);
    check(run.generation.response_status === (early ? 'completed' : 'incomplete') && selections === output + Number(early) &&
      selections <= cap && run.privacy.generation_requested_output_tokens === cap && run.privacy.generation_output_tokens === output &&
      run.privacy.generation_executed_decode_steps === selections - 1, 'Confirmation EOS/output/selection work differs');
    for (const [index, selection] of binding.selections.entries()) {
      check(selection.position === 150 + index && selection.cache_lengths.length > 0 &&
        selection.cache_lengths.every((length) => length === selection.position) && integer(selection.checked_values) && selection.checked_values > 0 &&
        ['logits_and_active_kv', 'selected_token', 'rng_before', 'rng_after'].every((key) => digest(selection[key])), 'Confirmation full state/position missing');
      checkedValues += selection.checked_values;
    }
    same(e.oracle, { kind: 'salted_full_logit_kv_sampler_commitments', outcome: 'passed',
      checked_values: binding.selections.reduce((sum, selection) => sum + selection.checked_values, 0) }, 'Confirmation oracle changed');
    const identity = { source_lock_digest: report.configuration.source_lock_digest, prompt_digest: report.configuration.prompt_digest,
      wan_emulation_digest: report.configuration.wan_emulation_digest, sampling: report.configuration.sampling,
      body: binding.numeric_body_fingerprint, tokens: run.tokens, output: run.generation.output_text_digest, build: e.build };
    same(identity, summary.identity, 'Confirmation summary identity changed');
    check(identity.source_lock_digest === plan.source_lock_digest && e.fixture.digest === identity.source_lock_digest &&
      result.model.source_lock_digest === identity.source_lock_digest &&
      e.comparison.source_digest === identity.source_lock_digest && Object.keys(e.comparison).length === 7 && Object.values(e.comparison).every(digest) &&
      e.comparison.workload_digest === result.workload_digest && digest(result.configuration_digest) && digest(result.environment.digest),
    'Confirmation source/comparison commitments changed');
    check(result.environment.digest === hash(Buffer.concat([Buffer.from('pllm.environment.v1\0'),
      Buffer.from(canonical(result.environment.attributes))])), 'Confirmation environment hash changed');
    const comparison = { identity, environment: result.environment, suite: e.suite_digest,
      ...select(e.comparison, ['numeric_digest', 'state_digest', 'workload_digest', 'output_digest']) };
    if (common) same(comparison, common, 'Confirmation cross-replicate environment/numeric/state/workload/output changed');
    common = comparison;
    if (states) same(audit.bindings, states, 'Confirmation logits/KV/history/RNG mismatch');
    states = audit.bindings;
    const records = new Map(result.metrics.map((metric) => [metric.id, metric]));
    check(records.size === result.metrics.length, 'Confirmation duplicate metric');
    const values = Object.fromEntries([...records].map(([id, metric]) => [id, metric.value]));
    const metricContracts = result.metrics.map(({ value, ...contract }) => {
      const unknown = value === null;
      check(contract.sample_count === Number(!unknown) && contract.origin === (unknown ? 'not_available' : 'measured') &&
        (unknown ? typeof contract.unavailable_reason === 'string' && contract.unavailable_reason.length > 0 : contract.unavailable_reason === null),
      'Confirmation metric availability changed');
      if (['network.covered_body_bytes', 'network.startup_body_bytes'].includes(contract.id)) {
        check(contract.measurement_scope === `covered_application_bodies:${name === 'maintained'
          ? 'prepared_protocol_lifecycle_controls_and_correction_acks_v3' : 'prepared_body_and_private_boundary_lifecycle_v4'}`, 'Confirmation body metric scope changed');
        return { ...contract, measurement_scope: 'explicit_cross_topology_body_coverage' };
      }
      return contract;
    });
    if (contracts) same(metricContracts, contracts, 'Confirmation metric owner/unit/window/denominator differs');
    contracts = metricContracts;
    for (const id of confirmationMetrics) {
      check(id === 'response.decode_tps' && output === 1 && values[id] === null ||
        typeof values[id] === 'number' && Number.isFinite(values[id]) && values[id] >= 0, 'Confirmation priority metric missing');
      check(records.get(id).unit === (id.endsWith('rss') || id.startsWith('network.') ? 'bytes'
        : id === 'response.decode_tps' ? 'per_second' : 'seconds'), 'Confirmation priority metric unit changed');
    }
    same(records.get('response.decode_tps').denominator, { unit: 'generated_tokens', value: output - 1 }, 'Confirmation decode denominator changed');
    check(output !== 1 || values['response.decode_tps'] === null, 'One-token decode TPS must remain unknown');
    equalNumbers(values['response.ttft'], run.durations.ttft_seconds, 'Confirmation TTFT differs from raw duration');
    equalNumbers(values['request.0.full'], run.durations.full_seconds, 'Confirmation request duration changed');
    if (output > 1) {
      equalNumbers(values['response.decode_tps'], (output - 1) / run.durations.generation_seconds, 'Confirmation N-1 decode timing changed');
      equalNumbers(values['response.decode_tps'], run.durations.tokens_per_second, 'Confirmation decode rate differs from raw duration');
    }
    for (const role of ['client', 'preparation']) equalNumbers(values[`${role}.cold_cpu`],
      report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role[role], 'Confirmation CPU window changed');
    equalNumbers(values['preparation.peak_rss'], run.processes.preparation.rss_peak_bytes, 'Confirmation preparation peak changed');
    same(e.details.timings.wall_seconds, [values['target.latency']], 'Confirmation target duration changed');
    const peak = values['process.peak_rss'], clientPeak = report.client_process_memory.after.lifetime_peak_rss_bytes;
    check(integer(peak) && integer(clientPeak) && clientPeak > 0 && peak >= clientPeak && clientPeak === values['client.peak_rss'] &&
      records.get('process.peak_rss').window === 'process_birth_through_target' &&
      records.get('process.peak_rss').measurement_scope === 'process_lifetime_high_water' &&
      report.client_process_memory.scope.includes('client and in-process dashboard') && report.memory_guard.tripped === false &&
      report.memory_guard.maximum_host_swap_growth_bytes === 0, 'Confirmation whole-worker RSS/swap evidence changed');
    const wanSamples = report.native_network_accounting.samples;
    const roles = ['client', 'inference', 'preparation', ...(name === 'maintained' ? [] : ['boundary_a', 'boundary_b'])];
    const expectedEdges = ['client', 'inference', 'preparation'].flatMap((a) => ['client', 'inference', 'preparation']
      .filter((b) => b !== a).map((b) => `${a}->${b}`));
    if (name !== 'maintained') for (const role of ['boundary_a', 'boundary_b']) expectedEdges.push(`client->${role}`, `${role}->client`);
    for (const snapshot of [wanSamples.startup.wan_emulation, wanSamples.runs[0].before.wan_emulation, wanSamples.runs[0].after.wan_emulation]) {
      helpers.checkedWan(snapshot);
      same(snapshot.role_parties, Object.fromEntries(roles.map((role) => [role, role])), 'Confirmation WAN role mapping changed');
      same(Object.keys(snapshot.parties).sort(), [...roles].sort(), 'Confirmation WAN parties missing');
      check(Object.keys(snapshot.directed_stream_bytes).every((edge) => expectedEdges.includes(edge)), 'Confirmation unknown WAN link');
      for (const role of roles) for (const [direction, index] of [['upload', 0], ['download', 1]]) {
        const admitted = snapshot.parties[role][direction].admitted_bytes;
        check(integer(admitted) && admitted === Object.entries(snapshot.directed_stream_bytes)
          .filter(([edge]) => edge.split('->')[index] === role).reduce((sum, [, value]) => sum + value, 0),
        'Confirmation WAN party counters differ from directed TCP');
      }
      check(snapshot.backend === 'native-shared-tcp-pacer' && snapshot.conditions_digest === identity.wan_emulation_digest &&
        Object.values(snapshot.parties).every((party) => party.upload.bytes_per_second === 5e6 && party.download.bytes_per_second === 12.5e6),
      'Confirmation native shared WAN rate changed');
    }
    const wan = wanSamples.runs[0].after.wan_emulation;
    same(Object.keys(wan.directed_stream_bytes).sort(), expectedEdges.sort(), 'Confirmation final WAN link missing');
    check(report.wan_readiness.emulation.native_stream_rate_snapshots_checked === true && wan.total_stream_bytes === values['network.all_link_stream_bytes'],
      'Confirmation WAN metric differs');
    const body = confirmationBodies(report, values, name !== 'maintained', early,
      experiment.pipeline.components.linear.params.prefill_chunk_rows ?? 16, helpers);
    const work = { output_tokens: output, sampled_selections: selections, executed_decode_steps: selections - 1, terminated_before_cap: early,
      ...select(run.privacy, ['preparation_rows', 'preparation_prefill_rows_issued', 'preparation_decode_rows_issued',
        'preparation_padding_rows_issued', 'inference_stage_calls', ...rowNames.slice(1).map((key) => `prepared_stage_rows_${key}`)]) };
    groups.get(name).push({ result, report, experiment, values, peak, wan, body, work });
  }
  check(checkedValues === summary.checked_values && summary.maximum_observed_swap_growth_bytes === 0 &&
    summary.state_commitment === hash(canonical(states, true)) && summary.long_output_cap_reached === (summary.identity.tokens.output_tokens === cap),
  'Confirmation summary state/swap/EOS totals changed');
  const rows = confirmationNames.map((name) => {
    const samples = groups.get(name).sort((a, b) => a.result.execution.replicate - b.result.execution.replicate), recorded = summary.groups[name];
    check(samples.length === 3, 'Confirmation replicate missing');
    const first = samples[0];
    for (const sample of samples) {
      same(sample.experiment, first.experiment, 'Confirmation recipe changed across replicates');
      same(sample.result.execution.comparison, first.result.execution.comparison, 'Confirmation arm comparison changed');
      // Resolved configuration includes each fresh worker's deployment root;
      // retain those bindings, but compare the common recorded case and pipeline.
      check(sample.result.configuration_digest === first.result.configuration_digest &&
        sample.report.experiment.pipeline_digest === first.report.experiment.pipeline_digest, 'Confirmation configuration/pipeline binding changed');
    }
    same(recorded.samples, samples.map((sample) => sample.values), 'Confirmation cached samples changed');
    same(recorded.executed_work, samples.map((sample) => sample.work), 'Confirmation cached executed work changed');
    same(Object.keys(recorded.medians).sort(), [...confirmationMetrics].sort(), 'Confirmation median metric set changed');
    for (const id of confirmationMetrics) same(recorded.medians[id], samples.some((sample) => sample.values[id] === null)
      ? null : median(samples.map((sample) => sample.values[id])), 'Confirmation cached median changed');
    const maximum = Math.max(...samples.map((sample) => sample.peak));
    check(recorded.maximum_client_peak_bytes === maximum, 'Confirmation hid whole-worker outlier');
    same(recorded.observed_tiers, Object.fromEntries([['low', 256 * 2 ** 20], ['medium', 2 * 2 ** 30], ['high', 16 * 2 ** 30]]
      .map(([tier, bound]) => [tier, maximum <= bound])), 'Confirmation tier labels changed');
    const pipeline = first.experiment.pipeline, placement = helpers.executionPlacement(pipeline, first.report.configuration.roles, first.report.client_body_placement);
    const outputs = first.work.output_tokens;
    const optional = (fn, divisor = 1) => samples.some((sample) => fn(sample) == null) ? null : median(samples.map(fn)) / divisor;
    const values = { ...Object.fromEntries(helpers.metrics.map(({ id }) => [id, null])),
      ...Object.fromEntries(Object.entries(axes).map(([axis, [id, divisor]]) => [axis, recorded.medians[id] === null ? null : recorded.medians[id] / divisor])),
      requestTps: median(samples.map(({ report }) => outputs / report.runs[0].durations.full_seconds)),
      clientNetwork: median(samples.map(({ wan }) => wan.client_stream_bytes)) / 1e6,
      networkPerToken: recorded.medians['network.all_link_stream_bytes'] / 1e6 / outputs,
      aggregateCpu: optional(({ report }) => report.process_cpu_accounting?.aggregate_cold_first_response_cpu_seconds),
      providerPeak: optional(({ report }) => helpers.providerPeak(report.configuration.roles, report.runs[0].processes)),
      preparationPeak: recorded.medians['preparation.peak_rss'] / 1e6,
      inferencePeak: optional(({ report }) => report.runs[0].processes?.inference?.rss_peak_bytes, 1e6),
      totalPeak: optional(({ report }) => report.native_process_memory?.complete === true ? report.native_process_memory.sampled_total_peak_rss_bytes : null, 1e6),
      bodyOffload: placement.clientMacPercent === null ? null : 100 - placement.clientMacPercent };
    return { id: name, label: name, file: spec.frozen_factory.file, fileSha256: spec.frozen_factory.sha256,
      configurationDigest: first.result.configuration_digest, sourceLock: summary.identity.source_lock_digest,
      pipeline, experiment: first.experiment, reference: name === 'maintained' ? 'prepared' : null,
      paper: null, publicPrefixTokens: 0, placement, metrics: values, sampleCount: 3,
      observedTiers: recorded.observed_tiers, executedWork: recorded.executed_work,
      clientMemory: clientMemoryEvidence(samples.map((sample) => sample.peak), 'process_birth_through_target'),
      provenance: { report: spec.report, reportSha256: spec.sha256, plan: spec.plan, planSha256: spec.plan_sha256 },
      buildDigest: first.result.execution.build.digest, frozenSources: plan.frozen_sources,
      comparisonCommitments: first.result.execution.comparison, recordedCase: first.result.execution.case, samplingContract: suite.sampling,
      experimentBindings: samples.map(({ result, report }) => ({ replicate: result.execution.replicate, ...report.experiment })),
      measurementContracts: first.result.metrics.map(({ value, ...contract }) => contract),
      replay: { originalDriver: spec.replay_file, originalDriverSha256: spec.replay_sha256,
        sourceSnapshots: structuredClone(snapshots ?? {}), requiresRegisteredSourceSnapshots: snapshots !== undefined,
        source: spec.report.split('/').at(-1).replace(/\.json$/, ''), case: name,
        callable: 'pllm.metrics.BenchmarkCase.from_spec', argument: 'recordedCase', requiresOriginalBuildAndLifecycle: true },
      scopedTimings: { completeTargetSeconds: recorded.medians['target.latency'] } };
  });
  const winner = confirmationNames[2], control = confirmationNames[1], reconciled = [];
  for (let rep = 0; rep < 3; rep++) {
    const a = groups.get(control)[rep].body, b = groups.get(winner)[rep].body;
    for (const key of ['prepared', 'fixed', 'boundary']) same(a[key], b[key], 'Confirmation fixed boundary-arm work changed');
    same(Object.keys(a.stages).sort(), Object.keys(b.stages).sort(), 'Confirmation fixed stage set changed');
    let variation = 0;
    for (const stage of Object.keys(a.stages)) {
      for (const edge of Object.keys(a.stages[stage])) if (edge !== 'inference->client') same(a.stages[stage][edge], b.stages[stage][edge], 'Confirmation fixed stage work changed');
      variation += Math.abs(b.stages[stage]['inference->client'] - a.stages[stage]['inference->client']);
    }
    check(a.bound === b.bound && variation <= a.bound && b.head < a.head, 'Confirmation server_ns bound/head reply changed');
    const total = b.total - a.total, head = b.head - a.head, controls = b.controls - a.controls, metadata = b.reply - a.reply;
    reconciled.push({ replicate: rep, covered_body_delta_bytes: total, head_reply_delta_bytes: head,
      lifecycle_control_delta_bytes: controls, prepared_stage_reply_delta_bytes: metadata,
      prepared_stage_reply_metadata_compatibility_bound_bytes: a.bound,
      bootstrap_and_metadata_residual_bytes: total - head - controls - metadata });
  }
  same(summary.body_reconciliation, { [winner]: reconciled }, 'Confirmation cached body savings changed');
  const comparisons = {};
  for (const name of [control, 'maintained']) {
    comparisons[name] = {};
    for (const id of confirmationMetrics) {
      const left = summary.groups[name].medians[id], right = summary.groups[winner].medians[id];
      const known = left !== null && right !== null, min = id !== 'response.decode_tps';
      comparisons[name][id] = { control: left, candidate: right, delta: known ? right - left : null,
        improvement_fraction: known && left ? (min ? 1 - right / left : right / left - 1) : null,
        candidate_wins: known ? (min ? right < left : right > left) : null,
        paired_deltas: groups.get(name).map((sample, rep) => sample.values[id] === null || groups.get(winner)[rep].values[id] === null
          ? null : groups.get(winner)[rep].values[id] - sample.values[id]) };
    }
  }
  equalNumbers(summary.comparisons, { [winner]: comparisons }, 'Confirmation cached comparison/win changed');
  const first = groups.get('maintained')[0], pipeline = rows[0].pipeline, output = first.work.output_tokens;
  return [{ ...spec, baseline: 'maintained', model: pipeline.model.source,
    identity: { model: pipeline.model.source, source: pipeline.model, body: summary.identity.body,
      prompt: summary.identity.prompt_digest, quantization: pipeline.components.quantization, inputs: 150, outputs: output, cap,
      warm: false, output: summary.identity.output, nonlinear: null, state: null, verification: null,
      sampling: first.report.runs[0].sampling, backend: 'native', wan: summary.identity.wan_emulation_digest,
      links: null, wanBackend: 'native-shared-tcp-pacer', clientIsolation: true },
    outputWorkload: { emittedTokens: output, outputCap: cap, sampledSelections: first.work.sampled_selections,
      executedDecodeSteps: first.work.executed_decode_steps, terminatedBeforeCap: first.work.terminated_before_cap },
    measurement: { requestScope: 'single audited cold request', phase: 'cold', sampleCount: 3,
      clientRssLabel: 'Client RAM · complete worker lifetime', clientCpuLabel: 'Client CPU · cold first response',
      preparationCpuLabel: 'Prep CPU · birth to cold first response',
      cpu: 'SDK, supervisor, dashboard, native relay and full-state audit charged to fresh worker',
      rss: 'Largest finalized complete-worker lifetime peak across all three replicates qualifies observed tiers; admission unchanged',
      timing: 'Online TTFT and actual emitted N−1 decode; natural EOS selection is not emitted output; one-token TPS unknown',
      traffic: 'Source-present original bodies and directed TCP streams; variable MessagePack server_ns stays charged; no physical wire or checkpoint/tokenizer distribution claim' }, rows }];
}
