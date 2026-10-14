/** Hash-bound original SDK evidence plus its reproducible, read-only validation. */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { clientMemoryEvidence } from './research-memory.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const check = (condition, message) => { if (!condition) throw new Error(message); };
const median = (values) => {
  check(values.length > 0 && values.every((value) => Number.isFinite(value) && value >= 0), 'Invalid boundary observation');
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export function privateBoundaryCohorts(spec, document, read, helpers) {
  const raw = read(spec.summary), source = read(spec.summary_source), controlRaw = read(spec.control);
  const summary = JSON.parse(raw), control = JSON.parse(controlRaw);
  check(hash(raw) === spec.summary_sha256 && hash(source) === spec.summary_source_sha256 &&
    hash(controlRaw) === spec.control_sha256 && summary.source_sha256 === spec.sha256 &&
    summary.summarizer_sha256 === spec.summary_source_sha256 && summary.control_sha256 === spec.control_sha256,
  'Private-boundary summary, validator or control source changed');
  check(summary.schema === 'pllm.private_boundary_sdk_summary.v1' &&
    summary.evidence_class === 'matched-source-present-sdk-screen' && spec.evidenceTier === 'screen' &&
    document.complete === true && document.tiny === false && document.source_case === 'source-present' &&
    document.wan_requested === true && document.replicates >= 3 && document.planned_cells === 2 * document.replicates &&
    document.results.length === document.planned_cells && summary.replicates === document.replicates &&
    summary.output_tokens === 8 && summary.output_cap === 8 && summary.maximum_swap_growth_bytes === 0 &&
    document.swap.maximum_growth_bytes === 0 && summary.cohort_commitment === document.cohort_commitment &&
    summary.source_lock_digest === document.source_lock_digest && isDeepStrictEqual(summary.build, document.build) &&
    isDeepStrictEqual(summary.verification, document.verification) && summary.verification.exact_full_state_parity === true &&
    isDeepStrictEqual(summary.driver_implementation, document.driver_implementation), 'Incomplete or changed boundary cohort');
  for (const [file, expected] of Object.entries(summary.driver_implementation)) {
    check(hash(read(`scripts/${file}`)) === expected, 'Original boundary measurement driver changed');
  }
  const rows = ['baseline', 'coupled'].map((variant) => {
    const group = summary.groups[variant], samples = document.results.filter((cell) => cell.variant === variant);
    check(samples.length === document.replicates && group.sample_count === document.replicates &&
      isDeepStrictEqual(samples.map((cell) => cell.replicate).sort(), Array.from({ length: document.replicates }, (_, i) => i)),
    'Missing boundary replicate');
    const experiment = group.experiment, pipeline = experiment.pipeline;
    const expected = structuredClone(control);
    expected.budget = { requests: 1, max_input_tokens: summary.input_tokens, max_new_tokens: 8 };
    if (variant === 'coupled') {
      expected.pipeline.components.boundary = { component: 'pllm/private-lookup-full-head/v1', params: {} };
      expected.pipeline.components.topology = { component: 'pllm/prepared-body-two-online-boundary/v1', params: {} };
    }
    check(isDeepStrictEqual(experiment, expected) && document.memory_admission[`${summary.input_tokens}/${variant}`].admitted === true &&
      document.memory_admission[`${summary.input_tokens}/${variant}`].topology.estimate.pipeline_digest === group.pipeline_digest,
    'Boundary recipe differs from original admission');
    const from = (select) => median(samples.map(select));
    const measured = {
      request_seconds: from((cell) => cell.client.responses[0].request_seconds),
      first_text_seconds: from((cell) => cell.client.responses[0].first_text_seconds),
      text_delta_decode_tps: from((cell) => cell.client.responses[0].decode_tps),
      complete_case_seconds: from((cell) => cell.complete_case_seconds),
      client_request_cpu_seconds: from((cell) => cell.client.responses[0].client_request_cpu_seconds),
      client_lifetime_cpu_seconds: from((cell) => cell.client.client_lifetime_cpu_seconds),
      aggregate_cleanup_cpu_seconds: from((cell) => cell.aggregate_cleanup_cpu_seconds),
      preparation_birth_cpu_seconds: from((cell) => cell.provider_birth_cpu_seconds.preparation),
      client_subprocess_peak_rss_bytes: from((cell) => cell.client.client_peak_rss_bytes),
      preparation_sampled_peak_rss_bytes: from((cell) => cell.provider_sampled_peak_rss_bytes.preparation),
      inference_sampled_peak_rss_bytes: from((cell) => cell.provider_sampled_peak_rss_bytes.inference),
      provider_sampled_peak_rss_bytes: from((cell) => Math.max(...Object.values(cell.provider_sampled_peak_rss_bytes))),
      all_link_tcp_stream_bytes: from((cell) => cell.wan.total_stream_bytes),
      client_tcp_stream_bytes: from((cell) => cell.wan.client_stream_bytes),
      covered_body_bytes: from((cell) => cell.client.ledger.all_link_serialized_body_bytes),
      online_body_bytes: from((cell) => cell.client.ledger.online_all_link_serialized_body_bytes),
      bundle_delivery_bytes: from((cell) => cell.client.audit.bundle_network_bytes),
    };
    check(isDeepStrictEqual(measured, group.medians) && group.whole_client_peak_rss_bytes === null &&
      group.whole_client_memory_tier_qualified === false && group.maximum_client_subprocess_peak_rss_bytes ===
        Math.max(...samples.map((cell) => cell.client.client_peak_rss_bytes)), 'Boundary metrics or owner scope changed');
    const values = { ...Object.fromEntries(helpers.metrics.map(({ id }) => [id, null])),
      full: measured.request_seconds, ttft: measured.first_text_seconds, decodeTps: measured.text_delta_decode_tps,
      requestTps: from((cell) => 8 / cell.client.responses[0].request_seconds),
      aggregateCpu: measured.aggregate_cleanup_cpu_seconds, clientCpu: measured.client_lifetime_cpu_seconds,
      preparationCpu: measured.preparation_birth_cpu_seconds, clientPeak: measured.client_subprocess_peak_rss_bytes / 1e6,
      preparationPeak: measured.preparation_sampled_peak_rss_bytes / 1e6, inferencePeak: measured.inference_sampled_peak_rss_bytes / 1e6,
      providerPeak: measured.provider_sampled_peak_rss_bytes / 1e6,
      covered: measured.covered_body_bytes / 1e6 / 8, online: measured.online_body_bytes / 1e6 / 8,
      totalNetwork: measured.all_link_tcp_stream_bytes / 1e6, clientNetwork: measured.client_tcp_stream_bytes / 1e6,
      networkPerToken: measured.all_link_tcp_stream_bytes / 1e6 / 8 };
    const roles = Object.keys(samples[0].wan.role_parties);
    return { id: variant, label: variant === 'baseline' ? 'Local tied boundary control' : 'Private lookup + full head',
      file: spec.frozen_factory.file, fileSha256: spec.frozen_factory.sha256,
      configurationDigest: group.configuration_digest, sourceLock: summary.source_lock_digest, pipeline, experiment,
      reference: variant === 'baseline' ? 'prepared' : null, paper: null, publicPrefixTokens: 0,
      placement: helpers.executionPlacement(pipeline, roles), metrics: values, sampleCount: samples.length,
      // The measured subprocess peak omits a separate client-owned relay/parent.
      clientMemory: clientMemoryEvidence(samples.map(() => null), 'whole_client_unknown_parent_relay_excluded'),
      observedClientSubprocessPeakBytes: group.maximum_client_subprocess_peak_rss_bytes,
      provenance: { report: spec.report, reportSha256: spec.sha256, plan: null, planSha256: null },
      buildDigest: summary.build.expected_digest, frozenSources: summary.driver_implementation,
      replay: { originalDriver: spec.replay_file, originalDriverSha256: spec.replay_sha256,
        source: spec.report.split('/').at(-1).replace(/\.json$/, ''), case: variant,
        callable: 'recorded_contract', requiresOriginalBuildAndLifecycle: true },
      scopedTimings: { completeCellSeconds: measured.complete_case_seconds,
        clientRequestCpuSeconds: measured.client_request_cpu_seconds } };
  });
  const pipeline = rows[0].pipeline;
  return [{ ...spec, baseline: 'baseline', model: pipeline.model.source,
    identity: { model: pipeline.model.source, source: pipeline.model, body: summary.body_fingerprint,
      prompt: summary.cohort_commitment, quantization: pipeline.components.quantization,
      inputs: summary.input_tokens, outputs: 8, cap: 8, warm: false, output: summary.output_commitment,
      nonlinear: null, state: null, verification: null, sampling: document.sampler_fixture,
      backend: 'native', wan: summary.wan_digest, links: null, wanBackend: 'native-shared-tcp-pacer', clientIsolation: true },
    measurement: { requestScope: 'single cold request', phase: 'cold', sampleCount: summary.replicates,
      clientRssLabel: 'Client subprocess RAM · relay excluded', clientCpuLabel: 'Client child CPU · lifetime',
      preparationCpuLabel: 'Prep CPU · birth to client exit',
      metricLabels: { ttft: 'First text · setup included', decodeTps: 'Decode · N−1 text events',
        aggregateCpu: 'Aggregate CPU · startup through cleanup',
        preparationPeak: 'Prep RAM · post-startup samples', inferencePeak: 'Inference RAM · post-startup samples',
        providerPeak: 'Provider RAM · post-startup samples' },
      cpu: 'Aggregate CPU includes parent/relay and all reaped children through cleanup; client CPU is child-only',
      rss: 'Client subprocess lifetime peak excludes parent/relay; whole-client RAM tiers and aggregate RSS remain unknown; provider samples exclude startup peaks',
      timing: 'Request first text includes setup; eight text events match eight authoritative outputs; full logits/KV audit is charged equally',
      traffic: 'Source-present complete-cell TCP streams; checkpoint/tokenizer distribution and physical wire unknown' }, rows }];
}
