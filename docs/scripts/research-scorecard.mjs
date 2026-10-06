/** Evidence-backed scorecards. Never combine independently salted cohorts. */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';
import { defaultMetrics } from '../lib/research-rankings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const hash = (value) => createHash('sha256').update(value).digest('hex');

export const metrics = [
  { id: 'covered', label: 'Covered bodies / output token', unit: 'MB/token', direction: 'min' },
  { id: 'online', label: 'Online bodies / output token', unit: 'MB/token', direction: 'min' },
  { id: 'decodeBytes', label: 'Decode bodies / later output', unit: 'MB/token', direction: 'min' },
  { id: 'full', label: 'Full request latency', unit: 's', direction: 'min' },
  { id: 'ttft', label: 'Time to first token', unit: 's', direction: 'min' },
  { id: 'requestTps', label: 'Full request throughput', unit: 'tokens/s', direction: 'max' },
  { id: 'decodeTps', label: 'Decode throughput (N−1)', unit: 'tokens/s', direction: 'max' },
  { id: 'aggregateCpu', label: 'Cold aggregate process CPU', unit: 'CPU s', direction: 'min' },
  { id: 'clientCpu', label: 'Cold client process CPU', unit: 'CPU s', direction: 'min' },
  { id: 'bodyOffload', label: 'Body-linear compute offloaded (plan)', unit: '%', direction: 'max', scoreBounds: [0, 100] },
  { id: 'providerPeak', label: 'Largest provider process peak RSS', unit: 'MB', direction: 'min' },
  { id: 'preparationPeak', label: 'Preparation lifetime peak RSS', unit: 'MB', direction: 'min' },
  { id: 'inferencePeak', label: 'Inference lifetime peak RSS', unit: 'MB', direction: 'min' },
  { id: 'artifactBytes', label: 'Pre-positioned public artifacts', unit: 'MB', direction: 'min' },
  { id: 'clientPeak', label: 'Isolated client peak RSS', unit: 'MB', direction: 'min' },
  { id: 'totalPeak', label: 'Sampled aggregate live-process peak RSS', unit: 'MB', direction: 'min' },
  { id: 'clientNetwork', label: 'Cold client TCP-stream traffic', unit: 'MB', direction: 'min' },
  { id: 'totalNetwork', label: 'Cold all-link TCP-stream traffic', unit: 'MB', direction: 'min' },
  { id: 'networkPerToken', label: 'Cold all-link TCP traffic / output token', unit: 'MB/token', direction: 'min' },
  { id: 'wire', label: 'Full physical wire / output', unit: 'MB/token', direction: 'min' },
  { id: 'quality', label: 'Representative generation quality', unit: 'score', direction: 'max' },
  { id: 'energy', label: 'Whole-response energy', unit: 'J', direction: 'min' },
];

const shortLabels = {
  requestTps: 'Request speed', decodeTps: 'Decode speed', covered: 'Covered traffic',
  aggregateCpu: 'Agg CPU', providerPeak: 'Provider RAM', online: 'Online traffic',
  decodeBytes: 'Decode traffic', full: 'Request latency', ttft: 'First token',
  clientCpu: 'Client CPU', bodyOffload: 'Body offload', preparationPeak: 'Preparation RAM', inferencePeak: 'Inference RAM',
  artifactBytes: 'Public artifacts', clientPeak: 'Client RAM', wire: 'Physical wire',
  quality: 'Quality', energy: 'Energy',
  totalPeak: 'Agg RAM', clientNetwork: 'Client network', totalNetwork: 'Agg network', networkPerToken: 'MB / token',
};
const metricGroups = {
  clientCpu: 'Client', bodyOffload: 'Client', clientPeak: 'Client', clientNetwork: 'Client',
  aggregateCpu: 'Agg', totalPeak: 'Agg', totalNetwork: 'Agg', energy: 'Agg', artifactBytes: 'Agg',
  providerPeak: 'Providers', preparationPeak: 'Providers', inferencePeak: 'Providers',
};
for (const metric of metrics) {
  metric.shortLabel = shortLabels[metric.id];
  metric.group = metricGroups[metric.id] ?? 'Performance';
}

export function baselineKind(spec, pipeline, placement, isBaseline) {
  const kind = spec.reference ?? (isBaseline ? 'prepared' : null);
  if (kind === 'prepared' && placement.topology !== 'Prepared' ||
      kind === 'client' && placement.topology !== 'Client-only') throw new Error('Baseline role graph mismatch');
  if (kind === 'offset') {
    const linear = pipeline.components.linear;
    if (placement.topology !== 'Two workers' || placement.kernel !== 'CPU' ||
        linear?.component !== 'pllm/two-online-offset-linear/v1' ||
        ![undefined, 'full'].includes(linear.params?.input_encoding) ||
        ![undefined, 'full'].includes(linear.params?.output_encoding) ||
        ![undefined, 'sequential'].includes(linear.params?.dispatch)) {
      throw new Error('Naive baseline must use unoptimized CPU offset workers');
    }
  }
  if (kind !== null && !['prepared', 'client', 'offset'].includes(kind)) throw new Error('Unknown baseline kind');
  return kind;
}

function clientBodyMacPercent(roles, body) {
  const clientOnly = roles.length === 1 && roles[0] === 'client';
  let percent = clientOnly ? 100 : null;
  if (body != null) {
    const local = body.declared_body_linear_macs_per_row_client;
    const remote = body.declared_body_linear_macs_per_row_remote;
    if (body.schema !== 'pllm.client_body_placement.v1' ||
        !Number.isSafeInteger(local) || !Number.isSafeInteger(remote) || local < 0 || remote < 0 ||
        !Number.isSafeInteger(local + remote)) {
      throw new Error('Invalid declared client body placement');
    }
    // Historical bundle samples omit the separate client-only engine's weights.
    if (!clientOnly && local + remote > 0) percent = 100 * local / (local + remote);
  }
  return percent;
}

export function executionPlacement(pipeline, roles, body = null) {
  const components = pipeline.components;
  const declared = [...roles].sort().join(',');
  const topology = {
    client: 'Client-only',
    'client,inference,preparation': 'Prepared',
    'client,worker_a,worker_b': 'Two workers',
  }[declared];
  if (!topology) throw new Error('Unsupported measured role graph');
  const client = components.placement;
  const clientBody = topology === 'Client-only' || Boolean(client);
  const kernel = { 'pllm/cpu': 'CPU', 'pllm/apple-metal-int8/v1': 'CPU + Metal' }[components.kernels?.component];
  if (!kernel) throw new Error('Unsupported measured kernel');
  const ownership = topology === 'Client-only' ? 'Full decoder at client' : client
    ? 'Selected body linear stages at client' : 'Body linear stages at providers';
  const clientMacPercent = clientBodyMacPercent(roles, body);
  return { topology, kernel, clientBody, ownership, clientMacPercent };
}

export function providerPeak(roles, processes) {
  const peaks = roles.filter((role) => role !== 'client').map((role) => processes?.[role]?.rss_peak_bytes);
  if (peaks.some((value) => value == null)) return null;
  if (peaks.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error('Invalid provider peak RSS');
  }
  return Math.max(0, ...peaks) / 1e6;
}

export function clientAndTotalResources(report) {
  const isolated = report.configuration.client_process_isolated === true;
  const memory = report.native_process_memory;
  const network = report.native_network_accounting?.samples?.runs?.at(-1)?.after?.wan_emulation;
  let clientNetwork = null, totalNetwork = null;
  if (network) {
    if (network.schema !== 'pllm.native_wan.v1' || network.enforced !== true ||
        report.wan_readiness?.emulation?.native_stream_rate_snapshots_checked !== true ||
        network.conditions_digest !== report.configuration.wan_emulation_digest) {
      throw new Error('Native network scores require checked shared access caps');
    }
    let total = 0, client = 0;
    for (const [link, bytes] of Object.entries(network.directed_stream_bytes)) {
      const roles = link.split('->');
      if (roles.length !== 2 || roles.some((role) => !report.configuration.roles.includes(role)) ||
          !Number.isSafeInteger(bytes) || bytes < 0) throw new Error('Invalid native network counter');
      total += bytes;
      if (roles.includes('client')) client += bytes;
    }
    if (total !== network.total_stream_bytes || client !== network.client_stream_bytes) {
      throw new Error('Native network link conservation failed');
    }
    clientNetwork = client / 1e6;
    totalNetwork = total / 1e6;
  }
  return {
    clientPeak: isolated && report.client_process_memory?.after?.lifetime_peak_rss_bytes != null
      ? report.client_process_memory.after.lifetime_peak_rss_bytes / 1e6 : null,
    totalPeak: isolated && memory?.schema === 'pllm.native_process_memory.v1' && memory.complete === true && memory.samples > 0 && memory.sampled_total_peak_rss_bytes != null
      ? memory.sampled_total_peak_rss_bytes / 1e6 : null,
    clientNetwork, totalNetwork,
    networkPerToken: totalNetwork === null ? null : totalNetwork / report.runs[0].tokens.output_tokens,
  };
}

export function reportMetrics(report, artifactBytes = null) {
  if (report.runs?.length !== 1 || report.warmup_runs?.length) throw new Error('Require single-response metric observations');
  const run = report.runs[0], config = report.configuration;
  const comm = report.communication_per_token.summary, cpu = report.process_cpu_accounting;
  const clientMacPercent = clientBodyMacPercent(config.roles, report.client_body_placement);
  const values = {
    covered: comm.setup_inclusive_mb_per_output_token, online: comm.online_mb_per_output_token,
    decodeBytes: comm.decode_online_mb_per_output_token, full: run.durations.full_seconds,
    ttft: run.durations.ttft_seconds,
    requestTps: run.durations.full_seconds > 0 ? run.tokens.output_tokens / run.durations.full_seconds : null,
    decodeTps: run.tokens.output_tokens > 1 && run.durations.generation_seconds > 0
      ? (run.tokens.output_tokens - 1) / run.durations.generation_seconds : null,
    aggregateCpu: cpu?.aggregate_cold_first_response_cpu_seconds ?? null,
    clientCpu: cpu?.cold_first_response_cpu_seconds_by_role?.client ?? null,
    bodyOffload: clientMacPercent === null ? null : 100 - clientMacPercent,
    providerPeak: providerPeak(config.roles, run.processes),
    preparationPeak: run.processes?.preparation?.rss_peak_bytes == null ? null : run.processes.preparation.rss_peak_bytes / 1e6,
    inferencePeak: run.processes?.inference?.rss_peak_bytes == null ? null : run.processes.inference.rss_peak_bytes / 1e6,
    artifactBytes: artifactBytes === null ? null : artifactBytes / 1e6,
    ...clientAndTotalResources(report), wire: null, quality: null, energy: null,
  };
  for (const value of Object.values(values)) {
    if (value !== null && (!Number.isFinite(value) || value < 0)) throw new Error('Invalid metric');
  }
  return values;
}

export function cohortRows(cohort, document, read = (file) => fs.readFileSync(path.join(root, file))) {
  if (!Array.isArray(document.candidates) || document.candidates.length < 2) {
    throw new Error('Require a canonical multi-Experiment cohort with its shared salt');
  }
  if (document.checks?.passed !== true) throw new Error('Comparison policy did not admit rankings');
  if (document.analysis_of) {
    const original = read(document.analysis_of.file);
    let observations = JSON.parse(original).candidates;
    const selected = document.analysis_of.candidates;
    if (selected !== undefined) {
      if (!Array.isArray(selected) || selected.length < 2 ||
          selected.some((name) => typeof name !== 'string') || new Set(selected).size !== selected.length) {
        throw new Error('Invalid reanalysis cohort selection');
      }
      observations = observations.filter((candidate) => selected.includes(candidate.name));
      if (observations.length !== selected.length) throw new Error('Reanalysis selection is missing observations');
    }
    if (hash(original) !== document.analysis_of.sha256 ||
        !isDeepStrictEqual(observations, document.candidates)) {
      throw new Error('Reanalysis changed original measurements');
    }
  }
  for (const [file, expected] of Object.entries(cohort.configuration_sources ?? {})) {
    if (hash(read(file)) !== expected) throw new Error(`Measured configuration source changed: ${file}`);
  }
  const publisher = cohort.publisher_report ? JSON.parse(read(cohort.publisher_report)) : null;
  const calibration = cohort.calibration_report ? JSON.parse(read(cohort.calibration_report)) : null;
  const identities = new Set();
  const sourceLocks = new Map();
  const names = new Set();
  const rows = document.candidates.map((candidate) => {
    if (names.has(candidate.name)) throw new Error('Duplicate candidate');
    names.add(candidate.name);
    const spec = cohort.candidates[candidate.name];
    if (!spec || !spec.file.endsWith('.py')) throw new Error('Every score needs its Python configuration');
    const fileSha256 = hash(read(spec.file));
    if (fileSha256 !== spec.sha256) throw new Error(`Measured Python configuration changed: ${spec.file}`);
    const report = candidate.report;
    if (!report.checks?.passed || report.runs?.length !== 1 || report.warmup_runs?.length) {
      throw new Error('Require checked single-response, zero-warmup evidence');
    }
    const run = report.runs[0], config = report.configuration;
    if (run.status !== 'completed' || !run.tokens.authoritative || !run.generation?.output_text_digest ||
        !run.model_id || !run.model_fingerprint || !config.prompt_digest || !config.source_lock_digest || !run.tokens.output_tokens) {
      throw new Error('Incomplete authoritative identity');
    }
    const identity = {
      model: run.model_id, source: candidate.pipeline.model, body: run.model_fingerprint, prompt: config.prompt_digest,
      quantization: candidate.pipeline.components.quantization, inputs: run.tokens.input_tokens,
      verification: candidate.pipeline.components.verification ?? null,
      outputs: run.tokens.output_tokens, cap: run.max_output_tokens, sampling: run.sampling,
      warm: run.warm, output: run.generation.output_text_digest, backend: config.provider_backend,
      wan: config.wan_emulation_digest ?? null, links: config.link_conditions_digest ?? null,
      wanBackend: config.wan_emulation ? report.native_network_accounting ? 'native-shared-tcp-pacer'
        : report.docker_accounting ? 'linux-tbf-routed-party-ports' : 'unknown' : null,
      clientIsolation: config.client_process_isolated === true,
    };
    identities.add(JSON.stringify(identity));
    const roles = [...config.roles].sort().join(',');
    if (sourceLocks.has(roles) && sourceLocks.get(roles) !== config.source_lock_digest) {
      throw new Error('Source lock changed within a role graph');
    }
    sourceLocks.set(roles, config.source_lock_digest);
    let artifactBytes = 0, publicPrefixTokens = 0;
    for (const [slot, key, implementation] of [
      ['tokenizer', 'tokenizer', 'pllm/indexed-tokenizer/v1'],
      ['public_prefix', 'capsule', 'pllm/public-prefix-capsule/v1'],
    ]) {
      const selected = candidate.pipeline.components[slot];
      if (!selected) continue;
      const artifact = publisher?.[key];
      if (publisher?.schema !== 'pllm.public_artifact_build.v1' ||
          publisher.body_fingerprint !== run.model_fingerprint ||
          publisher.input_tokens !== run.tokens.input_tokens || publisher.output_cap !== run.max_output_tokens ||
          selected.component !== implementation || selected.params.digest !== artifact?.digest ||
          !Number.isSafeInteger(artifact?.bytes) || artifact.bytes <= 0) {
        throw new Error('Public artifact does not match publisher evidence and measured Pipeline');
      }
      artifactBytes += artifact.bytes;
      if (slot === 'public_prefix') {
        publicPrefixTokens = publisher.public_prefix_tokens;
        if (selected.params.size_bytes !== artifact.bytes || !Number.isSafeInteger(publicPrefixTokens) ||
            publicPrefixTokens <= 0 || publicPrefixTokens > run.tokens.input_tokens) {
          throw new Error('Invalid public-prefix artifact bounds');
        }
      }
    }
    const quantization = candidate.pipeline.components.quantization;
    if (quantization?.component === 'pllm/public-per-channel-equalized/v1') {
      if (calibration?.schema !== 'pllm.public_equalization_build.v1' ||
          calibration.profile_digest !== quantization.params.profile_digest ||
          calibration.source_lock_digest !== config.source_lock_digest ||
          !isDeepStrictEqual(calibration.model, candidate.pipeline.model) ||
          !Number.isSafeInteger(calibration.profile_bytes) || calibration.profile_bytes <= 0) {
        throw new Error('Public calibration artifact does not match measured numeric source');
      }
      // Unique public profile bytes; provider distribution is outside this
      // response's counters. Client-delivered scales are already metered.
      artifactBytes += calibration.profile_bytes;
    }
    if (spec.artifact_bytes !== artifactBytes || (spec.public_prefix_tokens ?? 0) !== publicPrefixTokens) {
      throw new Error('Public artifact cost or prefix scope differs from measured evidence');
    }
    const metricValues = reportMetrics(report, artifactBytes);
    const placement = executionPlacement(candidate.pipeline, config.roles, report.client_body_placement);
    return { id: candidate.name, label: spec.label, file: spec.file, fileSha256,
      configurationDigest: candidate.configuration_digest, pipeline: candidate.pipeline,
      sourceLock: config.source_lock_digest, paper: spec.paper ?? null,
      reference: baselineKind(spec, candidate.pipeline, placement, candidate.name === cohort.baseline),
      publicPrefixTokens, placement, metrics: metricValues };
  });
  if (identities.size !== 1) throw new Error('Cohort identity, numeric contract, or outputs differ');
  if (!names.has(cohort.baseline) || names.size !== Object.keys(cohort.candidates).length) {
    throw new Error('Baseline or configured score missing');
  }
  return { ...cohort, candidates: undefined, model: document.candidates[0].report.runs[0].model_id,
    identity: JSON.parse([...identities][0]), rows };
}

export function buildScorecard() {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/benchmark-cohorts.json')));
  const implementations = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/implementations.json'))).papers;
  const library = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/paper-library.json'))).papers;
  const registry = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/research/papers.json'))).papers;
  const papers = library.map((paper) => ({ id: paper.id, title: paper.title,
    slug: paper.registry_id ? registry.find((item) => item.id === paper.registry_id).slug.replaceAll('_', '-') : paper.id,
    status: implementations[paper.id]?.status ?? 'Not implemented',
  }));
  const cohorts = manifest.cohorts.map((cohort) => {
      const raw = fs.readFileSync(path.join(root, cohort.report));
      const value = cohortRows(cohort, JSON.parse(raw));
      for (const row of value.rows) {
        if (row.paper && !implementations[row.paper]) throw new Error('Scored paper has no implementation record');
        row.papers = papers.flatMap((paper) => {
          if (row.paper === paper.id) return [{ id: paper.id, kind: 'adaptation', scope: implementations[paper.id].scope }];
          const links = implementations[paper.id]?.scorecard_components ?? [];
          return links.filter((link) => Object.values(row.pipeline.components).some((component) => component.component === link.component))
            .map((link) => ({ id: paper.id, kind: 'related', scope: link.scope }));
        });
      }
      return { ...value, reportSha256: hash(raw) };
    });
  return { schema: 'pllm.research_scorecard.v1', metrics, defaultMetrics, papers, cohorts };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const target = path.join(root, 'docs/data/research/scorecard.json');
  fs.writeFileSync(target, JSON.stringify(buildScorecard(), null, 2) + '\n');
}
