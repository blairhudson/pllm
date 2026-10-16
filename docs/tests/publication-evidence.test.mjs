import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { buildPublicationEvidence, renderPublicationEvidence, manuscriptTables } from '../../scripts/audit_publication_evidence.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = file => fs.readFileSync(path.join(root, file));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const registryFile = 'docs/data/research/benchmark-cohorts.json';
const contractFile = 'paper/comparison-contract.v1.json';
const tierId = 'qwen25-private-boundary-tiers-150-8';
const confirmationId = 'qwen25-private-boundary-decode-confirmation-150-8';
const candidate = 'coupled-parallel-uniform_residues';
const originalResults = (cohort, recipe) => JSON.parse(read(cohort.report)).results
  .filter(result => result.execution.case.name === recipe).sort((a, b) => a.execution.replicate - b.execution.replicate);

// Rehash only in-memory fixtures so semantic controls get past byte-integrity
// checks. These transformed records are never written or treated as measurements.
function changedOriginal(id, mutate) {
  const registry = JSON.parse(read(registryFile)), spec = registry.adapted_cohorts.find(row => row.id === id);
  const original = JSON.parse(read(spec.report));
  mutate(original);
  const bytes = Buffer.from(JSON.stringify(original));
  spec.sha256 = hash(bytes);
  return buildPublicationEvidence(file => file === registryFile ? Buffer.from(JSON.stringify(registry))
    : file === spec.report ? bytes : read(file));
}

test('publication tables preserve original cohorts, raw replicates and largest complete peaks', () => {
  const evidence = buildPublicationEvidence(read);
  assert.deepEqual(evidence, JSON.parse(read('paper/artifacts/historical-results-v1.json')));
  assert.equal(renderPublicationEvidence(evidence), read('paper/HISTORICAL-RESULTS.md').toString());
  for (const [file, table] of Object.entries(manuscriptTables(evidence))) {
    assert.ok(read(file).toString().includes(table), `Canonical table changed: ${file}`);
  }
  assert.equal(new Set(evidence.cohorts.map(c => c.privateCohortCommitment)).size, 4);
  assert.deepEqual(evidence.cohorts.map(c => c.rows.length), [8, 3, 3, 3]);
  for (const cohort of evidence.cohorts) {
    for (const row of cohort.rows) {
      const results = originalResults(cohort, row.recipe);
      const peaks = results.map(r => r.metrics.find(m => m.id === 'process.peak_rss').value);
      assert.equal(row.maximumCompleteClientBytes, Math.max(...peaks));
      assert.equal(row.observedLow, Math.max(...peaks) <= 268435456);
      assert.equal(row.sampleCount, 3);
      assert.equal(row.metrics['network.all_link_stream_bytes'].samples.length, 3);
      assert.deepEqual(row.replicateIds, [0, 1, 2]);
    }
  }
});

test('raw provider bytes preserve median, maximum, all roles and missing boundary observations', () => {
  const evidence = buildPublicationEvidence(read), first = evidence.cohorts[0].rows[0].providerPeakBytes;
  assert.deepEqual(first.samples, [1077248000, 1085276160, 1009123328]);
  assert.equal(first.median, 1077248000);
  assert.equal(first.maximum, 1085276160);
  assert.notEqual(first.median, first.maximum);
  for (const cohort of evidence.cohorts) for (const row of cohort.rows) {
    const observations = originalResults(cohort, row.recipe), provider = row.providerPeakBytes;
    assert.ok(!('largestProviderPeakBytes' in row));
    const roles = observations[0].execution.details.report.configuration.roles.filter(role => role !== 'client').sort();
    assert.deepEqual(provider.owners, roles);
    for (const role of roles) assert.deepEqual(provider.samplesByRole[role],
      observations.map(result => result.execution.details.report.runs[0].processes[role].rss_peak_bytes));
    assert.ok(Object.values(provider.samplesByRole).flat().every(value => value === null || Number.isSafeInteger(value)));
    if (roles.includes('boundary_a')) {
      assert.deepEqual(provider.samplesByRole.boundary_a, [null, null, null]);
      assert.deepEqual(provider.samplesByRole.boundary_b, [null, null, null]);
      assert.deepEqual(provider.samples, [null, null, null]);
      for (const statistic of ['median', 'minimum', 'maximum']) assert.equal(provider[statistic], null);
      assert.ok(provider.samplesByRole.inference.every(Number.isSafeInteger));
    } else for (const statistic of ['median', 'minimum', 'maximum']) assert.ok(Number.isSafeInteger(provider[statistic]));
  }
});

test('one missing provider sample makes aggregate peak statistics unknown, without dropping known samples', () => {
  const evidence = changedOriginal(tierId, original => {
    const result = original.results.find(row => row.execution.case.name === 'maintained' && row.execution.replicate === 0);
    delete result.execution.details.report.runs[0].processes.inference.rss_peak_bytes;
  });
  const peak = evidence.cohorts[0].rows[0].providerPeakBytes;
  assert.deepEqual(peak.samples, [null, 1085276160, 1009123328]);
  for (const statistic of ['median', 'minimum', 'maximum']) assert.equal(peak[statistic], null);
  assert.deepEqual(peak.samplesByRole.preparation, [211353600, 213598208, 217776128]);
});

test('provider byte observations reject fractional values', () => {
  assert.throws(() => changedOriginal(tierId, original => {
    original.results[0].execution.details.report.runs[0].processes.inference.rss_peak_bytes += 0.5;
  }), /provider peak/i);
});

test('all-role CPU preserves raw samples and first-response windows, excluding cleanup', () => {
  const evidence = buildPublicationEvidence(read);
  for (const cohort of evidence.cohorts) for (const row of cohort.rows) {
    const observations = originalResults(cohort, row.recipe), cpu = row.coldCpuSeconds;
    const roles = [...observations[0].execution.details.report.configuration.roles].sort();
    assert.deepEqual(cpu.owners, roles);
    assert.equal(cpu.cleanupIncluded, false);
    assert.equal(cpu.clientWindow, 'benchmark_start_through_first_response');
    assert.equal(cpu.providerWindow, 'role_birth_through_first_response');
    for (const role of roles) assert.deepEqual(cpu.samplesByRole[role], observations.map(result =>
      result.execution.details.report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role[role]));
    for (const [index, result] of observations.entries()) {
      const recorded = result.execution.details.report.process_cpu_accounting.aggregate_cold_first_response_cpu_seconds;
      assert.ok(Math.abs(cpu.aggregate.samples[index] - recorded) < 1e-12);
    }
    assert.equal(row.aggregateColdCpuSeconds, cpu.aggregate.median);
  }
});

const cpuMutations = {
  deletedBoundaryRole: record => { delete record.cold_first_response_cpu_seconds_by_role.boundary_a; },
  extraRole: record => { record.cold_first_response_cpu_seconds_by_role.unmeasured_helper = 0; },
  unknownBoundaryCpu: record => { record.cold_first_response_cpu_seconds_by_role.boundary_b = null; },
  negativeBoundaryCpu: record => { record.cold_first_response_cpu_seconds_by_role.boundary_b = -1; },
  zeroAggregate: record => { record.aggregate_cold_first_response_cpu_seconds = 0; },
  missingAggregate: record => { delete record.aggregate_cold_first_response_cpu_seconds; },
  cleanupWindow: record => { record.scope = 'startup through cleanup'; },
};
for (const [name, mutate] of Object.entries(cpuMutations)) test(`rehashed CPU control rejects ${name}`, () => {
  assert.throws(() => changedOriginal(confirmationId, original => {
    for (const result of original.results.filter(row => row.execution.case.name === candidate)) {
      mutate(result.execution.details.report.process_cpu_accounting);
    }
  }), /Publication.*CPU/);
});

const metricMutations = {
  secondsForBytes: metric => { metric.unit = 'seconds'; },
  requestInsteadOfRelayWindow: metric => { metric.window = 'request_setup_through_completion'; },
  clientOnlyOwners: metric => { metric.owners = ['client']; },
  wrongRole: metric => { metric.role = 'client'; },
  physicalWireScope: metric => { metric.measurement_scope = 'physical_wire'; },
  perTokenDenominator: metric => { metric.denominator = { unit: 'generated_tokens', value: 8 }; },
};
for (const [name, mutate] of Object.entries(metricMutations)) test(`consistent rehashed tier contracts reject ${name}`, () => {
  assert.throws(() => changedOriginal(tierId, original => {
    for (const result of original.results) mutate(result.metrics.find(metric => metric.id === 'network.all_link_stream_bytes'));
  }), /Publication raw metric contract changed: network.all_link_stream_bytes/);
});

test('tier publication measurements must match raw owners, not just cached metric/summary copies', () => {
  assert.throws(() => changedOriginal(tierId, original => {
    original.results[0].execution.details.report.runs[0].durations.full_seconds += 1;
  }), /Publication metric differs from raw owner: request.0.full/);
});

test('explicit cohort/recipe IDs stabilize labels, row order and replicate pairings', () => {
  const evidence = buildPublicationEvidence(read), expected = manuscriptTables(evidence), shuffled = structuredClone(evidence);
  shuffled.cohorts.reverse();
  for (const cohort of shuffled.cohorts) cohort.rows.reverse();
  assert.deepEqual(manuscriptTables(shuffled), expected);
  const reordered = changedOriginal(tierId, original => { original.results.reverse(); });
  assert.deepEqual(reordered.cohorts[0].rows, evidence.cohorts[0].rows);
  assert.deepEqual(reordered.cohorts[0].comparisons, evidence.cohorts[0].comparisons);
  for (const cohort of evidence.cohorts) for (const comparison of cohort.comparisons) {
    const control = originalResults(cohort, comparison.control), candidate = originalResults(cohort, comparison.candidate);
    for (const [id, metric] of Object.entries(comparison.metrics)) {
      const direction = cohort.rows.find(row => row.recipe === comparison.control).metrics[id].direction;
      const paired = control.map((result, index) => {
        const a = result.metrics.find(metric => metric.id === id).value;
        const b = candidate[index].metrics.find(metric => metric.id === id).value;
        return a > 0 && b !== null ? (direction === 'min' ? 1 - b / a : b / a - 1) : null;
      });
      assert.deepEqual(metric.pairedImprovementFractions, paired);
    }
  }
  shuffled.cohorts[0].rows.push(shuffled.cohorts[0].rows[0]);
  assert.throws(() => manuscriptTables(shuffled), /recipe membership changed/);
});

test('normative sensitivity grids come from hash-bound publication contract; upper-tier coverage stays absent', () => {
  const evidence = buildPublicationEvidence(read), contract = JSON.parse(read(contractFile));
  assert.deepEqual(evidence.comparisonContract, { file: contractFile, sha256: hash(read(contractFile)) });
  assert.equal(evidence.inputs[contractFile], hash(read(contractFile)));
  assert.deepEqual(evidence.memorySensitivity.normativeCeilingsBytes, contract.memory.sensitivity_bytes);
  assert.deepEqual(evidence.memorySensitivity.normativeCeilingsBytes.low.map(bytes => bytes / 2 ** 20), [192, 224, 256, 288, 320]);
  for (const cohort of evidence.cohorts) {
    for (const [tier, grid] of Object.entries(contract.memory.sensitivity_bytes)) {
      const rows = cohort.observedRssSensitivity.filter(row => row.normativeTiers.includes(tier));
      assert.deepEqual(rows.map(row => row.ceilingBytes), grid);
      if (tier !== 'low') {
        assert.equal(cohort.rssNearCeilingCoverage[tier], 'absent');
        assert.ok(cohort.rows.every(row => row.maximumCompleteClientBytes < grid[0]));
      }
    }
  }
  assert.deepEqual(evidence.cohorts[0].observedRssSensitivity.filter(row => row.normativeTiers.includes('low'))
    .map(row => row.eligibleRecipes.length), [3, 5, 8, 8, 8]);
  assert.deepEqual(evidence.cohorts[1].observedRssSensitivity.filter(row => row.normativeTiers.includes('low'))
    .map(row => row.eligibleRecipes.length), [0, 0, 3, 3, 3]);
  contract.memory.sensitivity_bytes.low[0]++;
  assert.throws(() => buildPublicationEvidence(file => file === contractFile ? Buffer.from(JSON.stringify(contract)) : read(file)),
    /sensitivity grid differs/);
});

test('historical renderer documents write scope and never converts unknown memory into zero', () => {
  const evidence = buildPublicationEvidence(read), row = evidence.cohorts[0].rows[0];
  row.metrics['preparation.peak_rss'].median = null;
  const rendered = renderPublicationEvidence(evidence);
  assert.ok(rendered.includes('`--write` regenerates the evidence JSON and this Markdown only; both modes validate the manuscript table strings.'));
  assert.ok(rendered.includes('their near-ceiling coverage is absent'));
  assert.ok(rendered.includes(`| maintained | ${row.metrics['client.cold_cpu'].median.toFixed(2)} | ${row.metrics['preparation.cold_cpu'].median.toFixed(2)} | unknown | ${row.aggregateColdCpuSeconds.toFixed(2)} |`));
});

test('publication replay refuses changed originals and historical-to-new claim promotion', () => {
  assert.throws(() => buildPublicationEvidence(file => file.endsWith('private-boundary-decode-confirmation-150x8.json')
    ? Buffer.from('{}') : read(file)), /evidence changed/i);
  const evidence = JSON.parse(read('paper/artifacts/historical-results-v1.json'));
  for (const field of ['freshMeasurements', 'supportsNewSotaClaim', 'permitsCohortPooling']) {
    assert.throws(() => renderPublicationEvidence({ ...evidence, [field]: true }), /scope changed/);
  }
});
