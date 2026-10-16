import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { rowCodecCoverage } from '../scripts/research-row-codec-coverage.mjs';
import { adaptedCohorts } from '../scripts/research-scorecard-adapters.mjs';

const read = (file) => fs.readFileSync(new URL(`../../${file}`, import.meta.url));
const registry = () => JSON.parse(read('docs/data/research/benchmark-cohorts.json'));

test('row-codec typed coverage separates workloads and never supplies scoring observations', () => {
  const manifest = registry(), before = structuredClone(manifest.row_codec_coverage);
  const entries = rowCodecCoverage(manifest.row_codec_coverage);
  assert.deepEqual(manifest.row_codec_coverage, before);
  assert.deepEqual(entries.map((entry) => entry.outputCap), [8, 64]);
  assert.ok(entries.every((entry) => entry.status === 'accepted-archived-pending-chart-audit'));
  for (const entry of entries) {
    assert.equal(entry.evidenceTier, 'screen');
    assert.equal(entry.rankable, false);
    for (const key of ['metrics', 'observedTiers', 'codecSourceCertificationVerified', 'cohort']) {
      assert.equal(entry[key], null);
    }
    assert.equal(entry.coldNetworkComplete, false);
    assert.equal(entry.sotaClaimAdmissible, false);
    assert.equal(entry.roles.length, 5);
    assert.match(read(entry.reference).toString(), /Coverage only/);
    assert.ok(!manifest.adapted_cohorts.some((cohort) => cohort.id === entry.id));
    assert.match(entry.blockers.join(' '), /Independent read-only.*chart audit not implemented/);
  }
  assert.deepEqual(Object.keys(entries[0].archive), ['report', 'plan', 'receipt', 'summary']);
  assert.deepEqual(Object.keys(entries[1].archive), ['report', 'plan', 'receipt', 'summary']);
  assert.match(entries[0].archive.report.file, /150x8\/certified-head-qwen150x8-v1\.json$/);
  assert.match(entries[1].archive.report.file, /150x64\/certified-head-qwen150x64-v2\.json$/);
  assert.ok(entries.every((entry) => !entry.blockers.some((blocker) => /topology_memory_limit/.test(blocker))));
});

test('historical memory-blocked attempt remains supported without borrowing fresh long results', () => {
  const historical = structuredClone(registry().row_codec_coverage[1]);
  historical.status = 'memory-blocked-attempt';
  const base = 'docs/evidence/private-boundary-row-codec-qwen25-v1/150x64-rejected-v1/certified-head-qwen150x64-v1';
  historical.archive = {
    report: { file: `${base}.json`, sha256: 'bf143b709597079c9bca9035a9861badb394771df4f09705f16bdd58a25f269f' },
    plan: { file: `${base}.plan.json`, sha256: 'c3e1bef3a7c057638181510b85e6bf451bbb22e64e6c7081021178a43be2c7d0' },
    progress: { file: `${base}.progress.json`, sha256: '0a23808c09cbee97a1a545728504c6edcf2bfe48d20b1ecaad884e753c87487c' },
  };
  const calls = [];
  const [entry] = rowCodecCoverage([historical], (file) => { calls.push(file); return read(file); });
  assert.deepEqual(calls, Object.values(historical.archive).map(({ file }) => file));
  assert.match(entry.blockers.join(' '), /five cells blocked by topology_memory_limit before providers/);
  assert.match(entry.blockers.join(' '), /No terminal receipt or summary/);
  assert.equal(entry.metrics, null);
  assert.equal(entry.cohort, null);
  assert.equal(entry.rankable, false);
  historical.status = 'accepted-archived-pending-chart-audit';
  assert.throws(() => rowCodecCoverage([historical]), /archive originals/);
});

test('legacy pending coverage remains valid without reading target or archive files', () => {
  const pending = registry().row_codec_coverage.map((entry) => ({ ...entry,
    status: 'awaiting-completed-archive', archive: null }));
  let reads = 0;
  const entries = rowCodecCoverage(pending, () => { reads++; throw new Error('Unexpected read'); });
  assert.equal(reads, 0);
  assert.ok(entries.every((entry) => entry.archive === null && entry.metrics === null));
  assert.match(entries[0].blockers[0], /not registered/);
});

test('every registered original is hash checked, never parsed as a scoring summary', () => {
  const manifest = registry().row_codec_coverage;
  const calls = [];
  rowCodecCoverage(manifest, (file) => { calls.push(file); return read(file); });
  assert.deepEqual(calls, manifest.flatMap((entry) => Object.values(entry.archive).map(({ file }) => file)));
  for (const file of calls) {
    assert.throws(() => rowCodecCoverage(manifest, (path) => path === file ? Buffer.from('changed') : read(path)),
      /Row-codec archive bytes changed/);
  }
  // Even a hash-bound "summary" need not be JSON here: this verifies byte linkage, not its assertions.
  const bytes = Buffer.from('not a valid summary; must not be scored');
  for (const index of [0, 1]) {
    const synthetic = structuredClone(manifest);
    const summary = synthetic[index].archive.summary;
    summary.sha256 = createHash('sha256').update(bytes).digest('hex');
    const entries = rowCodecCoverage(synthetic, (file) => file === summary.file ? bytes : read(file));
    assert.ok(entries.every((entry) => entry.metrics === null && entry.rankable === false && entry.cohort === null));
  }
});

test('coverage rejects fabricated observations, promotion, pooling and unknown fields', () => {
  const changes = [
    (entry) => { entry.evidenceTier = 'confirmation'; },
    (entry) => { entry.status = 'completed'; },
    (entry) => { entry.archive = { summary: 'untrusted-summary.json' }; },
    (entry) => { entry.cohort = 'a'.repeat(64); },
    (entry) => { entry.metrics = {}; },
    (entry) => { entry.inputTokens = 128; },
    (entry) => { entry.outputCap = 32; },
    (entry) => { entry.cases.pop(); },
    (entry) => { entry.roles.pop(); },
    (entry) => { entry.status = 'memory-blocked-attempt'; },
    (entry) => { entry.archive.report.file = 'target/certified-head-qwen150x8-v1.json'; },
    (entry) => { entry.archive.report.sha256 = 'not-a-hash'; },
    (entry) => { entry.archive.report.metrics = {}; },
    (entry) => { delete entry.archive.receipt; },
    (entry) => { entry.archive.progress = entry.archive.report; },
  ];
  for (const change of changes) {
    const entries = structuredClone(registry().row_codec_coverage);
    change(entries[0]);
    assert.throws(() => rowCodecCoverage(entries), /row-codec|Row-codec/);
  }
  const [entry] = registry().row_codec_coverage;
  assert.throws(() => rowCodecCoverage([entry, entry]), /duplicate/);
  assert.throws(() => rowCodecCoverage(null), /array/);
  const long = structuredClone(registry().row_codec_coverage[1]);
  long.status = 'memory-blocked-attempt';
  assert.throws(() => rowCodecCoverage([long]), /archive originals/);
  const pooled = structuredClone(registry().row_codec_coverage);
  pooled[1].archive = structuredClone(pooled[0].archive);
  assert.throws(() => rowCodecCoverage(pooled), /archive pointer/);
  const stale = structuredClone(registry().row_codec_coverage[1]);
  stale.archive.report.file = stale.archive.report.file.replace('-v2.json', '-v1.json');
  assert.throws(() => rowCodecCoverage([stale]), /archive pointer/);
});

test('tiny archived summary cannot become serving evidence through router', () => {
  const base = 'docs/evidence/private-boundary-row-codec-tiny-v1/';
  const index = JSON.parse(read(base + 'index.json'));
  const tiny = JSON.parse(read(base + index.accepted_summary));
  assert.equal(tiny.tiny, true);
  for (const adapter of ['private-boundary-row-codec-v1', 'private-boundary-row-codec-coverage-v1']) {
    let reads = 0;
    assert.throws(() => adaptedCohorts({ adapter, summary: base + index.accepted_summary }, () => {
      reads++; return Buffer.from(JSON.stringify(tiny));
    }, {}), /coverage-only pending independent read-only audit/);
    assert.equal(reads, 0, 'No summary-only fallback or pending-target reads');
  }
});

test('Research landing exposes owned evidence page without ranking it', () => {
  const page = read('docs/app/research/page.tsx').toString();
  assert.match(page, /rowCodecCoverage\(registry\.row_codec_coverage\)/);
  assert.match(page, /github\.com\/blairhudson\/pllm\/blob\/main\/docs\/research\/private-boundary-row-codec-evidence\.md/);
  assert.match(read('docs/scripts/research-scorecard.mjs').toString(), /rowCodecCoverage: rowCodecCoverage\(manifest\.row_codec_coverage\)/);
});
