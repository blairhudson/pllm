/** Coverage only. Never converts a cached row-codec summary into scored evidence. */
import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const check = (condition, message) => { if (!condition) throw new Error(message); };
const names = ['parallel-uniform-residues', 'parallel-row-residues'];
const roles = ['client', 'inference', 'preparation', 'boundary_a', 'boundary_b'];
const reference = 'docs/research/private-boundary-row-codec-evidence.md';
const archiveRoot = 'docs/evidence/private-boundary-row-codec-qwen25-v1/';
const auditBlocker = 'Independent read-only certificate/native/workload/accounting/RSS chart audit not implemented; Python acceptance and summary validity do not supply chart scores.';
const readOriginal = (file) => readFileSync(new URL(`../../${file}`, import.meta.url));

export function rowCodecCoverage(entries = [], read = readOriginal) {
  check(Array.isArray(entries), 'Row-codec coverage must be an array');
  const seen = new Set();
  return entries.map((entry) => {
    check(entry && typeof entry === 'object' && !Array.isArray(entry), 'Invalid row-codec coverage');
    check(isDeepStrictEqual(Object.keys(entry).sort(), ['id', 'adapter', 'evidenceTier', 'status', 'inputTokens',
      'outputCap', 'cases', 'roles', 'reference', 'archive', 'cohort'].sort()), 'Unknown row-codec coverage fields');
    check(entry.adapter === 'private-boundary-row-codec-coverage-v1' && entry.evidenceTier === 'screen' &&
      entry.inputTokens === 150 && [8, 64].includes(entry.outputCap) &&
      entry.id === `qwen25-private-boundary-row-codec-sdk-150-${entry.outputCap}` && !seen.has(entry.id),
    'Invalid or duplicate row-codec screen coverage');
    seen.add(entry.id);
    check(isDeepStrictEqual(entry.cases, names) && isDeepStrictEqual(entry.roles, roles) &&
      entry.reference === reference && entry.cohort === null,
    'Row-codec coverage cannot claim pooled or chart-audited evidence');
    let blockers;
    if (entry.status === 'awaiting-completed-archive') {
      check(entry.archive === null, 'Row-codec pending coverage cannot claim archived evidence');
      blockers = ['Completed original report, frozen plan, terminal receipt and summary not registered.', auditBlocker];
    } else {
      const accepted = entry.status === 'accepted-archived-pending-chart-audit';
      const blocked = entry.status === 'memory-blocked-attempt' && entry.outputCap === 64;
      check(accepted || blocked, 'Invalid row-codec archive coverage status');
      const files = accepted ? ['report', 'plan', 'receipt', 'summary'] : ['report', 'plan', 'progress'];
      check(entry.archive && isDeepStrictEqual(Object.keys(entry.archive).sort(), [...files].sort()),
        'Invalid row-codec archive originals');
      const directory = accepted ? `150x${entry.outputCap}` : '150x64-rejected-v1';
      const version = accepted ? { 8: 'v1', 64: 'v2' }[entry.outputCap] : 'v1';
      for (const kind of files) {
        const pointer = entry.archive[kind];
        const suffix = kind === 'report' ? '.json' : `.${kind}.json`;
        check(pointer && isDeepStrictEqual(Object.keys(pointer).sort(), ['file', 'sha256']) &&
          pointer.file === `${archiveRoot}${directory}/certified-head-qwen150x${entry.outputCap}-${version}${suffix}` &&
          typeof pointer.sha256 === 'string' && /^[a-f0-9]{64}$/.test(pointer.sha256),
        'Invalid row-codec archive pointer');
        check(createHash('sha256').update(read(pointer.file)).digest('hex') === pointer.sha256,
          `Row-codec archive bytes changed: ${pointer.file}`);
      }
      // Hash checking originals is linkage only: deliberately do not consume cached metrics.
      blockers = accepted ? [auditBlocker] : [
        'Only one control completed; five cells blocked by topology_memory_limit before providers (safe headroom 5.04–5.50 GiB versus 6.03–6.04 GiB required).',
        'No terminal receipt or summary; this attempt has no accepted balanced six-client 150+64 cohort.', auditBlocker];
    }
    return { ...structuredClone(entry), rankable: false, metrics: null, observedTiers: null,
      codecSourceCertificationVerified: null, coldNetworkComplete: false, sotaClaimAdmissible: false,
      blockers };
  });
}
