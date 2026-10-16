export type RowCodecArchiveFile = { file: string; sha256: string };
export type RowCodecCoverageEntry = {
  id: string;
  adapter: 'private-boundary-row-codec-coverage-v1';
  evidenceTier: 'screen';
  inputTokens: 150;
  outputCap: 8 | 64;
  cases: ['parallel-uniform-residues', 'parallel-row-residues'];
  roles: ['client', 'inference', 'preparation', 'boundary_a', 'boundary_b'];
  reference: 'docs/research/private-boundary-row-codec-evidence.md';
  cohort: null;
} & (
  | { status: 'awaiting-completed-archive'; archive: null }
  | { status: 'accepted-archived-pending-chart-audit';
      archive: Record<'report' | 'plan' | 'receipt' | 'summary', RowCodecArchiveFile> }
  | { status: 'memory-blocked-attempt';
      archive: Record<'report' | 'plan' | 'progress', RowCodecArchiveFile> }
);
export type RowCodecCoverage = RowCodecCoverageEntry & {
  rankable: false;
  metrics: null;
  observedTiers: null;
  codecSourceCertificationVerified: null;
  coldNetworkComplete: false;
  sotaClaimAdmissible: false;
  blockers: string[];
};
export function rowCodecCoverage(entries?: RowCodecCoverageEntry[], read?: (file: string) => Uint8Array): RowCodecCoverage[];
