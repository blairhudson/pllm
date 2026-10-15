export type RelationshipMetric = { id: string; direction: string };
export type RelationshipRow = { id: string; label: string; metrics: Record<string, number | null>;
  placement: { topology: string }; clientMemory?: { maxPeakBytes: number | null } };
type Range = { min: number; max: number } | null;
export type Relationship = {
  x: string; y: string; n: number; missing: number; status: 'insufficient' | 'constant' | 'measured';
  points: Array<{ id: string; label: string; topology: string; x: number; y: number }>;
  pearson: number | null; spearman: number | null; alignedPearson: number | null; alignedSpearman: number | null;
  xRange: Range; yRange: Range;
  leaveOneOut: Record<'pearson' | 'spearman', { range: Range; defined: number; omitted: number }>;
};
export const relationshipMetrics: string[];
export const minimumConfigurations: number;
export function metricRelationship(rows: RelationshipRow[], x: RelationshipMetric, y: RelationshipMetric): Relationship;
export function relationshipTable<R extends RelationshipRow>(cohort: { id: string; rows: R[] }, topology: string,
  metrics: RelationshipMetric[], maxBytes?: number | null, includeTopologies?: boolean): {
    rows: R[]; unknownCount: number; overBudgetCount: number; cohortId: string; topology: string | null;
    includeTopologies: boolean; maxBytes: number | null; minimumConfigurations: number; pairs: Relationship[];
  };
