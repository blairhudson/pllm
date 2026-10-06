type Row = { id: string; metrics: Record<string, number | null> };
type Metric = { id: string; direction: string; scoreBounds?: number[] };
export function leaders(rows: Row[], metric: Metric): string[];
export function improvement(value: number | null, baseline: number | null, direction: string): number | null;
export const defaultMetrics: string[];
export function groupedMetrics<M extends { group: string }>(metrics: M[]): Array<{ label: string; title: string; metrics: M[] }>;
export function normalizedAxes<M extends Metric>(rows: Row[], metrics: M[]): Array<M & {
  min: number | null;
  max: number | null;
  scores: Record<string, number | null>;
}>;
export function measuredSegments(values: Array<number | null>): Array<Array<{ index: number; value: number }>>;
