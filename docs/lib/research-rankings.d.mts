type Row = { id: string; metrics: Record<string, number | null> };
type Metric = { id: string; direction: string; scoreBounds?: number[] };
export function leaders(rows: Row[], metric: Metric): string[];
export function sortMetricRows<R extends Row>(rows: R[], metricId: string, direction: 'ascending' | 'descending'): R[];
export function improvement(value: number | null, baseline: number | null, direction: string): number | null;
export const defaultMetrics: string[];
export const sotaMetrics: string[];
export function sotaTable<R extends Row & { placement: { topology: string } }>(
  cohort: { rows: R[] }, topology: string, metrics: Metric[],
): { rows: R[]; leaders: Record<string, string[]>; precision: Record<string, number> };
type Component = { component: string; params?: Record<string, unknown> };
export function componentCombination(pipeline: { components: Record<string, Component | undefined> }): Array<
  Component & { slot: string; name: string }
>;
export const priorities: Array<{ label: string; anchor: string; metrics: string[]; why: string }>;
export function groupedMetrics<M extends { group: string }>(metrics: M[]): Array<{ label: string; title: string; metrics: M[] }>;
export function normalizedAxes<M extends Metric>(rows: Row[], metrics: M[]): Array<M & {
  min: number | null;
  max: number | null;
  scores: Record<string, number | null>;
}>;
export function measuredSegments(values: Array<number | null>): Array<Array<{ index: number; value: number }>>;
