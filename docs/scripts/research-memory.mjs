/** Observed process peaks qualify a budget; a median never hides an overrun. */
export function clientMemoryEvidence(peaks, scope) {
  if (!peaks.length || peaks.some((peak) => peak != null && (!Number.isSafeInteger(peak) || peak <= 0))) {
    throw new Error('Invalid client process peak RSS evidence');
  }
  return { maxPeakBytes: peaks.some((peak) => peak == null) ? null : Math.max(...peaks),
    sampleCount: peaks.length, scope };
}
