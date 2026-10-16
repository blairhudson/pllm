import { prefixHtml } from '@/lib/paths.mjs';
import { markdownPathForRoute } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './research.module.css';
import { rowCodecCoverage, type RowCodecCoverageEntry } from '@/scripts/research-row-codec-coverage.mjs';

export const metadata = {
  title: 'PLLM research',
  description: 'Private inference as eight competing research problems: client resources, Preparation, responsiveness, and base plus per-token traffic.',
  alternates: { canonical: '/research/', types: { 'text/markdown': markdownPathForRoute('/research/') } },
};
export default function Research() {
  const registry = JSON.parse(readFileSync(join(process.cwd(), 'data/research/benchmark-cohorts.json'), 'utf8')) as {
    row_codec_coverage?: RowCodecCoverageEntry[];
  };
  const coverage = rowCodecCoverage(registry.row_codec_coverage);
  return <main id="main-content" className={styles.researchPage}>
    <div dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />
    <section aria-labelledby="row-codec-coverage">
      <h2 id="row-codec-coverage">Certified head-codec evidence coverage</h2>
      <p>Coverage only; chart scores await independent evidence adaptation. Separate source-present initial screens, not confirmation or global SOTA claims.</p>
      <ul>{coverage.map((entry) => <li key={entry.id}>
        150+{entry.outputCap}: {entry.status === 'accepted-archived-pending-chart-audit'
          ? 'archived initial screen; independent chart audit pending.'
          : entry.status === 'memory-blocked-attempt'
            ? 'memory-blocked attempt; no accepted comparison.'
            : 'awaiting completed archive and independent read-only audit.'}
        {' '}Five-owner topology; chart metrics and RAM tiers remain unscored.
      </li>)}</ul>
      <p><a href="https://github.com/blairhudson/pllm/blob/main/docs/research/private-boundary-row-codec-evidence.md">Evidence admission gates and limitations</a></p>
    </section>
  </main>;
}
