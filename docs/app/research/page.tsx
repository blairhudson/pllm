import { prefixHtml } from '@/lib/paths.mjs';
import { markdownPathForRoute } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './research.module.css';

export const metadata = {
  title: 'PLLM research',
  description: 'Private inference as eight competing research problems: client resources, Preparation, responsiveness, and base plus per-token traffic.',
  alternates: { canonical: '/research/', types: { 'text/markdown': markdownPathForRoute('/research/') } },
};
export default function Research() {
  return <main id="main-content" className={styles.researchPage} dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />;
}
