import { prefixHtml } from '@/lib/paths.mjs';
import { markdownPathForRoute } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './research.module.css';

export const metadata = {
  title: 'PLLM research',
  description: 'PLLM original research, technical paper, tracked papers, and reimplementation backlog.',
  alternates: { canonical: '/research/', types: { 'text/markdown': markdownPathForRoute('/research/') } },
};
export default function Research() {
  return <main id="main-content" className={styles.researchPage} dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />;
}
