import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './research.module.css';

export const metadata = {
  title: 'PLLM research',
  description: 'The mission, protocol, and evidence for a global market in private AI compute.',
  alternates: { canonical: '/research/' },
};
export default function Research() {
  return <main id="main-content" className={styles.researchPage} dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />;
}
