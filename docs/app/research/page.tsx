import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './research.module.css';

export const metadata = { title: 'Research history', description: 'PLLM protocol research from the BFV bridge to offline prepared inference.' };
export default function Research() {
  return <main id="main-content" className={styles.researchPage} dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />;
}
