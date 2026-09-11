import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './whitepaper.module.css';

export const metadata = {
  title: 'Whitepaper',
  description: 'A two-page decision brief for PLLM prepared private inference.',
};

export default function Whitepaper() {
  const html = readFileSync(join(process.cwd(), 'content/whitepaper.html'), 'utf8');
  return <main id="main-content" className={styles.whitepaperPage} dangerouslySetInnerHTML={{ __html: prefixHtml(html) }} />;
}
