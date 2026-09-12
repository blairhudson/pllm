import { prefixHtml } from '@/lib/paths.mjs';
import type { Metadata } from 'next';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './home.module.css';

const home = readFileSync(join(process.cwd(), 'content/home.html'), 'utf8');

export const metadata: Metadata = { alternates: { canonical: '/' } };

export default function Home() {
  // Trusted repository content stays separate so non-React editors can maintain copy.
  return <main className={styles.page} id="main-content" dangerouslySetInnerHTML={{ __html: prefixHtml(home) }} />;
}
