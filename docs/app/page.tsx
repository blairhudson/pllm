import { markdownPathForRoute, prefixHtml } from '@/lib/paths.mjs';
import { packageVersion } from '@/scripts/package-version.mjs';
import type { Metadata } from 'next';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import styles from './home.module.css';

const home = readFileSync(join(process.cwd(), 'content/home.html'), 'utf8').replaceAll(
  '__PLLM_VERSION__',
  packageVersion,
);

export const metadata: Metadata = {
  alternates: { canonical: '/', types: { 'text/markdown': markdownPathForRoute('/') } },
};

export default function Home() {
  // Trusted repository content stays separate so non-React editors can maintain copy.
  return <main className={styles.page} id="main-content" dangerouslySetInnerHTML={{ __html: prefixHtml(home) }} />;
}
