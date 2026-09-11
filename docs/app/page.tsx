import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const home = readFileSync(join(process.cwd(), 'content/home.html'), 'utf8');

export default function Home() {
  // Trusted repository content stays separate so non-React editors can maintain copy.
  return <main id="main-content" dangerouslySetInnerHTML={{ __html: prefixHtml(home) }} />;
}
