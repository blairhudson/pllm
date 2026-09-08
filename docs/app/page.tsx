import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
export default function Home() {
  // This is a checked, repository-owned design asset, not user content.
  const html = readFileSync(join(process.cwd(), 'content/home.html'), 'utf8');
  return <main id="main-content" dangerouslySetInnerHTML={{ __html: prefixHtml(html) }} />;
}
