import { prefixHtml } from '@/lib/paths.mjs';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
export const metadata = { title: 'Research', description: 'Private LLM Inference: the paper, measurements, and execution boundary.' };
export default function Research() {
  return <main id="main-content" dangerouslySetInnerHTML={{ __html: prefixHtml(readFileSync(join(process.cwd(), 'content/research.html'), 'utf8')) }} />;
}
