/** Opt-in HTTP regression against a real `bun run dev` (or preview) server.
 * PLLM_DOCS_TEST_ORIGIN=http://localhost:3127 node --test tests/research-runtime.test.mjs
 * Generation must run first: source/hash/coverage audits remain in that pipeline.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const origin = process.env.PLLM_DOCS_TEST_ORIGIN;
test('real server bundles every research route without favicon error propagation', { skip: !origin, timeout: 240_000 }, async () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../public/docs-manifest.json', import.meta.url)));
  const routes = [...new Set(['/research/', '/favicon.ico', ...manifest.pages
    .map((page) => page.canonicalUrl).filter((url) => url.startsWith('/research/'))])];
  assert.ok(routes.length > 80, 'Exercise complete research publication graph');
  // Bound concurrency: cold MDX compilation is intentionally exercised here.
  let next = 0;
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (next < routes.length) {
      const route = routes[next++];
      const response = await fetch(new URL(route, origin), { signal: AbortSignal.timeout(120_000) });
      // No favicon.ico asset exists; its normal 404 must not become a build 500.
      assert.equal(response.status, route === '/favicon.ico' ? 404 : 200, route);
      if (route === '/favicon.ico') { await response.arrayBuffer(); continue; }
      const html = await response.text();
      assert.match(response.headers.get('content-type'), /text\/html/, route);
      assert.doesNotMatch(html, /Module not found|Can't resolve|NEXT_HTTP_ERROR_FALLBACK;500/, route);
      if (route === '/research/') {
        assert.match(html, /Certified head-codec evidence coverage/);
        assert.match(html, /archived initial screen; independent chart audit pending/);
        assert.match(html, /chart metrics and RAM tiers remain unscored/);
        assert.match(html, /<h2 id="paper-title">Publications<\/h2>/);
      }
      if (['/research/paper/', '/research/low-memory/', '/research/whitepaper/'].includes(route)) {
        const start = html.indexOf('id="nd-sidebar"');
        assert.ok(start >= 0, route);
        const sidebar = html.slice(start, html.indexOf('</aside>', start));
        assert.match(sidebar, />Publications</, route);
        for (const slug of ['paper', 'low-memory', 'whitepaper']) {
          assert.ok(sidebar.includes(`href="/research/${slug}/"`), `${route}: ${slug}`);
        }
        assert.match(sidebar, /academic draft/);
        assert.match(sidebar, /non-academic whitepaper/);
      }
    }
  }));
  console.log(`Verified ${routes.length} research/favicon URLs at ${origin}`);
});
