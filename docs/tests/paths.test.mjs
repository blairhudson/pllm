import test from 'node:test';
import assert from 'node:assert/strict';
import { markdownPathForRoute, normalizeBasePath, withBasePath, prefixHtml } from '../lib/paths.mjs';
import { canonicalDocsUrl } from '../lib/docs-routes.mjs';

test('root and project Pages paths are normalized', () => {
  assert.equal(normalizeBasePath('/'), '');
  assert.equal(normalizeBasePath('/pllm/'), '/pllm');
  assert.equal(normalizeBasePath(''), '');
});
test('invalid paths fail before build', () => {
  for (const value of ['https://example.com', '//evil', '/../evil', '/bad?x']) {
    assert.throws(() => normalizeBasePath(value));
  }
});
test('downloads and search use the project path once', () => {
  assert.equal(withBasePath('/downloads/paper.pdf', '/pllm'), '/pllm/downloads/paper.pdf');
  assert.equal(withBasePath('/search-index.json', '/pllm'), '/pllm/search-index.json');
  assert.equal(withBasePath('/pllm/llms.txt', '/pllm'), '/pllm/llms.txt');
  assert.equal(withBasePath('/research/whitepaper/', '/pllm'), '/pllm/research/whitepaper/');
  assert.equal(withBasePath('/downloads/whitepaper-source.zip', '/pllm'), '/pllm/downloads/whitepaper-source.zip');
});
test('external links, anchors and protocol relative links stay unchanged', () => {
  for (const href of ['https://example.com', '//cdn.example.com/a', '#start', 'mailto:hello@example.com']) {
    assert.equal(withBasePath(href, '/pllm'), href);
  }
});
test('trusted HTML links work on GitHub project Pages', () => {
  assert.equal(prefixHtml('<a href="/">Docs</a><img src="/icon.svg">', '/pllm'), '<a href="/pllm/">Docs</a><img src="/pllm/icon.svg">');
});
test('canonical routes map to stable Markdown alternates without index leaves', () => {
  assert.equal(markdownPathForRoute('/'), '/index.md');
  assert.equal(markdownPathForRoute('/sdk/components/'), '/sdk/components.md');
  assert.equal(markdownPathForRoute('/learn/start/installation'), '/learn/start/installation.md');
  assert.equal(markdownPathForRoute('/research/'), '/research.md');
  assert.equal(markdownPathForRoute('/sdk/reference/components/'), '/sdk/reference/components.md');
  assert.equal(markdownPathForRoute('/research/whitepaper'), '/research/whitepaper.md');
});
test('nested CLI reference sources map before generic SDK reference routes', () => {
  assert.equal(canonicalDocsUrl('content/docs/reference/cli/index.mdx'), '/cli/reference/');
  assert.equal(canonicalDocsUrl('content/docs/reference/cli/config/index.mdx'), '/cli/reference/config/');
  assert.equal(canonicalDocsUrl('content/docs/cli/private-inference.mdx'), '/cli/private-inference/');
  assert.equal(canonicalDocsUrl('content/docs/reference/cli/components/list.mdx'), '/cli/reference/components/list/');
});

test('nested Learn integration sources keep canonical public routes', () => {
  assert.equal(canonicalDocsUrl('content/docs/learn/integrations/index.mdx'), '/learn/integrations/');
  assert.equal(canonicalDocsUrl('content/docs/learn/integrations/responses-api.mdx'), '/learn/integrations/responses-api/');
  assert.equal(canonicalDocsUrl('content/docs/learn/integrations/openai-agents.mdx'), '/learn/integrations/openai-agents/');
  assert.equal(canonicalDocsUrl('content/docs/learn/integrations/opencode.mdx'), '/learn/integrations/opencode/');
});
