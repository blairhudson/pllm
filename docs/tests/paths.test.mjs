import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBasePath, withBasePath, prefixHtml } from '../lib/paths.mjs';

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
  assert.equal(withBasePath('/downloads/whitepaper.tex', '/pllm'), '/pllm/downloads/whitepaper.tex');
});
test('external links, anchors and protocol relative links stay unchanged', () => {
  for (const href of ['https://example.com', '//cdn.example.com/a', '#start', 'mailto:hello@example.com']) {
    assert.equal(withBasePath(href, '/pllm'), href);
  }
});
test('trusted HTML links work on GitHub project Pages', () => {
  assert.equal(prefixHtml('<a href="/docs/">Docs</a><img src="/icon.svg">', '/pllm'), '<a href="/pllm/docs/">Docs</a><img src="/pllm/icon.svg">');
});
