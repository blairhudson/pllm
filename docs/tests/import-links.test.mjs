import assert from 'node:assert/strict';
import test from 'node:test';

import { importReferences, inlineImportTransformer } from '../lib/import-links.mjs';

function textOf(node) {
  return node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');
}

test('highlighted Python imports link in place without changing copyable code', () => {
  const code = [
    'from pllm.nonlinear import (',
    '    ArithmeticGarblingSiluQ7,',
    '    RippleLookup as Lookup,',
    ')',
    'import pllm.models as models',
    'from pathlib import Path',
    'from openai import OpenAI',
  ].join('\n');
  const transformer = inlineImportTransformer();
  const links = [];
  for (const [index, line] of code.split('\n').entries()) {
    const node = {
      type: 'element', tagName: 'span', properties: { className: ['line'] },
      children: [...line].map((char) => ({
        type: 'element', tagName: 'span', properties: { style: 'color: red' },
        children: [{ type: 'text', value: char }],
      })),
    };
    transformer.line.call({ options: { lang: 'python' }, source: code }, node, index + 1);
    assert.equal(textOf(node), line);
    for (const child of node.children.filter((item) => item.tagName === 'a')) {
      links.push({ text: textOf(child), href: child.properties.href });
    }
  }
  assert.deepEqual(links, [
    { text: 'pllm.nonlinear', href: '/sdk/reference/python/pllm/nonlinear/' },
    { text: 'ArithmeticGarblingSiluQ7', href: '/sdk/reference/python/pllm/nonlinear/#arithmeticgarblingsiluq7' },
    { text: 'RippleLookup', href: '/sdk/reference/python/pllm/nonlinear/#ripplelookup' },
    { text: 'pllm.models', href: '/sdk/reference/python/pllm/models/' },
    { text: 'pathlib', href: 'https://docs.python.org/3/library/pathlib.html' },
    { text: 'Path', href: 'https://docs.python.org/3/library/pathlib.html#pathlib.Path' },
    { text: 'openai', href: 'https://github.com/openai/openai-python' },
    { text: 'OpenAI', href: 'https://github.com/openai/openai-python' },
  ]);
  assert.deepEqual(importReferences('from pllm import Model as Source, Experiment'), [
    { label: 'pllm', href: '/sdk/reference/python/pllm/' },
    { label: 'pllm.Model', href: '/sdk/reference/python/pllm/#model' },
    { label: 'pllm.Experiment', href: '/sdk/reference/python/pllm/#experiment' },
  ]);
  assert.deepEqual(importReferences('from pllm.nonlinear import fit_compact_silu_q7_reference'), [
    { label: 'pllm.nonlinear', href: '/sdk/reference/python/pllm/nonlinear/' },
    { label: 'pllm.nonlinear.fit_compact_silu_q7_reference', href: '/sdk/reference/python/pllm/nonlinear/#fit_compact_silu_q7_reference' },
  ]);
});

test('undocumented imported modules fail at build time', () => {
  assert.throws(() => importReferences('from unknown_sdk import NewMethod'), /without a reference mapping/);
});
