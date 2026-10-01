/** Make imported names clickable inside Shiki-highlighted Python code. */

const STDLIB = new Set(['asyncio', 'importlib', 'json', 'math', 'pathlib', 'platform', 'struct', '__future__']);
const EXTERNAL = {
  openai: 'https://github.com/openai/openai-python',
  numpy: 'https://numpy.org/doc/stable/',
  agents: 'https://openai.github.io/openai-agents-python/',
  'agents.models.openai_responses': 'https://openai.github.io/openai-agents-python/ref/models/openai_responses/',
};
const INTERNAL_GUIDES = {
  'pllm.runtime.authenticated_mpc': '/sdk/run/runtime-internals/',
};

function referenceHref(module, name) {
  if (INTERNAL_GUIDES[module]) return INTERNAL_GUIDES[module];
  if (module === 'pllm' || module.startsWith('pllm.')) {
    const suffix = module === 'pllm' ? '' : `${module.slice(5).replaceAll('.', '-').replaceAll('_', '-')}/`;
    return `/sdk/reference/python/pllm/${suffix}${name ? `#${name.toLowerCase()}` : ''}`;
  }
  if (STDLIB.has(module)) {
    return `https://docs.python.org/3/library/${module}.html${name ? `#${module}.${name}` : ''}`;
  }
  if (EXTERNAL[module]) return EXTERNAL[module];
  throw new Error(`Python example imports ${module}${name ? `.${name}` : ''} without a reference mapping`);
}

function importTokens(code) {
  const refs = [];
  const add = (module, name, start, length) => {
    const label = name ? `${module}.${name}` : module;
    refs.push({ label, href: referenceHref(module, name), start, end: start + length });
  };
  for (const match of code.matchAll(/^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:\(([\s\S]*?)\)|([^\n]*))/gm)) {
    const module = match[1];
    const moduleStart = match.index + match[0].indexOf(module);
    add(module, null, moduleStart, module.length);
    const body = match[2] ?? match[3];
    const bodyStart = match.index + match[0].indexOf('import') + 'import'.length + (match[2] ? 2 : 1);
    const uncommented = body.replaceAll(/#[^\n]*/g, (comment) => ' '.repeat(comment.length));
    for (const token of uncommented.matchAll(/(?:^|,)\s*([A-Za-z_]\w*)/g)) {
      const name = token[1];
      add(module, name, bodyStart + token.index + token[0].lastIndexOf(name), name.length);
    }
  }
  for (const match of code.matchAll(/^[ \t]*import[ \t]+([^\n#]+)/gm)) {
    const bodyStart = match.index + match[0].indexOf('import') + 'import'.length + 1;
    for (const token of match[1].matchAll(/(?:^|,)\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)/g)) {
      const module = token[1];
      add(module, null, bodyStart + token.index + token[0].lastIndexOf(module), module.length);
    }
  }
  return refs.sort((left, right) => left.start - right.start);
}

export function importReferences(code) {
  return importTokens(code).map(({ label, href }) => ({ label, href }));
}

function linkLine(node, refs) {
  const children = [];
  let offset = 0;
  let linked = null;
  for (const child of node.children) {
    const textNode = child.type === 'text'
      ? child
      : child.tagName === 'span' && child.children.length === 1 && child.children[0].type === 'text'
        ? child.children[0] : null;
    const length = textNode?.value.length ?? 0;
    const overlaps = refs.some((ref) => ref.start < offset + length && ref.end > offset);
    if (!textNode && overlaps) throw new Error('Cannot link an imported name in highlighted Python code');
    if (!textNode) {
      children.push(child);
      continue;
    }
    const edges = [...new Set([0, length, ...refs.flatMap((ref) => [ref.start - offset, ref.end - offset])
      .filter((position) => position > 0 && position < length)])].sort((a, b) => a - b);
    for (let i = 0; i < edges.length - 1; i++) {
      const start = edges[i];
      const end = edges[i + 1];
      const text = textNode.value.slice(start, end);
      const fragment = child.type === 'text'
        ? { type: 'text', value: text }
        : { ...child, children: [{ type: 'text', value: text }] };
      const ref = refs.find((item) => item.start <= offset + start && item.end >= offset + end);
      if (ref) {
        if (!linked || linked.properties.href !== ref.href) {
          linked = {
            type: 'element', tagName: 'a',
            properties: { href: ref.href, dataImportLink: true, ariaLabel: `API: ${ref.label}` },
            children: [],
          };
          children.push(linked);
        }
        linked.children.push(fragment);
      } else {
        linked = null;
        children.push(fragment);
      }
    }
    offset += length;
  }
  node.children = children;
}

export function inlineImportTransformer() {
  return {
    name: 'pllm:inline-python-imports',
    line(node, lineNumber) {
      if (!/^(python\d*|py)$/.test(this.options.lang)) return;
      const lines = this.source.split('\n');
      const lineStart = lines.slice(0, lineNumber - 1).reduce((sum, line) => sum + line.length + 1, 0);
      const lineEnd = lineStart + lines[lineNumber - 1].length;
      const refs = importTokens(this.source)
        .filter((ref) => ref.start >= lineStart && ref.end <= lineEnd)
        .map((ref) => ({ ...ref, start: ref.start - lineStart, end: ref.end - lineStart }));
      if (refs.length) linkLine(node, refs);
    },
  };
}
