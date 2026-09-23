import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import navigation from '../navigation.json' with { type: 'json' };
import { readPages, readSearchPages, readStandalonePages, validate, siteRoot, walk } from '../scripts/content.mjs';
import { importReferences } from '../lib/import-links.mjs';

const pages = readPages();
const standalone = readStandalonePages();
const byRoute = new Map(readSearchPages().map((page) => [page.canonicalUrl, page]));

test('every Python example import links to an existing API or upstream reference', () => {
  let linked = 0;
  for (const page of readSearchPages()) {
    for (const snippet of page.content.matchAll(/```(?:python\d*|py)(?:[^\n]*)\n([\s\S]*?)\n```/g)) {
      for (const ref of importReferences(snippet[1])) {
        linked++;
        if (!ref.href.startsWith('/')) {
          assert.match(ref.href, /^https:\/\//, `${page.canonicalUrl}: ${ref.label}`);
          continue;
        }
        const [route, anchor] = ref.href.split('#');
        const reference = byRoute.get(route)?.content;
        assert.ok(reference, `${page.canonicalUrl}: ${ref.label} has no documentation at ${route}`);
        if (anchor && route.startsWith('/sdk/reference/python/pllm/')) {
          const name = ref.label.split('.').at(-1);
          assert.ok(reference.includes(`### \`${name}\``),
            `${page.canonicalUrl}: ${ref.label} has no API entry at ${ref.href}`);
        }
      }
    }
  }
  assert.ok(linked > 200, `Only ${linked} imports checked`);
});

test('canonical domain hierarchy and research journeys exist', () => {
  for (const route of [
    '/learn/', '/learn/installation/', '/learn/first-private-request/', '/learn/first-local-benchmark/', '/learn/inspect-a-plan/',
    '/learn/concepts/', '/learn/concepts/architecture/', '/learn/concepts/trust-boundary/', '/learn/concepts/privacy-assurance/', '/learn/concepts/evidence-claims/',
    '/learn/integrations/', '/learn/integrations/local-gateway/', '/learn/integrations/responses-api/', '/learn/integrations/chat-completions/',
    '/learn/integrations/openai-python/', '/learn/integrations/openai-agents/', '/learn/integrations/codex/', '/learn/integrations/opencode/',
    '/cli/', '/cli/reference/', '/cli/reference/config/', '/cli/reference/config/show/',
    '/cli/reference/benchmark/run/', '/cli/reference/components/list/',
    '/sdk/', '/sdk/experiments/', '/sdk/models/', '/sdk/inference/', '/sdk/components/',
    '/sdk/run/', '/sdk/plans/', '/sdk/evaluate/', '/sdk/extend/', '/sdk/reference/',
    '/sdk/extend/model-adapter/', '/sdk/models/families/', '/sdk/models/families/qwen2/', '/sdk/models/families/qwen3-5/',
    '/sdk/plans/operators/', '/sdk/plans/numerics/', '/sdk/plans/representations/', '/sdk/plans/conversions/',
    '/sdk/components/protocols/', '/sdk/inference/prepared-protocol/', '/sdk/components/nonlinear/garbling/',
    '/sdk/components/protocols/composition/', '/sdk/components/preparation/reference-paths/',
    '/sdk/components/nonlinear/garbling/arithmetic/', '/sdk/components/nonlinear/garbling/half-gates/',
    '/sdk/components/nonlinear/garbling/lookup-tables/', '/sdk/components/nonlinear/garbling/weighted-path/',
    '/sdk/components/preparation/', '/sdk/components/kernels/native-matrix/', '/sdk/plans/compiler-internals/', '/sdk/run/runtime-internals/',
    '/sdk/evaluate/benchmark/', '/sdk/evaluate/search/', '/sdk/evaluate/evidence/', '/sdk/evaluate/assurance/',
    '/sdk/run/clients/', '/sdk/run/local/', '/sdk/run/gateway/', '/sdk/run/embedded-gateway/',
    '/sdk/run/provider-roles/', '/sdk/run/plan-bound-runtime/', '/sdk/run/lifecycle/', '/sdk/run/deployment/',
    '/sdk/reference/', '/sdk/reference/python/pllm/', '/sdk/reference/native/', '/sdk/reference/schemas/', '/sdk/reference/components/', '/sdk/reference/status/',
    '/sdk/extend/agents/',
    '/research/', '/research/papers/', '/research/records/', '/research/records/metrics/',
    '/research/recipes/', '/research/recipes/build-method/', '/research/recipes/compare/', '/research/recipes/reproduce/', '/research/recipes/experiments/', '/research/recipes/reproductions/', '/research/recipes/agent-map/', '/research/recipes/reproduction-checklist/',
    '/research/sources/', '/research/methods/', '/research/compositions/', '/research/evidence/', '/research/publications/', '/research/clean-room/',
  ]) assert.ok(byRoute.has(route), route);
});

test('retired SDK phase URLs redirect to capability guides without duplicate canonical pages', () => {
  const redirects = fs.readFileSync(path.join(siteRoot, 'public/_redirects'), 'utf8');
  const migrations = new Map([
    ['/sdk/build/', '/sdk/'],
    ['/sdk/configuration/', '/sdk/experiments/files/'],
    ['/sdk/build/model-adapters/', '/sdk/extend/model-adapter/'],
    ['/sdk/build/models/qwen35/', '/sdk/models/families/qwen3-5/'],
    ['/sdk/build/operators/activation-silu/', '/sdk/plans/operators/activation-silu/'],
    ['/sdk/build/numerics/', '/sdk/plans/numerics/'],
    ['/sdk/build/search/', '/sdk/evaluate/search/'],
    ['/sdk/pipeline/', '/sdk/inference/'],
    ['/sdk/pipeline/protocols/', '/sdk/components/protocols/'],
    ['/sdk/pipeline/protocols/masked-linear/', '/sdk/inference/prepared-protocol/'],
    ['/sdk/pipeline/protocols/garbling/half-gates/', '/sdk/components/nonlinear/garbling/half-gates/'],
    ['/sdk/pipeline/preparation/', '/sdk/components/preparation/'],
    ['/sdk/pipeline/kernels/', '/sdk/components/kernels/native-matrix/'],
    ['/sdk/pipeline/compiler/', '/sdk/plans/compiler-internals/'],
    ['/sdk/pipeline/runtime/', '/sdk/run/runtime-internals/'],
    ['/sdk/research/', '/sdk/evaluate/'],
    ['/sdk/research/assure/', '/sdk/evaluate/assurance/'],
    ['/sdk/research/benchmarks/', '/sdk/evaluate/evidence/'],
    ['/sdk/operate/deployment/status/', '/sdk/run/lifecycle/'],
    ['/sdk/contribute/publish-a-provider/', '/sdk/extend/provider/'],
  ]);
  for (const [oldRoute, canonical] of migrations) {
    assert.ok(!byRoute.has(oldRoute), `${oldRoute} remains canonical`);
    assert.ok(byRoute.has(canonical), canonical);
    assert.ok(redirects.includes(`${oldRoute} ${canonical} 308\n`), oldRoute);
    const oldMarkdown = `${oldRoute.slice(0, -1)}.md`;
    const newMarkdown = `${canonical.slice(0, -1)}.md`;
    assert.ok(redirects.includes(`${oldMarkdown} ${newMarkdown} 308\n`), oldMarkdown);
  }
  for (const page of readSearchPages()) {
    assert.doesNotMatch(page.canonicalUrl, /^\/sdk\/(?:build|pipeline|research|operate|contribute)(?:\/|$)/);
    if (page.sourcePath.startsWith('content/docs/') || page.sourcePath === 'content/home.html') {
      assert.doesNotMatch(page.content, /\/sdk\/(?:build|pipeline|research|operate|contribute)\//, page.sourcePath);
    }
  }
});

test('private inference papers are newest-first sidebar pages with reciprocal component citations', () => {
  const dataRoot = path.join(siteRoot, 'data/research');
  const registry = JSON.parse(fs.readFileSync(path.join(dataRoot, 'papers.json'), 'utf8'));
  const library = JSON.parse(fs.readFileSync(path.join(dataRoot, 'paper-library.json'), 'utf8'));
  const planned = JSON.parse(fs.readFileSync(path.join(siteRoot, '..', 'python/pllm/components/planned_methods.json'), 'utf8')).entries;
  const plannedByPaper = new Map(planned.map((method) => [method.paper, method]));
  const citationRegistry = JSON.parse(fs.readFileSync(path.join(dataRoot, 'component-citations.json'), 'utf8'));
  const summaries = JSON.parse(fs.readFileSync(path.join(dataRoot, 'public-summaries.json'), 'utf8'));
  const newSummaries = JSON.parse(fs.readFileSync(path.join(dataRoot, 'paper-library-notes.json'), 'utf8'));
  const citations = new Map(citationRegistry.citations.map((citation) => [citation.paper_id, citation]));
  const index = fs.readFileSync(path.join(siteRoot, 'content/docs/research/papers/index.mdx'), 'utf8');

  assert.ok(index.includes('{/* Generated by docs/scripts/generate-research-papers.mjs. */}'));
  assert.ok(!index.includes('<!-- Generated'));
  assert.doesNotMatch(index, /\bR\d{2}\b/);
  const papersById = new Map(registry.papers.map((paper) => [paper.id, paper]));
  const publicPapers = library.papers;
  assert.deepEqual([...plannedByPaper.keys()].sort(), publicPapers.map((paper) => paper.id).sort());
  assert.equal(publicPapers.length, 84);
  assert.equal(publicPapers.filter((paper) => paper.status === 'available').length, 81);
  assert.deepEqual(publicPapers.filter((paper) => paper.status === 'pdf_unavailable').map((paper) => paper.id).sort(),
    ['moai', 'mozzarella']);
  assert.deepEqual(publicPapers.filter((paper) => paper.status === 'unverified_primary').map((paper) => paper.id),
    ['ripple']);
  const chronological = [...publicPapers].sort((left, right) => right.year - left.year || left.title.localeCompare(right.title));
  const slug = (paper) => papersById.get(paper.registry_id)?.slug.replaceAll('_', '-') ?? paper.id;
  assert.deepEqual(
    [...index.matchAll(/^## (\d{4})$/gm)].map((match) => Number(match[1])),
    [...new Set(chronological.map((paper) => paper.year))],
  );
  const meta = JSON.parse(fs.readFileSync(path.join(siteRoot, 'content/docs/research/papers/meta.json'), 'utf8'));
  assert.deepEqual(meta.pages.filter((entry) => !entry.startsWith('---') && entry !== 'index'),
    chronological.map(slug));
  assert.match(index, /className="paper-timeline"/);
  assert.match(index, /className="paper-timeline-entry"/);
  assert.match(fs.readFileSync(path.join(siteRoot, 'app/global.css'), 'utf8'), /\.paper-timeline-year::before/);
  assert.match(fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8'),
    /return \{ \.\.\.folder, name: 'Private Inference Papers', root: true \}/);
  let priorPosition = -1;
  for (const paper of chronological) {
    const route = `/research/papers/${slug(paper)}/`;
    const position = index.indexOf(`<a href="${route}">`);
    assert.ok(position > priorPosition, paper.title);
    priorPosition = position;
    const summary = summaries[paper.registry_id] ?? newSummaries[paper.id];
    assert.ok(summary?.overview && summary?.reading, paper.id);
    assert.ok(index.includes(summary.overview), paper.id);
    const pagePath = path.join(siteRoot, `content/docs/research/papers/${slug(paper)}.mdx`);
    const page = fs.readFileSync(pagePath, 'utf8');
    const method = plannedByPaper.get(paper.id);
    assert.ok(method, paper.id);
    const referenceRoute = `/sdk/reference/python/pllm/${method.module.replace('pllm.', '').replaceAll('.', '-').replaceAll('_', '-')}/`;
    const classLink = `${referenceRoute}#${method.name.toLowerCase()}`;
    assert.equal(method.slug || paper.id, slug(paper));
    const implemented = method.status === 'implemented';
    assert.ok(page.includes(implemented ? 'title="Related Python API"' : 'title="Planned Python API"'), paper.id);
    assert.ok(page.includes(`[${method.module}.${method.name}](${classLink})`), paper.id);
    const reference = byRoute.get(referenceRoute)?.content;
    assert.ok(reference?.includes(`[paper and provenance](${route})`), `${classLink} must cite ${route}`);
    assert.ok(reference?.includes(implemented ? 'Status: **Implemented Python API**' : 'Status: **Not yet implemented**'),
      `${classLink} must report its current API status`);
    assert.doesNotMatch(page, /verified research PDF|ignored local papers\/|PDF fingerprint/i, paper.id);
    assert.ok(page.includes(summary.overview), paper.id);
    assert.ok(page.includes(summary.reading), paper.id);
    assert.doesNotMatch(page, /Back to the chronological bibliography/);
    if (paper.status === 'unverified_primary') {
      assert.ok(page.includes('[Citing source: Curl](/research/papers/curl/)'));
      assert.ok(!page.includes(`[Original publication](${paper.source_url})`));
    } else {
      assert.ok(page.includes(`[Original publication](${paper.source_url})`), paper.id);
    }
    if (paper.status === 'available') {
      assert.ok(paper.authors?.length > 0 || paper.registry_id, `Missing publication authors: ${paper.id}`);
      assert.match(paper.sha256, /^[a-f0-9]{64}$/, paper.id);
      assert.ok(paper.bytes > 2048, paper.id);
      // papers/ is an intentionally ignored research cache, absent on CI.
      const localCopy = path.join(siteRoot, '..', paper.file);
      if (fs.existsSync(localCopy)) {
        const bytes = fs.readFileSync(localCopy);
        assert.equal(bytes.subarray(0, 5).toString(), '%PDF-', paper.id);
        assert.equal(createHash('sha256').update(bytes).digest('hex'), paper.sha256, paper.id);
        assert.equal(bytes.length, paper.bytes, paper.id);
      }
    } else {
      assert.equal(paper.file, null, paper.id);
    }
    assert.ok(byRoute.has(route), route);

    const citation = citations.get(paper.registry_id);
    if (citation) {
      assert.ok(page.includes('title="Related PLLM components"'), paper.id);
      assert.ok(page.includes(citation.summary), paper.id);
      for (const component of citation.components) {
        assert.ok(page.includes(`[${component.title}](${component.href})`), `${paper.id}: ${component.href}`);
        const componentPage = fs.readFileSync(path.join(siteRoot, component.source), 'utf8');
        assert.ok(componentPage.includes(route), `${component.source} must cite ${route}`);
      }
    } else {
      assert.ok(!page.includes('title="Related PLLM components"'), paper.id);
    }
  }
  assert.doesNotMatch(index, /^### \[.*\]\(\/research\/papers\//m,
    'Fumadocs wraps headings in anchors; timeline paper links must not be inside headings');
  assert.ok(index.includes('Maverick: Private and Verifiable LLM Inference Made Practical'));
  assert.equal(citations.size, 6);
  const roadmap = byRoute.get('/sdk/components/research-method-roadmap/')?.content;
  assert.ok(roadmap);
  for (const method of planned) {
    assert.ok(roadmap.includes(`[\`${method.name}\`](/sdk/reference/python/pllm/${method.module.replace('pllm.', '').replaceAll('.', '-').replaceAll('_', '-')}/#${method.name.toLowerCase()})`), method.paper);
    assert.ok(roadmap.includes(`| ${method.status === 'implemented' ? 'Implemented' : 'Pending'} | ${method.gate} |`), method.paper);
  }
  for (const file of walk(path.join(siteRoot, 'content/docs/research')).filter((item) => item.endsWith('.mdx'))) {
    assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /\bR\d{2}\b/, file);
  }
});

test('SDK pages have checked examples or explicit API boundaries', () => {
  const sdkPages = pages.filter((page) => page.canonicalUrl.startsWith('/sdk/'));
  assert.ok(sdkPages.length > 0);

  for (const page of sdkPages) {
    const examples = [...page.content.matchAll(/```python[^\n]*\n([\s\S]*?)```/g)].map((match) => match[1]);
    const hasExample = page.content.includes('## Python SDK example');
    const hasBoundary = page.content.includes('No public Python API');
    assert.notEqual(hasExample, hasBoundary, `${page.canonicalUrl} must choose an example or no-API boundary`);
    if (hasBoundary) {
      assert.equal(examples.length, 0, `${page.canonicalUrl} no-API page has Python code`);
      assert.match(page.content, /\]\(\/(?:sdk\/reference\/status|research)\//);
      continue;
    }
    assert.ok(examples.length >= 1, `${page.canonicalUrl} must have a Python example`);
    assert.match(
      page.content,
      /^API: .*\(\/sdk\/reference\/python\/pllm\/(?:[a-z0-9-]+\/)?#objects-and-signatures\)/m,
      `${page.canonicalUrl} has no exact Python API link`,
    );

    for (const [index, example] of examples.entries()) {
      assert.match(example, /(?:from|import)\s+pllm\b/, `${page.canonicalUrl} example ${index + 1} does not use the SDK`);
      assert.match(example, /^\s*assert\s/m, `${page.canonicalUrl} example ${index + 1} has no checked result`);
      const parsed = spawnSync(
        'python3',
        ['-c', 'import ast, sys; ast.parse(sys.stdin.read())'],
        { encoding: 'utf8', input: example },
      );
      assert.equal(parsed.status, 0, `${page.canonicalUrl}: ${parsed.stderr}`);
    }
  }
});

test('inference option guides resolve real experiments and lead to serving', () => {
  for (const slug of [
    'index', 'compare', 'masked-linear', 'verified-masked-linear',
    'proprietary-guarded', 'proprietary-blinded', 'direct-fhe',
  ]) {
    const route = `/sdk/inference/${slug === 'index' ? '' : `${slug}/`}`;
    const content = byRoute.get(route)?.content;
    assert.ok(content, route);
    assert.match(content, /pllm gateway --local --experiment experiment\.py:experiment --trust-python/, route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.equal(examples.length, 1, route);
    const execution = spawnSync('uv', ['run', 'python', '-c', examples[0][1]], {
      cwd: path.join(siteRoot, '..'), encoding: 'utf8',
      env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
      timeout: 120_000,
    });
    assert.equal(execution.status, 0, `${route}:\n${execution.stdout}\n${execution.stderr}`);
  }
  const verified = byRoute.get('/sdk/inference/verified-masked-linear/').content;
  assert.ok(verified.includes('/research/papers/slalom/'));
});

test('experiment guides construct and round-trip supported configuration', () => {
  for (const slug of ['index', 'define', 'files', 'resolve', 'identity']) {
    const route = `/sdk/experiments/${slug === 'index' ? '' : `${slug}/`}`;
    const content = byRoute.get(route)?.content;
    assert.ok(content, route);
    assert.ok(content.includes('pllm gateway --local --experiment experiment.py:experiment --trust-python'), route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.equal(examples.length, 1, route);
    const execution = spawnSync('uv', ['run', 'python', '-c', examples[0][1]], {
      cwd: path.join(siteRoot, '..'), encoding: 'utf8',
      env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
      timeout: 120_000,
    });
    assert.equal(execution.status, 0, `${route}:\n${execution.stdout}\n${execution.stderr}`);
  }
});

test('relocated evaluation guides execute their real SDK examples', () => {
  for (const route of [
    '/sdk/evaluate/benchmark/', '/sdk/evaluate/search/',
    '/sdk/evaluate/evidence/', '/sdk/evaluate/assurance/',
  ]) {
    const content = byRoute.get(route)?.content;
    assert.ok(content, route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.ok(examples.length > 0, route);
    for (const [index, example] of examples.entries()) {
      const execution = spawnSync('uv', ['run', 'python', '-c', example[1]], {
        cwd: path.join(siteRoot, '..'), encoding: 'utf8',
        env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
        timeout: 120_000,
      });
      assert.equal(execution.status, 0,
        `${route} example ${index + 1}:\n${execution.stdout}\n${execution.stderr}`);
    }
  }
});

test('component option guides execute supported examples and cite bounded research', () => {
  const options = [
    ['/sdk/components/protocols/masked-linear/', 'pllm/masked-linear', '/research/papers/slalom/'],
    ['/sdk/components/protocols/guarded-linear/', 'pllm/guarded-linear/v1', null],
    ['/sdk/components/protocols/blinded-linear/', 'pllm/blinded-linear/v1', null],
    ['/sdk/components/protocols/direct-fhe/', 'pllm/direct-fhe', null],
    ['/sdk/components/protocols/secure-linear/', 'pllm/secure-linear/v1', null],
    ['/sdk/components/protocols/cleartext-linear/', 'pllm/cleartext-linear', null],
    ['/sdk/components/preparation/model-aware-corrections/', 'pllm/model-aware-corrections', null],
    ['/sdk/components/preparation/bfv-correlations/', 'pllm/bfv-correlations/v1', null],
    ['/sdk/components/preparation/he-authenticated-preprocessing/', 'pllm/he-authenticated-preprocessing', null],
    ['/sdk/components/correlation/seeded-expansion/', 'pllm/seeded-expansion', null],
    ['/sdk/components/kernels/cpu/', 'pllm/cpu', null],
    ['/sdk/components/nonlinear/arithmetic-garbling-silu-q7/', 'pllm/arithmetic-garbling-silu-q7/v1', '/research/papers/dash/'],
    ['/sdk/components/nonlinear/binary-table/', 'pllm/binary-table/v1', null],
    ['/sdk/components/nonlinear/r03-crt/', 'pllm/r03-crt/v1', '/research/papers/garbling-gadgets/'],
    ['/sdk/components/schedulers/bounded-independent-elements/', 'pllm/bounded-independent-elements/v1', null],
    ['/sdk/components/schedulers/scalar/', 'pllm/scalar/v1', null],
    ['/sdk/components/schedulers/independent-lanes/', 'pllm/independent-lanes/v1', null],
    ['/sdk/components/schedulers/chunked-independent-lanes/', 'pllm/chunked-independent-lanes/v1', null],
    ['/sdk/components/roles/inference/', 'pllm/inference', null],
    ['/sdk/components/state/client-local-kv/', 'pllm/client-local-kv', null],
    ['/sdk/components/verification/freivalds/', 'pllm/freivalds-verify/v1', '/research/papers/slalom/'],
    ['/sdk/components/verification/linear-integrity/', 'pllm/linear-integrity', null],
    ['/sdk/components/passes/kv-cache-eviction/', 'pllm/kv-cache-eviction', '/research/papers/mpcache/'],
  ];
  for (const [route, identity, citation] of options) {
    const content = byRoute.get(route)?.content;
    assert.ok(content, route);
    assert.ok(content.includes(identity), route);
    if (citation) assert.ok(content.includes(`](${citation})`), route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.equal(examples.length, 1, route);
    const execution = spawnSync('uv', ['run', 'python', '-c', examples[0][1]], {
      cwd: path.join(siteRoot, '..'), encoding: 'utf8',
      env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
      timeout: 120_000,
    });
    assert.equal(execution.status, 0, `${route}:\n${execution.stdout}\n${execution.stderr}`);
  }
  const inventory = spawnSync('uv', ['run', 'python', '-c',
    'import json; from pllm.components import list_component_classes; print(json.dumps(sorted(cls.describe().component for cls in list_component_classes())))'], {
    cwd: path.join(siteRoot, '..'), encoding: 'utf8',
  });
  assert.equal(inventory.status, 0, inventory.stderr);
  const metricIds = [
    'pllm/accuracy', 'pllm/communication', 'pllm/cost', 'pllm/energy',
    'pllm/latency', 'pllm/memory', 'pllm/perplexity', 'pllm/throughput',
  ];
  assert.deepEqual([...options.map(([, identity]) => identity), ...metricIds].sort(),
    JSON.parse(inventory.stdout), 'every built-in component must have an option guide or a metric option');
});

test('all metric option guides execute declarations without invented measurements', () => {
  for (const slug of [
    'latency', 'throughput', 'communication', 'memory',
    'energy', 'accuracy', 'perplexity', 'cost',
  ]) {
    const route = `/sdk/evaluate/metrics/${slug}/`;
    const content = byRoute.get(route)?.content;
    assert.ok(content?.includes(`pllm/${slug}`), route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.equal(examples.length, 1, route);
    const execution = spawnSync('uv', ['run', 'python', '-c', examples[0][1]], {
      cwd: path.join(siteRoot, '..'), encoding: 'utf8',
      env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
      timeout: 120_000,
    });
    assert.equal(execution.status, 0, `${route}:\n${execution.stdout}\n${execution.stderr}`);
  }
});

test('source option guides distinguish configured, imported, and runnable models', () => {
  for (const slug of [
    'hugging-face', 'safetensors', 'gguf', 'mlx', 'ollama', 'tiny',
  ]) {
    const route = `/sdk/models/sources/${slug}/`;
    const content = byRoute.get(route)?.content;
    assert.ok(content?.includes('## Python SDK example'), route);
    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)];
    assert.equal(examples.length, 1, route);
    const execution = spawnSync('uv', ['run', 'python', '-c', examples[0][1]], {
      cwd: path.join(siteRoot, '..'), encoding: 'utf8',
      env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
      timeout: 120_000,
    });
    assert.equal(execution.status, 0, `${route}:\n${execution.stdout}\n${execution.stderr}`);
  }
  assert.match(byRoute.get('/sdk/models/sources/gguf/')?.content ?? '', /gateway topology rejects/);
  assert.match(byRoute.get('/sdk/models/sources/ollama/')?.content ?? '', /not a model weight format admitted/);
  assert.match(byRoute.get('/sdk/models/sources/tiny/')?.content ?? '', /transport smoke test only/);
});

test('research experimentation guide bridges provenance, SDK, CLI, and matched evidence', () => {
  const route = '/research/recipes/experiments/';
  const guide = byRoute.get(route)?.content;
  assert.ok(guide, route);
  for (const destination of [
    '/research/papers/', '/research/methods/', '/research/evidence/',
    '/sdk/components/', '/sdk/reference/python/pllm/providers/', '/sdk/experiments/',
    '/sdk/experiments/files/', '/sdk/inference/', '/sdk/evaluate/search/',
    '/sdk/evaluate/benchmark/', '/cli/reference/gateway/local-experiments/',
    '/cli/reference/benchmark/run/', '/cli/reference/components/list/',
  ]) assert.ok(guide.includes(`](${destination})`), destination);
  assert.match(guide, /not\*\* executable PLLM dependencies/);
  assert.match(guide, /model fingerprint,[\s\S]*warm state match exactly/);
  for (const source of ['content/research.html', 'content/research/paper.mdx', 'content/docs/recipes/index.mdx']) {
    assert.ok(fs.readFileSync(path.join(siteRoot, source), 'utf8').includes(route), source);
  }
});

test('relocated advanced guides retain executable component walkthroughs', () => {
  const guides = new Map([
    ['/sdk/plans/compiler-internals/', [
      'KvCacheEviction', 'BinaryTableGatedMultiplyQ7',
      'R03CrtGatedMultiplyQ7', 'ScalarProtectedTensorSchedule',
    ]],
    ['/sdk/run/runtime-internals/', [
      'OpenAI', 'build_roles', 'Inference', 'ClientLocalKv', 'FreivaldsVerify',
    ]],
    ['/sdk/components/protocols/composition/', [
      'MaskedLinear', 'DirectFHE', 'GuardedLinear', 'BlindedLinear',
      'SecureLinear', 'CleartextLinear',
    ]],
    ['/sdk/inference/prepared-protocol/', [
      'MaskedLinear', 'ModelAwareCorrections', 'Inference', 'Cpu', 'FreivaldsVerify',
    ]],
    ['/sdk/components/preparation/reference-paths/', [
      'ModelAwareCorrections', 'BFVCorrelations', 'HEAuthenticatedPreprocessing',
      'TrustedPreprocessor',
    ]],
    ['/sdk/components/kernels/native-matrix/', ['Cpu', 'MaskedGEMM', 'CompiledMatrix', 'capabilities']],
    ['/sdk/components/nonlinear/garbling/', [
      'BinaryTableGatedMultiplyQ7', 'R03CrtGatedMultiplyQ7',
      'ScalarProtectedTensorSchedule', 'IndependentLanesProtectedTensorSchedule',
      'ChunkedIndependentLanesProtectedTensorSchedule',
    ]],
  ]);

  for (const [route, components] of guides) {
    const content = byRoute.get(route)?.content ?? '';
    assert.match(content, /^## (?:How|Pipeline selection|Methods and schedules) /m, `${route} has no guide introduction`);
    assert.match(content, /^## Python SDK example$/m, `${route} has no example section`);
    assert.doesNotMatch(content, /^\| .+ \| .+ \|/m, `${route} falls back to an inventory table`);
    assert.match(content, /\/sdk\/reference\/python\/pllm\/[a-z0-9-]+\//, `${route} has no module reference`);
    for (const component of components) {
      const linkToken = component === 'capabilities' ? 'capabilities()' : component;
      const heading = [...content.matchAll(/^### .+$/gm)].find((match) =>
        match[0].includes(`\`${linkToken}\``),
      );
      assert.ok(heading, `${route} has no ${component} subsection`);
      const nextHeading = content.indexOf('\n### ', heading.index + heading[0].length);
      const subsection = content.slice(heading.index, nextHeading < 0 ? undefined : nextHeading);
      assert.ok(
        subsection.includes(`[\`${linkToken}\`](/sdk/reference/python/pllm/`),
        `${route} does not link ${component} to its API reference in its subsection`,
      );
      assert.match(subsection, /```python\n/, `${route} has no ${component} example in its subsection`);
    }

    const examples = [...content.matchAll(/```python\n([\s\S]*?)```/g)].map((match) => match[1]);
    for (const [index, example] of examples.entries()) {
      const execution = spawnSync('uv', ['run', 'python', '-c', example], {
        cwd: path.join(siteRoot, '..'),
        encoding: 'utf8',
        env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
        timeout: 120_000,
      });
      assert.equal(
        execution.status,
        0,
        `${route} example ${index + 1} did not execute:\n${execution.stdout}\n${execution.stderr}`,
      );
    }
  }
});

test('runtime-backed component examples reach a private inference workflow', () => {
  const experimentComponents = new Map([
    ['/sdk/components/protocols/composition/', [
      'MaskedLinear', 'DirectFHE', 'GuardedLinear', 'BlindedLinear',
    ]],
    ['/sdk/inference/prepared-protocol/', [
      'MaskedLinear', 'ModelAwareCorrections', 'Inference', 'Cpu', 'FreivaldsVerify',
    ]],
    ['/sdk/components/preparation/reference-paths/', ['ModelAwareCorrections']],
    ['/sdk/components/kernels/native-matrix/', ['Cpu']],
    ['/sdk/run/runtime-internals/', ['build_roles', 'Inference', 'FreivaldsVerify']],
  ]);

  for (const [route, components] of experimentComponents) {
    const content = byRoute.get(route)?.content ?? '';
    assert.ok(
      content.includes('optional `he` dependency'),
      `${route} does not disclose the local gateway dependency`,
    );
    assert.ok(
      content.includes('pllm gateway --local') || route === '/sdk/run/runtime-internals/',
      `${route} does not show the gateway execution path`,
    );
    for (const component of components) {
      const heading = [...content.matchAll(/^### .+$/gm)].find((match) =>
        match[0].includes(`\`${component}\``),
      );
      assert.ok(heading, `${route} has no ${component} subsection`);
      const nextHeading = content.indexOf('\n### ', heading.index + heading[0].length);
      const subsection = content.slice(heading.index, nextHeading < 0 ? undefined : nextHeading);
      assert.match(subsection, /Experiment\(/, `${route} leaves ${component} outside an Experiment`);
    }
  }

  const runtime = byRoute.get('/sdk/run/runtime-internals/').content;
  const openAIStart = runtime.indexOf('### Send a synchronous streaming request with `OpenAI`');
  const openAIEnd = runtime.indexOf('\n### ', openAIStart + 5);
  assert.match(runtime.slice(openAIStart, openAIEnd), /responses\.create\(/);
});

test('primary reader journeys cross areas at the decision point', () => {
  const journeys = new Map([
    ['learn/first-local-benchmark.mdx', [
      '/cli/reference/benchmark/run/',
      '/research/evidence/',
    ]],
    ['learn/first-private-request.mdx', [
      '/sdk/experiments/files/',
      '/sdk/components/protocols/',
      '/learn/integrations/',
    ]],
    ['learn/index.mdx', ['/learn/integrations/']],
    ['cli/index.mdx', ['/learn/integrations/local-gateway/']],
    ['sdk/index.mdx', ['/learn/integrations/openai-python/']],
    ['sdk/run/lifecycle.mdx', ['/learn/integrations/local-gateway/']],
    ['learn/concepts/privacy-and-threat-models.mdx', ['/research/evidence/']],
    ['sdk/experiments/files.mdx', [
      '/cli/reference/config/show/',
      '/cli/reference/config/export/',
    ]],
    ['sdk/evaluate/benchmark.mdx', [
      '/cli/reference/benchmark/run/',
      '/research/evidence/',
    ]],
    ['research/sources.mdx', [
      '/research/papers/',
      '/research/backlog/',
    ]],
    ['research/methods.mdx', [
      '/sdk/plans/',
      '/sdk/components/',
      '/sdk/reference/components/',
    ]],
    ['research/evidence.mdx', [
      '/sdk/evaluate/evidence/',
      '/sdk/evaluate/assurance/',
    ]],
  ]);

  for (const [source, destinations] of journeys) {
    const content = fs.readFileSync(path.join(siteRoot, 'content/docs', source), 'utf8');
    for (const destination of destinations) {
      assert.ok(content.includes(`](${destination})`), `${source} should link to ${destination}`);
    }
  }
});

test('homepage leads with the trusted gateway and separates operated services', () => {
  const home = fs.readFileSync(path.join(siteRoot, 'content/home.html'), 'utf8');
  assert.match(home, /aria-selected="true" data-phase="start-local"/);
  assert.match(home, /uv tool install pllm\.run/);
  assert.doesNotMatch(home, /pllm\.run\[he\]/);
  assert.match(home, /href="\/learn\/installation\/">installation guide/);
  assert.match(home, /pllm gateway --local --model Qwen\/Qwen2\.5-0\.5B-Instruct/);
  assert.doesNotMatch(home, /Verified output:|"content": "Hello\."/);
  assert.match(home, /pllm gateway --config client\.toml/);
  assert.match(home, /pllm serve inference --config inference\.json/);
  assert.match(home, /pllm serve preparation --config preparation\.json/);
  assert.match(home, /co-location does not provide role separation or non-collusion/);
});

test('homepage consumption modes separate the native client from the trusted gateway', () => {
  const home = fs.readFileSync(path.join(siteRoot, 'content/home.html'), 'utf8');
  const start = home.indexOf('id="consume-title"');
  const end = home.indexOf('id="run-research"');
  const consumption = home.slice(start, end);
  assert.ok(start > home.indexOf('id="start-building"'));
  assert.ok(end > start);
  assert.match(home, /id="start-building"[^]*?<\/section>\s*<section class="home-section home-shell home-consume-section"/);

  const modes = [
    ['pllm', '/sdk/run/embedded-gateway/'],
    ['responses', '/learn/integrations/responses-api/'],
    ['chat', '/learn/integrations/chat-completions/'],
    ['openai', '/learn/integrations/openai-python/'],
    ['agents', '/learn/integrations/openai-agents/'],
    ['codex', '/learn/integrations/codex/'],
    ['opencode', '/learn/integrations/opencode/'],
  ];
  for (const [panel, guide] of modes) {
    const tab = `consume-${panel}`;
    assert.match(consumption, new RegExp(`role="tab" id="tab-${tab}" aria-controls="panel-${tab}"`));
    assert.match(consumption, new RegExp(`id="panel-${tab}" role="tabpanel" aria-labelledby="tab-${tab}"`));
    const snippet = consumption.match(new RegExp(`<div class="home-code" id="panel-${tab}"[^>]*data-panel="${tab}"[^>]*>([^]*?)</div>`))?.[1];
    assert.ok(snippet, tab);
    if (panel !== 'pllm') assert.match(snippet, /http:\/\/127\.0\.0\.1:8080\/v1/);
    if (panel !== 'pllm') assert.match(snippet, /local/);
    assert.match(consumption, new RegExp(`href="${guide}"`));
  }

  assert.doesNotMatch(consumption, /preparation_(?:url|base_url|api_key)|inference_(?:url|base_url|api_key)/);
  assert.match(consumption, /from pllm import OpenAI/);
  assert.match(consumption, /wire_api = "responses"/);
  assert.match(consumption, /web_search = "disabled"/);
  assert.match(consumption, /env_key = "PLLM_GATEWAY_API_KEY" # Set to local\./);
  assert.match(consumption, /"npm": "@ai-sdk\/openai-compatible"/);
  assert.match(consumption, /"enabled_providers": \["pllm"\]/);
  assert.match(consumption, /PLLM \(Chat Completions API\)/);
});

test('nested Fumadocs navigation has real overviews and declared children', () => {
  const root = JSON.parse(fs.readFileSync(path.join(siteRoot, 'content/docs/meta.json'), 'utf8'));
  assert.deepEqual(root.pages, ['learn', 'cli', 'sdk', 'research']);
  assert.equal(new Set(root.pages).size, root.pages.length);
  assert.ok(!root.pages.some((page) => page.startsWith('---')));
  for (const branch of navigation.publicationGroups.flatMap((group) =>
    group.entries.filter((entry) => entry.visible !== false).map((entry) => entry.sidebar).filter(Boolean))) {
    const directory = path.join(siteRoot, 'content/docs', branch);
    const meta = JSON.parse(fs.readFileSync(path.join(directory, 'meta.json'), 'utf8'));
    if (branch === 'research') {
      assert.ok(!meta.pages.includes('index'));
    } else {
      assert.equal(meta.pages[0], 'index', branch);
      assert.ok(fs.existsSync(path.join(directory, 'index.mdx')), branch);
    }
    for (const child of meta.pages) {
      if (child === 'index') continue;
      if (child.startsWith('---') || child.startsWith('[')) continue;
      assert.ok(
        fs.existsSync(path.join(directory, `${child}.mdx`)) || fs.existsSync(path.join(directory, child, 'index.mdx')),
        `${branch}/${child}`,
      );
    }
  }
});

test('each top-level journey owns an isolated Fumadocs sidebar', () => {
  const readMeta = (branch) => JSON.parse(
    fs.readFileSync(path.join(siteRoot, 'content/docs', branch, 'meta.json'), 'utf8'),
  );
  const learn = readMeta('learn');
  const cli = readMeta('cli');
  const sdk = readMeta('sdk');
  const research = readMeta('research');

  for (const [branch, meta] of Object.entries({ learn, cli, sdk })) {
    assert.equal(meta.root, true, branch);
    assert.equal(meta.pages[0], 'index', branch);
  }
  assert.equal(research.root, true);
  assert.ok(!research.pages.includes('index'));
  assert.deepEqual(learn.pages.slice(-2), ['concepts', 'integrations']);
  assert.equal(learn.title, 'Start');
  assert.deepEqual(navigation.publicationGroups.find((group) => group.id === 'get-started')
    .entries.slice(0, 3).map((entry) => entry.label), ['Start', 'Core concepts', 'Working with PLLM']);
  assert.deepEqual(cli.pages.slice(-1), ['../reference/cli']);
  assert.deepEqual(sdk.pages, [
    'index', 'experiments', 'models', 'inference', 'components',
    'run', 'plans', 'evaluate', 'extend', '../reference',
  ]);
  assert.deepEqual(research.pages.slice(-3), ['../recipes', 'papers', 'records']);

  for (const branch of [
    'learn/integrations', 'learn/concepts', 'reference/cli',
    'sdk/experiments', 'sdk/models', 'sdk/inference', 'sdk/components',
    'sdk/run', 'sdk/plans', 'sdk/evaluate', 'sdk/extend',
    'reference', 'recipes', 'research/papers',
    'research/records',
  ]) {
    const meta = readMeta(branch);
    assert.equal(meta.root, true, branch);
    assert.equal(meta.pages[0], 'index', branch);
    assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs', branch, 'index.mdx')), branch);
  }
  assert.equal(readMeta('learn/concepts').title, 'Core concepts');
  const sectionOrder = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.match(sectionOrder, /\['Start', 'Core concepts', 'Working with PLLM'\]/);
  assert.match(sectionOrder, /\['SDK', 'Experiments', 'Models', 'Inference options', 'Component options',/);

  const docsLayout = fs.readFileSync(path.join(siteRoot, 'components/docs-shell.tsx'), 'utf8');
  const source = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.ok(docsLayout.includes('tabs={false}'));
  assert.ok(docsLayout.includes('<DocsSectionSwitcher'));
  assert.ok(source.includes('areaFolders'));
  assert.ok(source.includes('getAreaSections'));
  assert.ok(source.includes('const sections: Folder[]'));
});

test('global navigation and local Fumadocs sidebars have separate ownership', () => {
  const landing = fs.readFileSync(path.join(siteRoot, 'content/docs/learn/index.mdx'), 'utf8');
  const header = fs.readFileSync(path.join(siteRoot, 'lib/navigation.ts'), 'utf8');
  const chrome = fs.readFileSync(path.join(siteRoot, 'components/site-chrome.tsx'), 'utf8');
  const mobileHeader = fs.readFileSync(path.join(siteRoot, 'components/docs-mobile-header.tsx'), 'utf8');
  const docsLayout = fs.readFileSync(path.join(siteRoot, 'components/docs-shell.tsx'), 'utf8');
  const routeLayout = fs.readFileSync(path.join(siteRoot, 'app/(docs)/[...slug]/layout.tsx'), 'utf8');
  const researchLayout = fs.readFileSync(path.join(siteRoot, 'app/research/layout.tsx'), 'utf8');
  const css = fs.readFileSync(path.join(siteRoot, 'app/global.css'), 'utf8');
  assert.ok(header.includes("import navigation from '@/navigation.json'"));
  assert.ok(chrome.includes('mainNavigation.map'));
  assert.doesNotMatch(chrome, /mobile-doc-links|pages\.map/);
  assert.ok(mobileHeader.includes('mainNavigation.map'));
  assert.doesNotMatch(docsLayout, /links=|mainNavigation/);
  assert.ok(docsLayout.includes('component: <DocsMobileHeader showMenu={sidebar} />'));
  assert.ok(docsLayout.includes('slots={{ navTitle: DocsSidebarTitle }}'));
  assert.ok(docsLayout.includes('getAreaPageTree(area)'));
  assert.ok(routeLayout.includes('navigationAreaForPathname(pathname)'));
  assert.ok(routeLayout.includes('<DocsShell area={area}>'));
  assert.ok(researchLayout.includes('<DocsShell area="Research" research>'));
  assert.match(css, /\.content-site-header\s*{\s*display:\s*none/);
  assert.ok(landing.includes('Install PLLM'));
  for (const group of navigation.publicationGroups) {
    for (const entry of group.entries) {
      assert.ok(byRoute.has(entry.href), entry.href);
    }
  }
  assert.deepEqual(navigation.main.map(({ label, href }) => ({ label, href })), [
    { href: '/learn/', label: 'Learn' },
    { href: '/cli/', label: 'CLI' },
    { href: '/sdk/', label: 'SDK' },
    { href: '/research/', label: 'Research' },
  ]);

  const owner = (route) => navigation.main
    .flatMap((item) => item.owns.map((prefix) => ({ item, prefix })))
    .filter(({ prefix }) => route === prefix || route.startsWith(`${prefix}/`))
    .sort((left, right) => right.prefix.length - left.prefix.length)[0]?.item.label;
  assert.equal(owner('/learn/installation'), 'Learn');
  assert.equal(owner('/cli/reference'), 'CLI');
  assert.equal(owner('/sdk/models/families/qwen2'), 'SDK');
  assert.equal(owner('/sdk/evaluate/benchmark'), 'SDK');
  assert.equal(owner('/research/recipes/compare'), 'Research');
  assert.equal(owner('/research/recipes/build-method'), 'Research');
  assert.equal(owner('/research/recipes/agent-map'), 'Research');
  assert.equal(owner('/research/whitepaper'), 'Research');
  assert.equal(owner(''), undefined);
});

test('documentation sections and pages share one bounded, state-aware sidebar scroll', () => {
  const header = fs.readFileSync(path.join(siteRoot, 'components/docs-mobile-header.tsx'), 'utf8');
  const css = fs.readFileSync(path.join(siteRoot, 'app/global.css'), 'utf8');
  const source = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.ok(header.includes('aria-controls="nd-sidebar-mobile"'));
  assert.ok(header.includes('aria-expanded={open}'));
  assert.ok(header.includes("event.key === 'Escape'"));
  assert.match(css, /\.docs-area-switcher nav\s*{[^}]*display:\s*none/s);
  assert.match(css, /\.docs-area-switcher\[open\] nav\s*{\s*display:\s*grid/s);
  const sectionList = css.match(/\.docs-section-switcher-list\s*{([^}]*)}/)?.[1] ?? '';
  assert.doesNotMatch(sectionList, /max-height|overflow-y/);
  assert.match(css, /\.docs-root :is\(#nd-sidebar, #nd-sidebar-mobile\)\s*{[^}]*overflow-y:\s*auto/s);
  assert.match(css, /\[data-radix-scroll-area-viewport\]\s*{[^}]*height:\s*auto !important;[^}]*overflow:\s*visible !important/s);
  assert.match(css, /#nd-sidebar-mobile\[data-state='open'\]/);
  assert.ok(source.includes("return { name: area, children: researchFolders() }"));
});

test('authored navigation is tracked and generated navigation is rebuilt', () => {
  const ignore = fs.readFileSync(path.join(siteRoot, '../.gitignore'), 'utf8');
  assert.ok(ignore.includes('!docs/content/docs/sdk/models/**'));
  assert.ok(ignore.includes('docs/content/docs/reference/python/pllm/'));
  assert.ok(ignore.includes('papers/'));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/sdk/models/families/meta.json')));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/research/papers/meta.json')));
});

test('authored copy uses direct technical English and SDK status is derived from public modules', () => {
  const generated = new Set([
    'content/docs/reference/components.mdx',
    'content/docs/reference/python/pllm/index.mdx',
  ]);
  for (const page of readSearchPages()) {
    if (page.sourcePath.startsWith('content/docs/reference/cli/') || generated.has(page.sourcePath)) continue;
    assert.doesNotMatch(page.content, /\b(?:inert|non-normative|organisations|behaviour|catalogue|authorise|optimise|centre)\b/i, page.canonicalUrl);
  }
  const status = byRoute.get('/sdk/reference/status/').content;
  const modules = JSON.parse(fs.readFileSync(path.join(siteRoot, 'content/docs/reference/python/pllm/meta.json'), 'utf8')).pages;
  const planned = JSON.parse(fs.readFileSync(path.join(siteRoot, '..', 'python/pllm/components/planned_methods.json'), 'utf8')).entries;
  const rows = [...status.matchAll(/^\| \[`(pllm(?:\.[^`]+)?)`\]\([^)]+\) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+%|—) \|$/gm)];
  assert.equal(rows.length, modules.length);
  assert.equal(rows.reduce((sum, row) => sum + Number(row[3]), 0), planned.length);
  for (const row of rows) {
    assert.equal(Number(row[2]) + Number(row[3]), Number(row[4]), row[1]);
  }
  assert.match(status, /\*\*not\*\* a whole-decoder, security,/);
});

test('every docs source is publication-discovered without registry duplication', () => {
  const sourcePaths = new Set(readSearchPages().map((page) => page.sourcePath));
  const visit = (directory) => fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const candidate = path.join(directory, entry.name);
    return entry.isDirectory() ? visit(candidate) : candidate.endsWith('.mdx') ? [candidate] : [];
  });
  for (const file of visit(path.join(siteRoot, 'content/docs'))) {
    const relative = path.relative(siteRoot, file).replaceAll(path.sep, '/');
    assert.ok(sourcePaths.has(relative), relative);
  }
});

test('generated CLI pages are publication-discovered and sidebar-reachable', () => {
  const cliRoot = path.join(siteRoot, 'content/docs/reference/cli');
  const generated = walk(cliRoot)
    .filter((file) => file.endsWith('.mdx'))
    .map((file) => path.relative(siteRoot, file).replaceAll(path.sep, '/'));
  const published = new Map(readSearchPages().map((page) => [page.sourcePath, page]));
  const reachable = new Set();
  const visit = (directory) => {
    const meta = JSON.parse(fs.readFileSync(path.join(directory, 'meta.json'), 'utf8'));
    for (const entry of meta.pages) {
      const childDirectory = path.join(directory, entry);
      const source = entry === 'index'
        ? path.join(directory, 'index.mdx')
        : fs.existsSync(path.join(childDirectory, 'index.mdx'))
          ? path.join(childDirectory, 'index.mdx')
          : path.join(directory, `${entry}.mdx`);
      reachable.add(path.relative(siteRoot, source).replaceAll(path.sep, '/'));
      if (entry !== 'index' && fs.existsSync(path.join(childDirectory, 'meta.json'))) visit(childDirectory);
    }
  };
  visit(cliRoot);

  assert.deepEqual([...reachable].sort(), generated.sort());
  for (const sourcePath of generated) {
    const page = published.get(sourcePath);
    assert.ok(page, sourcePath);
    assert.deepEqual(page.sourcePaths, [sourcePath, '../python/pllm/_cli/app.py']);
    assert.deepEqual(page.testPaths, ['../tests/test_cli.py', '../tests/test_developer_reference.py']);
  }
});

test('CLI navigation lists tasks directly with command and cross-area links', () => {
  const cliMeta = JSON.parse(
    fs.readFileSync(path.join(siteRoot, 'content/docs/cli/meta.json'), 'utf8'),
  );
  for (const page of ['private-inference', 'provider-roles', 'benchmarking', 'inspect-and-research']) {
    assert.ok(cliMeta.pages.indexOf(page) < cliMeta.pages.indexOf('../reference/cli'));
  }

  const expected = new Map([
    ['/cli/private-inference/', ['/cli/reference/gateway/', '/learn/integrations/', '/sdk/run/embedded-gateway/']],
    ['/cli/provider-roles/', ['/cli/reference/serve/inference/', '/cli/reference/serve/preparation/', '/sdk/run/deployment/']],
    ['/cli/benchmarking/', ['/cli/reference/benchmark/run/', '/cli/reference/dev/dashboard/', '/research/records/']],
    ['/cli/inspect-and-research/', ['/cli/reference/config/', '/cli/reference/components/', '/research/papers/', '/research/backlog/', '/research/methods/']],
  ]);
  for (const [route, links] of expected) {
    const page = byRoute.get(route);
    assert.ok(page, route);
    for (const href of links) assert.ok(page.content.includes(`](${href})`), `${route} -> ${href}`);
  }
});

test('all local links and fragments resolve', () => assert.deepEqual(validate().errors, []));

test('standalone publication pages remain in content graph', () => {
  assert.deepEqual(standalone.map((page) => page.canonicalUrl), ['/', '/research/']);
  assert.deepEqual(readSearchPages().filter((page) => page.sourcePath.endsWith('.html')).map((page) => page.id), ['pllm.home', 'pllm.research']);
});

test('documented commands use only the current CLI and development dashboard', () => {
  const sources = readSearchPages().map((page) => page.content).join('\n');
  assert.ok(sources.includes('pllm config show examples/pllm.yaml'));
  assert.ok(sources.includes('pllm components list'));
  assert.ok(sources.includes('pllm dev dashboard'));
  assert.ok(sources.includes('pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct'));
  assert.ok(sources.includes('pllm gateway --config client.toml'));
  assert.ok(sources.includes('pllm serve inference --config inference.json'));
  assert.ok(sources.includes('pllm serve preparation --config preparation.json'));
  assert.doesNotMatch(sources, /\bpllm research\b/);
  assert.doesNotMatch(sources, /pllm benchmark dashboard/);
  assert.doesNotMatch(sources, /<pre>[^]*?pllm (?:preparation serve|configure|chat|run|model lower|plan (?:check|compile|show)|party serve|benchmark (?:run|search|compare)|assure run|init)\b[^]*?<\/pre>/);
  for (const block of sources.match(/```(?:bash|sh|shell|console|text)?\n[^]*?```/g) ?? []) {
    assert.doesNotMatch(block, /^pllm (?:preparation serve|configure|chat|run|model lower|plan (?:check|compile|show)|party serve|benchmark (?:run|search|compare)|assure run|init)\b/m);
  }
});

test('serving and consumer guides preserve tested integration boundaries', () => {
  const integrationRoutes = [
    '/learn/integrations/',
    '/learn/integrations/local-gateway/',
    '/learn/integrations/responses-api/',
    '/learn/integrations/chat-completions/',
    '/learn/integrations/openai-python/',
    '/learn/integrations/openai-agents/',
    '/learn/integrations/codex/',
    '/learn/integrations/opencode/',
  ];
  for (const route of integrationRoutes) assert.ok(byRoute.has(route), route);

  const overview = byRoute.get('/learn/integrations/').content;
  for (const route of ['/v1/responses', '/v1/responses/compact', '/v1/chat/completions', '/v1/models']) {
    assert.ok(overview.includes(route), route);
  }
  for (const value of ['plaintext prompts', 'tool schemas', 'tool arguments', 'tool results']) {
    assert.ok(overview.includes(value), value);
  }
  assert.ok(overview.includes('does not establish operator separation or\nnon-collusion'));

  const local = byRoute.get('/learn/integrations/local-gateway/').content;
  const lifecycle = [
    'pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct',
    'pllm gateway --config client.toml',
    'pllm serve inference --config inference.json',
    'pllm serve preparation --config preparation.json',
  ];
  for (const command of lifecycle) assert.ok(local.includes(command), command);
  assert.match(local, /binds to `127\.0\.0\.1:8080`/);
  assert.match(local, /minimal `inference\.json`/);
  for (const block of local.match(/```json\n([^]*?)```/g) ?? []) {
    assert.doesNotThrow(() => JSON.parse(block.replace(/^```json\n|```$/g, '')));
  }

  const conformance = byRoute.get('/learn/integrations/responses-api/').content;
  assert.ok(conformance.includes('2026-04-24'));
  assert.ok(conformance.includes('[OpenResponses](https://www.openresponses.org/)'));
  assert.ok(conformance.includes('92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c'));
  assert.ok(conformance.includes('does not prove\nsupport for every Responses API field, hosted tool, transport, or model modality'));

  const codex = byRoute.get('/learn/integrations/codex/').content;
  assert.ok(codex.includes('wire_api = "responses"'));
  assert.ok(codex.includes('web_search = "disabled"'));
  assert.ok(codex.includes('provider-built-in web\nsearch is unsupported'));

  const opencode = byRoute.get('/learn/integrations/opencode/').content;
  assert.ok(opencode.includes('"enabled_providers": ["pllm"]'));
  assert.ok(opencode.includes('"npm": "@ai-sdk/openai-compatible"'));
  assert.ok(opencode.includes('custom-tool declarations map to local function tools'));
  assert.ok(opencode.includes('has not established acceptance of every external tool binary'));
  const opencodeConfig = opencode.match(/```json\n([^]*?)```/)?.[1];
  assert.ok(opencodeConfig);
  assert.doesNotThrow(() => JSON.parse(opencodeConfig));

  for (const route of ['/learn/integrations/openai-python/', '/learn/integrations/openai-agents/']) {
    const content = byRoute.get(route).content;
    const examples = [...content.matchAll(/```python\n([^]*?)```/g)].map((match) => match[1]);
    assert.equal(examples.length, 1, route);
    const parsed = spawnSync(
      'python3',
      ['-c', 'import ast, sys; ast.parse(sys.stdin.read())'],
      { encoding: 'utf8', input: examples[0] },
    );
    assert.equal(parsed.status, 0, `${route}: ${parsed.stderr}`);
    assert.match(content, /passes? (?:a )?repository (?:integration )?tests?/);
  }
});

test('documented PLLM lifecycle commands parse', () => {
  const commands = [
    'pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct',
    'pllm gateway --config client.toml',
    'pllm serve inference --config inference.json',
    'pllm serve preparation --config preparation.json',
  ];
  const program = [
    'import shlex',
    'from pllm._cli.app import build_parser',
    `commands = ${JSON.stringify(commands)}`,
    'for command in commands:',
    '    build_parser().parse_args(shlex.split(command)[1:])',
  ].join('\n');
  const result = spawnSync('python3', ['-c', program], {
    cwd: path.join(siteRoot, '..'),
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: path.join(siteRoot, '..', 'python') },
  });
  assert.equal(result.status, 0, result.stderr);
});

test('generated references do not expose repository maintenance instructions', () => {
  const generated = [
    ...walk(path.join(siteRoot, 'content/docs/reference/cli'))
      .filter((file) => file.endsWith('.mdx'))
      .map((file) => path.relative(path.join(siteRoot, 'content/docs/reference'), file).replace(/\.mdx$/, '')),
    'python/pllm/index', 'components',
  ];
  for (const name of generated) {
    const source = fs.readFileSync(path.join(siteRoot, `content/docs/reference/${name}.mdx`), 'utf8');
    assert.doesNotMatch(source, /Generated by .*do not edit/i);
  }
  assert.ok(byRoute.get('/sdk/components/').content.includes('pllm components list'));
  assert.equal(byRoute.get('/research/clean-room/').title, 'Contribute a clean-room reimplementation');
});

test('component catalog exposes independent status axes', () => {
  const components = byRoute.get('/sdk/reference/components/').content;
  for (const heading of ['Identity/version', 'Lifecycle', 'Implementation maturity', 'Provenance', 'Input/output representation', 'Roles/topology', 'Privacy/assurance', 'Model/operator coverage', 'Evidence/cohort', 'Known limitations']) {
    assert.ok(components.toLowerCase().includes(heading.toLowerCase()), heading);
  }
  assert.ok(components.includes('not recorded'));
  assert.ok(components.includes('no applicable evidence'));
});

test('research publication workflow cites source and separates PLLM adaptation', () => {
  const methods = byRoute.get('/research/methods/').content;
  const publications = byRoute.get('/research/publications/').content;
  assert.ok(methods.includes('crates/pllm-models/src/cache.rs'));
  assert.ok(methods.includes('pllm/kv-cache-eviction'));
  assert.ok(methods.includes('structural adaptation'));
  assert.ok(methods.includes('does not reproduce the paper'));
  for (const phrase of ['comparable evidence', 'assurance', 'explicit limitations', 'distinct method']) assert.ok(publications.includes(phrase), phrase);
  assert.ok(publications.includes('/research/paper/'));
  assert.ok(publications.includes('/research/whitepaper/'));
});

test('claim language guard rejects unsupported absolutes', () => {
  for (const page of readSearchPages()) {
    assert.doesNotMatch(page.content, /\b(?:universally secure|proven private|verified secure|fully reproduced|fastest|best-performing)\b/i, page.canonicalUrl);
  }
});

test('public copy uses standard technical English', () => {
  for (const page of readSearchPages()) {
    assert.doesNotMatch(
      page.content,
      /\b(?:inert|non-normative|organisations|behaviour|catalogue|authorise|optimise|centre)\b/i,
      page.canonicalUrl,
    );
  }
});

test('technical paper keeps historical and tiny verified cohorts separate and reports artifact medians', () => {
  const root = path.join(siteRoot, '..');
  const paper = fs.readFileSync(path.join(root, 'paper/manuscript.md'), 'utf8');
  const whitepaper = fs.readFileSync(path.join(root, 'paper/whitepaper.md'), 'utf8');
  const citation = fs.readFileSync(path.join(root, 'CITATION.cff'), 'utf8');
  const historical = JSON.parse(fs.readFileSync(path.join(siteRoot, 'evidence/current-runtime-2026-09-11.json'), 'utf8'));
  const tiny = ['baseline', 'verified'].map((variant) => JSON.parse(fs.readFileSync(
    path.join(siteRoot, `evidence/slalom-freivalds-tiny-${variant}.json`), 'utf8',
  )));

  for (const result of historical.results) {
    const m = result.median;
    const row = [
      `${result.context_tokens} / ${result.output_tokens[0]}`,
      m.ttft_seconds.toFixed(3), m.full_seconds.toFixed(3),
      (m.client_io_bytes / 1e6).toFixed(2),
    ];
    assert.ok(paper.includes(`| ${row.join(' | ')} |`), row.join(' | '));
    assert.ok(paper.includes(m.online_seconds.toFixed(3)));
    assert.ok(paper.includes((result.correction_push_bytes[0] / 1e6).toFixed(2)));
  }
  for (const record of tiny) {
    assert.ok(paper.includes(`${record.metrics[0].value.toFixed(3)} s`));
    assert.equal(record.repetitions, 1);
    assert.equal(record.plan_lock_digest, null);
  }
  assert.match(paper, /historical masked-protocol measurements are separate/i);
  assert.match(paper, /\*\*not\*\* a security proof or a matched external SOTA benchmark/i);
  assert.match(paper, /no cited method has pinned-Qwen comparative evidence/i);
  assert.match(whitepaper, /\*\*has\s+not measured prices, energy, or an economic return\*\*/i);
  assert.match(whitepaper, /high-performance private LLM multi-party inference runtime and\s+extensible autonomous research harness/i);
  assert.doesNotMatch(whitepaper, /\$\$/);
  for (const [title, source] of [
    ['PLLM: Private Multi-Party Inference and an Extensible Research Harness', whitepaper],
    ['PLLM: Private Multi-Party LLM Inference and Evidence-Bound Research Composition', paper],
  ]) {
    assert.ok(source.includes(`title: "${title}"`));
    assert.ok(citation.includes(`title: "${title}"`));
  }
});

test('paper, whitepaper, evidence, and generated CLI help remain downloadable', () => {
  for (const file of ['paper.pdf', 'paper-source.zip', 'whitepaper.pdf', 'whitepaper-source.zip', 'current-runtime-2026-09-11.json', 'evidence.zip', 'cli-help.txt']) {
    assert.ok(fs.statSync(path.join(siteRoot, 'public/downloads', file)).size > 0, file);
  }
});

test('whitepaper figures are readable, exported, and backed by pinned baseline runs', () => {
  const paperRoot = path.join(siteRoot, '..', 'paper');
  const web = fs.readFileSync(path.join(siteRoot, 'content/research/whitepaper.mdx'), 'utf8');
  const record = JSON.parse(fs.readFileSync(path.join(siteRoot, 'evidence/current-runtime-2026-09-11.json'), 'utf8'));
  const svg = fs.readFileSync(path.join(paperRoot, 'figures/qwen-baseline.svg'), 'utf8');
  for (const name of ['mechanics', 'research-loop', 'qwen-baseline']) {
    const source = fs.readFileSync(path.join(paperRoot, `figures/${name}.png`));
    const published = fs.readFileSync(path.join(siteRoot, `public/downloads/figures/${name}.png`));
    assert.ok(source.length > 1000, name);
    assert.deepEqual(source, published, name);
    assert.ok(web.includes(`<img src="/downloads/figures/${name}.png" alt="`), name);
  }
  for (const result of record.results) {
    assert.ok(svg.includes(`${result.median.full_seconds.toFixed(3)} s`));
    assert.ok(svg.includes(`${result.context_tokens} in / ${result.output_tokens[0]} out`));
  }
});

test('static export, local assets, and canonical metadata remain configured', () => {
  const nextConfig = fs.readFileSync(path.join(siteRoot, 'next.config.mjs'), 'utf8');
  assert.ok(nextConfig.includes("process.env.NODE_ENV !== 'development'"));
  assert.ok(nextConfig.includes("config.output = 'export'"));
  assert.ok(
    nextConfig.includes(
      "allowedDevOrigins: ['127.0.0.1', 'macbookblair.local', 'macbookblair.tail858b09.ts.net']",
    ),
  );
  assert.ok(!fs.readFileSync(path.join(siteRoot, 'app/layout.tsx'), 'utf8').includes('next/font/google'));
  for (const file of ['app/page.tsx', 'app/research/page.tsx', 'app/research/[...slug]/page.tsx', 'app/_docs-page.tsx']) {
    assert.ok(fs.readFileSync(path.join(siteRoot, file), 'utf8').includes('canonical'), file);
  }
  assert.ok(fs.existsSync(path.join(siteRoot, 'app/(docs)/[...slug]/page.tsx')));
  assert.ok(!fs.existsSync(path.join(siteRoot, 'app/docs')));
});

test('public source links use the canonical repository owner', () => {
  const home = fs.readFileSync(path.join(siteRoot, 'content/home.html'), 'utf8');
  assert.ok(home.includes('https://github.com/blairhudson/pllm'));
  assert.ok(!home.includes('github.com/probabilistic-alchemy'));
});

test('canonical trailing-slash routes resolve through the Fumadocs source', async () => {
  const { fumadocsHref } = await import('../lib/docs-routes.mjs');
  assert.equal(fumadocsHref('/sdk/configuration/'), '/sdk/configuration');
  assert.equal(fumadocsHref('/sdk/configuration'), '/sdk/configuration');
  assert.equal(fumadocsHref('/'), '/');
  const source = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.match(source, /source\.getPageByHref\(fumadocsHref\(href\)\)/);
});
