import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import navigation from '../navigation.json' with { type: 'json' };
import { readPages, readSearchPages, readStandalonePages, validate, siteRoot, walk } from '../scripts/content.mjs';

const pages = readPages();
const standalone = readStandalonePages();
const byRoute = new Map(readSearchPages().map((page) => [page.canonicalUrl, page]));

test('canonical domain hierarchy and research journeys exist', () => {
  for (const route of [
    '/learn/', '/learn/start/', '/learn/start/installation/', '/learn/start/first-private-request/', '/learn/start/first-local-benchmark/',
    '/learn/understand/', '/learn/understand/architecture/', '/learn/understand/trust-boundary/', '/learn/understand/privacy-assurance/', '/learn/understand/evidence-claims/',
    '/learn/integrations/', '/learn/integrations/local-gateway/', '/learn/integrations/responses-api/', '/learn/integrations/chat-completions/',
    '/learn/integrations/openai-python/', '/learn/integrations/openai-agents/', '/learn/integrations/codex/', '/learn/integrations/opencode/',
    '/cli/', '/cli/reference/', '/cli/reference/config/', '/cli/reference/config/show/',
    '/cli/reference/benchmark/run/', '/cli/reference/components/list/',
    '/sdk/', '/sdk/configuration/', '/sdk/plans/', '/sdk/components/',
    '/sdk/build/', '/sdk/build/model-adapters/', '/sdk/build/models/', '/sdk/build/operators/', '/sdk/build/numerics/', '/sdk/build/representations/', '/sdk/build/conversions/', '/sdk/build/search/',
    '/sdk/pipeline/', '/sdk/pipeline/protocols/', '/sdk/pipeline/protocols/masked-linear/', '/sdk/pipeline/protocols/garbling/', '/sdk/pipeline/protocols/garbling/arithmetic/',
    '/sdk/pipeline/protocols/garbling/half-gates/', '/sdk/pipeline/protocols/garbling/lookup-tables/', '/sdk/pipeline/protocols/garbling/weighted-path/',
    '/sdk/pipeline/preparation/', '/sdk/pipeline/kernels/', '/sdk/pipeline/compiler/', '/sdk/pipeline/runtime/',
    '/sdk/research/', '/sdk/research/benchmark/', '/sdk/research/search/', '/sdk/research/assure/', '/sdk/research/benchmarks/', '/sdk/research/assurance/',
    '/sdk/operate/', '/sdk/operate/client/', '/sdk/operate/client-boundary/', '/sdk/operate/provider-roles/', '/sdk/operate/deployment/', '/sdk/operate/deployment/status/',
    '/sdk/reference/', '/sdk/reference/python/pllm/', '/sdk/reference/native/', '/sdk/reference/schemas/', '/sdk/reference/components/', '/sdk/reference/status/',
    '/sdk/contribute/', '/sdk/contribute/agents/',
    '/research/', '/research/papers/', '/research/records/', '/research/records/metrics/',
    '/research/recipes/', '/research/recipes/build-method/', '/research/recipes/compare/', '/research/recipes/reproduce/', '/research/recipes/experiments/', '/research/recipes/reproductions/', '/research/recipes/agent-map/', '/research/recipes/reproduction-checklist/',
    '/research/sources/', '/research/methods/', '/research/compositions/', '/research/evidence/', '/research/publications/', '/research/clean-room/',
  ]) assert.ok(byRoute.has(route), route);
});

test('research papers expose one canonical dependency-ordered implementation plan', () => {
  const dataRoot = path.join(siteRoot, 'data/research');
  const registry = JSON.parse(fs.readFileSync(path.join(dataRoot, 'papers.json'), 'utf8'));
  const plan = JSON.parse(fs.readFileSync(path.join(dataRoot, 'reimplementation-plan.json'), 'utf8'));
  const index = fs.readFileSync(path.join(siteRoot, 'content/docs/research/papers/index.mdx'), 'utf8');
  const plannedIds = plan.phases.flatMap((phase) => phase.papers);

  assert.equal(plan.schema_version, 'pllm.reimplementation_plan.v1');
  assert.deepEqual(plan.phases.map((phase) => phase.priority),
    Array.from({ length: plan.phases.length }, (_, index) => index + 1));
  assert.deepEqual([...plannedIds].sort(), registry.papers.map((paper) => paper.id).sort());
  assert.equal(new Set(plannedIds).size, registry.papers.length);
  assert.match(index, /## Reimplementation plan/);
  for (const phase of plan.phases) {
    assert.ok(index.includes(`### ${phase.priority}. ${phase.name}`), phase.name);
    assert.ok(index.includes(`**Status:** \`${phase.status}\``), phase.name);
    for (const module of phase.module_homes) assert.ok(index.includes(`\`${module}\``), module);
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

test('pipeline guides explain each public component in linked subsections', () => {
  const guides = new Map([
    ['/sdk/pipeline/', [
      'MaskedLinearCpu', 'VerifiedMaskedLinearCpu', 'DirectFHEProfile',
      'ProprietaryGuarded', 'ProprietaryBlinded',
    ]],
    ['/sdk/pipeline/compiler/', [
      'KvCacheEviction', 'BinaryTableGatedMultiplyQ7',
      'R03CrtGatedMultiplyQ7', 'ScalarProtectedTensorSchedule',
    ]],
    ['/sdk/pipeline/runtime/', [
      'OpenAI', 'build_roles', 'Inference', 'ClientLocalKv', 'FreivaldsVerify',
    ]],
    ['/sdk/pipeline/protocols/', [
      'MaskedLinear', 'DirectFHE', 'GuardedLinear', 'BlindedLinear',
      'SecureLinear', 'CleartextLinear',
    ]],
    ['/sdk/pipeline/protocols/masked-linear/', [
      'MaskedLinear', 'ModelAwareCorrections', 'Inference', 'Cpu', 'FreivaldsVerify',
    ]],
    ['/sdk/pipeline/preparation/', [
      'ModelAwareCorrections', 'BFVCorrelations', 'HEAuthenticatedPreprocessing',
      'TrustedPreprocessor',
    ]],
    ['/sdk/pipeline/kernels/', ['Cpu', 'MaskedGEMM', 'CompiledMatrix', 'capabilities']],
    ['/sdk/pipeline/protocols/garbling/', [
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
    ['/sdk/pipeline/', [
      'MaskedLinearCpu', 'VerifiedMaskedLinearCpu', 'DirectFHEProfile',
      'ProprietaryGuarded', 'ProprietaryBlinded',
    ]],
    ['/sdk/pipeline/protocols/', [
      'MaskedLinear', 'DirectFHE', 'GuardedLinear', 'BlindedLinear',
    ]],
    ['/sdk/pipeline/protocols/masked-linear/', [
      'MaskedLinear', 'ModelAwareCorrections', 'Inference', 'Cpu', 'FreivaldsVerify',
    ]],
    ['/sdk/pipeline/preparation/', ['ModelAwareCorrections']],
    ['/sdk/pipeline/kernels/', ['Cpu']],
    ['/sdk/pipeline/runtime/', ['build_roles', 'Inference', 'FreivaldsVerify']],
  ]);

  for (const [route, components] of experimentComponents) {
    const content = byRoute.get(route)?.content ?? '';
    assert.ok(
      content.includes('pllm gateway --local') || route === '/sdk/pipeline/runtime/',
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

  const runtime = byRoute.get('/sdk/pipeline/runtime/').content;
  const openAIStart = runtime.indexOf('### Send a synchronous streaming request with `OpenAI`');
  const openAIEnd = runtime.indexOf('\n### ', openAIStart + 5);
  assert.match(runtime.slice(openAIStart, openAIEnd), /responses\.create\(/);
});

test('primary reader journeys cross areas at the decision point', () => {
  const journeys = new Map([
    ['start/first-local-benchmark.mdx', [
      '/cli/reference/benchmark/run/',
      '/research/evidence/',
    ]],
    ['start/first-private-request.mdx', [
      '/sdk/configuration/',
      '/sdk/pipeline/protocols/',
      '/learn/integrations/',
    ]],
    ['start/index.mdx', ['/learn/integrations/']],
    ['cli/index.mdx', ['/learn/integrations/local-gateway/']],
    ['sdk/index.mdx', ['/learn/integrations/openai-python/']],
    ['operate/deployment.mdx', ['/learn/integrations/local-gateway/']],
    ['learn/privacy-and-threat-models.mdx', ['/research/evidence/']],
    ['sdk/configuration.mdx', [
      '/cli/reference/config/show/',
      '/cli/reference/config/export/',
    ]],
    ['measure/benchmark.mdx', [
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
      '/sdk/research/benchmarks/',
      '/sdk/research/assurance/',
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
  assert.match(home, /pllm gateway --local --model Qwen\/Qwen2\.5-0\.5B-Instruct/);
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
    ['pllm', '/sdk/operate/client-boundary/'],
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
  assert.deepEqual(learn.pages.slice(-2), ['../start', '../understand']);
  assert.deepEqual(cli.pages.slice(-1), ['../reference/cli']);
  assert.deepEqual(sdk.pages.slice(-6), [
    '../build', '../pipeline', '../measure', '../operate', '../reference', '../contribute',
  ]);
  assert.deepEqual(research.pages.slice(-3), ['../recipes', 'papers', 'records']);

  for (const branch of [
    'start', 'understand', 'learn/integrations', 'reference/cli', 'build', 'pipeline', 'measure',
    'operate', 'reference', 'contribute', 'recipes', 'research/papers',
    'research/records',
  ]) {
    const meta = readMeta(branch);
    assert.equal(meta.root, true, branch);
    assert.equal(meta.pages[0], 'index', branch);
    assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs', branch, 'index.mdx')), branch);
  }

  const docsLayout = fs.readFileSync(path.join(siteRoot, 'components/docs-shell.tsx'), 'utf8');
  const source = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.ok(docsLayout.includes('tabs={false}'));
  assert.ok(docsLayout.includes('<DocsSectionSwitcher'));
  assert.ok(source.includes('sectionRoots'));
  assert.ok(source.includes('getAreaSections'));
  assert.ok(source.includes('{ ...root, children: localChildren }'));
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
  assert.ok(landing.includes('PLLM is a Rust-first, Python-friendly runtime'));
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
  assert.equal(owner('/learn/start/installation'), 'Learn');
  assert.equal(owner('/cli/reference'), 'CLI');
  assert.equal(owner('/sdk/build/models/qwen2'), 'SDK');
  assert.equal(owner('/sdk/research/benchmark'), 'SDK');
  assert.equal(owner('/research/recipes/compare'), 'Research');
  assert.equal(owner('/research/recipes/build-method'), 'Research');
  assert.equal(owner('/research/recipes/agent-map'), 'Research');
  assert.equal(owner('/research/whitepaper'), 'Research');
  assert.equal(owner(''), undefined);
});

test('mobile documentation navigation is bounded, scrollable, and state-aware', () => {
  const header = fs.readFileSync(path.join(siteRoot, 'components/docs-mobile-header.tsx'), 'utf8');
  const css = fs.readFileSync(path.join(siteRoot, 'app/global.css'), 'utf8');
  const source = fs.readFileSync(path.join(siteRoot, 'lib/source.ts'), 'utf8');
  assert.ok(header.includes('aria-controls="nd-sidebar-mobile"'));
  assert.ok(header.includes('aria-expanded={open}'));
  assert.ok(header.includes("event.key === 'Escape'"));
  assert.match(css, /\.docs-area-switcher nav\s*{[^}]*display:\s*none/s);
  assert.match(css, /\.docs-area-switcher\[open\] nav\s*{\s*display:\s*grid/s);
  assert.match(css, /\.docs-section-switcher-list\s*{[^}]*overflow-y:\s*auto/s);
  assert.match(css, /#nd-sidebar-mobile\[data-state='open'\]/);
  assert.ok(source.includes("area === 'Research' ? [researchPapers()] : []"));
});

test('authored navigation is tracked and generated navigation is rebuilt', () => {
  const ignore = fs.readFileSync(path.join(siteRoot, '../.gitignore'), 'utf8');
  assert.ok(ignore.includes('!docs/content/docs/models/**'));
  assert.ok(ignore.includes('docs/content/docs/reference/python/pllm/'));
  assert.ok(ignore.includes('papers/'));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/models/meta.json')));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/research/papers/meta.json')));
});

test('authored copy uses direct technical English and standard status labels', () => {
  const generated = new Set([
    'content/docs/reference/components.mdx',
    'content/docs/reference/python/pllm/index.mdx',
  ]);
  for (const page of readSearchPages()) {
    if (page.sourcePath.startsWith('content/docs/reference/cli/') || generated.has(page.sourcePath)) continue;
    assert.doesNotMatch(page.content, /\b(?:inert|non-normative|organisations|behaviour|catalogue|authorise|optimise|centre)\b/i, page.canonicalUrl);
  }
  const status = byRoute.get('/sdk/reference/status/').content;
  for (const label of ['Available', 'Experimental', 'Planned', 'Not supported', 'Not evaluated', 'Not applicable']) {
    assert.ok(status.includes(label), label);
  }
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
    ['/cli/private-inference/', ['/cli/reference/gateway/', '/learn/integrations/', '/sdk/operate/client-boundary/']],
    ['/cli/provider-roles/', ['/cli/reference/serve/inference/', '/cli/reference/serve/preparation/', '/sdk/operate/deployment/']],
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

test('public copy uses standard technical English and fixed support labels', () => {
  for (const page of readSearchPages()) {
    assert.doesNotMatch(
      page.content,
      /\b(?:inert|non-normative|organisations|behaviour|catalogue|authorise|optimise|centre)\b/i,
      page.canonicalUrl,
    );
  }

  const status = byRoute.get('/sdk/reference/status/').content;
  for (const label of ['Available', 'Experimental', 'Planned', 'Not supported', 'Not evaluated', 'Not applicable']) {
    assert.ok(status.includes(label), label);
  }
});

test('paper, whitepaper, evidence, and generated CLI help remain downloadable', () => {
  for (const file of ['paper.pdf', 'paper-source.zip', 'whitepaper.pdf', 'whitepaper-source.zip', 'current-runtime-2026-09-11.json', 'evidence.zip', 'cli-help.txt']) {
    assert.ok(fs.statSync(path.join(siteRoot, 'public/downloads', file)).size > 0, file);
  }
});

test('static export, local assets, and canonical metadata remain configured', () => {
  const nextConfig = fs.readFileSync(path.join(siteRoot, 'next.config.mjs'), 'utf8');
  assert.ok(nextConfig.includes("process.env.NODE_ENV !== 'development'"));
  assert.ok(nextConfig.includes("config.output = 'export'"));
  assert.ok(
    nextConfig.includes(
      "allowedDevOrigins: ['127.0.0.1', 'macbookblair.local']",
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
