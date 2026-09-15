import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import navigation from '../navigation.json' with { type: 'json' };
import { readPages, readSearchPages, readStandalonePages, validate, siteRoot } from '../scripts/content.mjs';

const pages = readPages();
const standalone = readStandalonePages();
const byRoute = new Map(readSearchPages().map((page) => [page.canonicalUrl, page]));

test('canonical domain hierarchy and research journeys exist', () => {
  for (const route of [
    '/learn/', '/learn/start/', '/learn/start/installation/', '/learn/start/first-private-request/', '/learn/start/first-local-benchmark/',
    '/learn/understand/', '/learn/understand/architecture/', '/learn/understand/trust-boundary/', '/learn/understand/privacy-assurance/', '/learn/understand/evidence-claims/',
    '/cli/', '/cli/reference/',
    '/sdk/', '/sdk/configuration/', '/sdk/plans/', '/sdk/components/',
    '/sdk/build/', '/sdk/build/model-adapters/', '/sdk/build/models/', '/sdk/build/operators/', '/sdk/build/numerics/', '/sdk/build/representations/', '/sdk/build/conversions/', '/sdk/build/search/',
    '/sdk/pipeline/', '/sdk/pipeline/protocols/', '/sdk/pipeline/protocols/masked-linear/', '/sdk/pipeline/protocols/garbling/', '/sdk/pipeline/protocols/garbling/arithmetic/',
    '/sdk/pipeline/protocols/garbling/half-gates/', '/sdk/pipeline/protocols/garbling/lookup-tables/', '/sdk/pipeline/protocols/garbling/weighted-path/',
    '/sdk/pipeline/preparation/', '/sdk/pipeline/kernels/', '/sdk/pipeline/compiler/', '/sdk/pipeline/runtime/',
    '/sdk/research/', '/sdk/research/benchmark/', '/sdk/research/assure/', '/sdk/research/benchmarks/', '/sdk/research/assurance/',
    '/sdk/operate/', '/sdk/operate/client-boundary/', '/sdk/operate/provider-roles/', '/sdk/operate/deployment/', '/sdk/operate/deployment/status/',
    '/sdk/reference/', '/sdk/reference/python/pllm/', '/sdk/reference/native/', '/sdk/reference/schemas/', '/sdk/reference/components/', '/sdk/reference/status/',
    '/sdk/contribute/', '/sdk/contribute/agents/',
    '/research/', '/research/papers/', '/research/records/', '/research/records/metrics/', '/research/records/method-catalog/',
    '/research/recipes/', '/research/recipes/build-method/', '/research/recipes/compare/', '/research/recipes/reproduce/', '/research/recipes/experiments/', '/research/recipes/reproductions/', '/research/recipes/agent-map/', '/research/recipes/reproduction-checklist/',
    '/research/sources/', '/research/methods/', '/research/compositions/', '/research/evidence/', '/research/publications/', '/research/clean-room/',
  ]) assert.ok(byRoute.has(route), route);
});

test('every SDK page has a syntax-valid Python SDK example', () => {
  const sdkPages = pages.filter((page) => page.canonicalUrl.startsWith('/sdk/'));
  assert.ok(sdkPages.length > 0);

  for (const page of sdkPages) {
    const examples = [...page.content.matchAll(/```python\n([\s\S]*?)```/g)].map((match) => match[1]);
    assert.ok(examples.length > 0, `${page.canonicalUrl} has no Python example`);
    assert.ok(
      examples.some((example) => /(?:from|import)\s+pllm\b/.test(example)),
      `${page.canonicalUrl} has no PLLM SDK example`,
    );

    for (const example of examples) {
      const parsed = spawnSync(
        'python3',
        ['-c', 'import ast, sys; ast.parse(sys.stdin.read())'],
        { encoding: 'utf8', input: example },
      );
      assert.equal(parsed.status, 0, `${page.canonicalUrl}: ${parsed.stderr}`);
    }
  }
});

test('homepage SDK tab demonstrates a public Python component API', () => {
  const home = fs.readFileSync(path.join(siteRoot, 'content/home.html'), 'utf8');
  assert.match(home, /data-panel="start-sdk"/);
  assert.match(home, /from pllm\.components import get_component/);
  assert.match(home, /get_component\("pllm\/masked-linear"\)/);
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
    'start', 'understand', 'reference/cli', 'build', 'pipeline', 'measure',
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

test('navigation-only content directories are not hidden by repository ignore rules', () => {
  const ignore = fs.readFileSync(path.join(siteRoot, '../.gitignore'), 'utf8');
  assert.ok(ignore.includes('!docs/content/docs/models/**'));
  assert.ok(ignore.includes('!docs/content/docs/research/papers/**'));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/models/meta.json')));
  assert.ok(fs.existsSync(path.join(siteRoot, 'content/docs/research/papers/meta.json')));
});

test('authored copy uses direct technical English and standard status labels', () => {
  const generated = new Set([
    'content/docs/reference/cli/index.mdx',
    'content/docs/reference/components.mdx',
    'content/docs/reference/research.mdx',
    'content/docs/reference/python/pllm/index.mdx',
  ]);
  for (const page of readSearchPages()) {
    if (generated.has(page.sourcePath)) continue;
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

test('all local links and fragments resolve', () => assert.deepEqual(validate().errors, []));

test('standalone publication pages remain in content graph', () => {
  assert.deepEqual(standalone.map((page) => page.canonicalUrl), ['/', '/research/', '/research/whitepaper/']);
  assert.deepEqual(readSearchPages().filter((page) => page.sourcePath.endsWith('.html')).map((page) => page.id), ['pllm.home', 'pllm.research', 'pllm.research.whitepaper']);
});

test('documented commands use only the current CLI and development dashboard', () => {
  const sources = readSearchPages().map((page) => page.content).join('\n');
  assert.ok(sources.includes('pllm config show examples/pllm.yaml'));
  assert.ok(sources.includes('pllm components list'));
  assert.ok(sources.includes('pllm research sources list'));
  assert.ok(sources.includes('pllm dev dashboard'));
  assert.doesNotMatch(sources, /pllm benchmark dashboard/);
  assert.doesNotMatch(sources, /<pre>[^]*?pllm (?:serve|preparation serve|configure|chat|run|model lower|plan (?:check|compile|show)|party serve|benchmark (?:run|search|compare)|assure run|init)\b[^]*?<\/pre>/);
  for (const block of sources.match(/```(?:bash|sh|shell|console|text)?\n[^]*?```/g) ?? []) {
    assert.doesNotMatch(block, /^pllm (?:serve|preparation serve|configure|chat|run|model lower|plan (?:check|compile|show)|party serve|benchmark (?:run|search|compare)|assure run|init)\b/m);
  }
});

test('generated references are marked and authored guides stay separate', () => {
  for (const name of ['cli/index', 'python/pllm/index', 'components', 'research']) {
    const source = fs.readFileSync(path.join(siteRoot, `content/docs/reference/${name}.mdx`), 'utf8');
    assert.ok(source.includes('Generated by `scripts/generate_developer_reference.py`; do not edit.'));
  }
  assert.ok(byRoute.get('/sdk/components/').content.includes('pllm components list'));
  assert.equal(byRoute.get('/research/clean-room/').title, 'Contribute a clean-room reimplementation');
});

test('catalog and method pages expose independent status axes', () => {
  const components = byRoute.get('/sdk/reference/components/').content;
  const research = byRoute.get('/research/records/method-catalog/').content;
  for (const heading of ['Identity/version', 'Lifecycle', 'Implementation maturity', 'Provenance', 'Input/output representation', 'Roles/topology', 'Privacy/assurance', 'Model/operator coverage', 'Evidence/cohort', 'Known limitations']) {
    assert.ok(components.toLowerCase().includes(heading.toLowerCase()), heading);
    assert.ok(research.toLowerCase().includes(heading.toLowerCase()), heading);
  }
  assert.ok(components.includes('not recorded'));
  assert.ok(research.includes('no applicable evidence'));
});

test('research publication workflow cites source and separates PLLM adaptation', () => {
  const methods = byRoute.get('/research/methods/').content;
  const publications = byRoute.get('/research/publications/').content;
  assert.ok(methods.includes('pllm.source.mpcache.arxiv-2501.06807v2'));
  assert.ok(methods.includes('https://arxiv.org/abs/2501.06807v2'));
  assert.ok(methods.includes('pllm.method.mpcache-structural-adaptation.v1'));
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
  for (const file of ['paper.pdf', 'paper-source.zip', 'whitepaper.pdf', 'whitepaper.tex', 'current-runtime-2026-09-11.json', 'evidence.zip', 'cli-help.txt']) {
    assert.ok(fs.statSync(path.join(siteRoot, 'public/downloads', file)).size > 0, file);
  }
});

test('static export, local assets, and canonical metadata remain configured', () => {
  assert.ok(fs.readFileSync(path.join(siteRoot, 'next.config.mjs'), 'utf8').includes("output: 'export'"));
  assert.ok(!fs.readFileSync(path.join(siteRoot, 'app/layout.tsx'), 'utf8').includes('next/font/google'));
  for (const file of ['app/page.tsx', 'app/research/page.tsx', 'app/research/whitepaper/page.tsx', 'app/research/[...slug]/page.tsx', 'app/_docs-page.tsx']) {
    assert.ok(fs.readFileSync(path.join(siteRoot, file), 'utf8').includes('canonical'), file);
  }
  assert.ok(fs.existsSync(path.join(siteRoot, 'app/(docs)/[...slug]/page.tsx')));
  assert.ok(!fs.existsSync(path.join(siteRoot, 'app/docs')));
});
