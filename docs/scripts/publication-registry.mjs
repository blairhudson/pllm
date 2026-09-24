import { discoveredDocs } from './discovered-docs.mjs';
import { packageVersion } from './package-version.mjs';
import { canonicalDocsUrl, markdownUrlForCanonical } from '../lib/docs-routes.mjs';

const page = (
  id,
  sourcePath,
  canonicalUrl,
  kind,
  metadata = {},
) => {
  const resolvedCanonical = sourcePath.startsWith('content/docs/')
    ? canonicalDocsUrl(sourcePath)
    : canonicalUrl === '/' ? '/' : `${canonicalUrl.replace(/\/$/, '')}/`;
  return {
    id,
    sourcePath,
    canonicalUrl: resolvedCanonical,
    markdownUrl: markdownUrlForCanonical(resolvedCanonical),
    kind,
    aliases: [],
    markdownAliases: (metadata.aliases ?? [])
      .filter((alias) => alias.endsWith('/'))
      .map((alias) => `${alias.slice(0, -1)}.md`),
    ...metadata,
  };
};

const legacyAliases = (route) => [route.replace(/\/$/, ''), `${route.replace(/\/$/, '')}/`];

const cliCommands = [
  ['config'],
  ['components'],
  ['benchmark'],
  ['dev'],
  ['config', 'show'],
  ['config', 'export'],
  ['components', 'list'],
  ['components', 'show'],
  ['benchmark', 'run'],
  ['benchmark', 'quality'],
  ['gateway'],
  ['serve'],
  ['serve', 'inference'],
  ['serve', 'preparation'],
  ['dev', 'dashboard'],
];
const cliParents = new Set(cliCommands.flatMap((words) =>
  words.slice(1).map((_word, index) => words.slice(0, index + 1).join('/'))));
cliParents.add('gateway');
const cliProvenance = (sourcePath) => ({
  sourcePaths: [sourcePath, '../python/pllm/_cli/app.py'],
  testPaths: ['../tests/test_cli.py', '../tests/test_developer_reference.py'],
});
const cliCommandPages = cliCommands.map((words) => {
  const commandPath = words.join('/');
  const sourcePath = `content/docs/reference/cli/${commandPath}${cliParents.has(commandPath) ? '/index' : ''}.mdx`;
  return page(
    `pllm.docs.reference.cli.${words.join('.')}`,
    sourcePath,
    '/',
    'reference',
    cliProvenance(sourcePath),
  );
});
const cliGuidePages = ['local-experiments', 'provider-connections'].map((slug) => {
  const sourcePath = `content/docs/reference/cli/gateway/${slug}.mdx`;
  return page(
    `pllm.docs.reference.cli.gateway.${slug}`,
    sourcePath,
    '/',
    'reference',
    cliProvenance(sourcePath),
  );
});

const pythonModules = [
  'pllm', 'pllm.client', 'pllm.compiler', 'pllm.components', 'pllm.config',
  'pllm.correlation', 'pllm.deployment', 'pllm.evidence', 'pllm.kernels',
  'pllm.metrics', 'pllm.models', 'pllm.native', 'pllm.nonlinear', 'pllm.official',
  'pllm.passes', 'pllm.pipeline', 'pllm.plan', 'pllm.preparation', 'pllm.profiles', 'pllm.quantization',
  'pllm.protocols', 'pllm.protocols.masked_linear', 'pllm.providers', 'pllm.research',
  'pllm.roles', 'pllm.runtime', 'pllm.schedulers', 'pllm.search', 'pllm.server',
  'pllm.sources', 'pllm.state', 'pllm.verification',
];
const pythonModulePages = pythonModules.slice(1).map((module) => {
  const slug = module.slice('pllm.'.length).replaceAll('.', '-').replaceAll('_', '-');
  const sourcePath = `content/docs/reference/python/pllm/${slug}.mdx`;
  return page(
    `pllm.docs.reference.python.${module.replaceAll('_', '-')}`,
    sourcePath,
    `/reference/python/pllm/${slug}`,
    'reference',
    {
      publicModules: [module],
      sourcePaths: [sourcePath, '../python/pllm'],
      testPaths: ['../tests/test_developer_reference.py'],
    },
  );
});

const declaredPublicationRegistry = {
  schemaVersion: '2.0.0',
  release: packageVersion,
  canonicalOrigin: 'https://pllm.run',
  pages: [
    page('pllm.home', 'content/home.html', '/', 'homepage'),

    page('pllm.docs.start', 'content/docs/learn/index.mdx', '/learn', 'guide', { aliases: legacyAliases('/learn/start') }),
    page('pllm.docs.start.installation', 'content/docs/learn/installation.mdx', '/learn/installation', 'guide', { aliases: legacyAliases('/learn/start/installation') }),
    page('pllm.docs.start.first-private-request', 'content/docs/learn/first-private-request.mdx', '/learn/first-private-request', 'guide', { aliases: legacyAliases('/learn/start/first-private-request') }),
    page('pllm.docs.start.first-local-benchmark', 'content/docs/learn/first-local-benchmark.mdx', '/learn/first-local-benchmark', 'guide', { aliases: legacyAliases('/learn/start/first-local-benchmark') }),
    page('pllm.docs.start.inspect-a-plan', 'content/docs/learn/inspect-a-plan.mdx', '/learn/inspect-a-plan', 'guide', { aliases: legacyAliases('/learn/start/inspect-a-plan') }),

    page('pllm.docs.learn.concepts', 'content/docs/learn/concepts/index.mdx', '/learn/concepts', 'concept'),
    page('pllm.docs.learn.concepts.privacy-and-threat-models', 'content/docs/learn/concepts/privacy-and-threat-models.mdx', '/learn/concepts/privacy-and-threat-models', 'concept', { aliases: legacyAliases('/learn/privacy-and-threat-models') }),
    page('pllm.docs.learn.concepts.parties-and-offline-work', 'content/docs/learn/concepts/parties-and-offline-work.mdx', '/learn/concepts/parties-and-offline-work', 'concept', { aliases: legacyAliases('/learn/parties-and-offline-work') }),
    page('pllm.docs.learn.concepts.masked-linear-inference', 'content/docs/learn/concepts/masked-linear-inference.mdx', '/learn/concepts/masked-linear-inference', 'concept', { aliases: legacyAliases('/learn/masked-linear-inference') }),
    page('pllm.docs.learn.concepts.garbling', 'content/docs/learn/concepts/garbling.mdx', '/learn/concepts/garbling', 'concept', { aliases: legacyAliases('/learn/arithmetic-and-boolean-garbling') }),
    page('pllm.docs.learn.concepts.numeric-semantics', 'content/docs/learn/concepts/numeric-semantics.mdx', '/learn/concepts/numeric-semantics', 'concept', { aliases: legacyAliases('/learn/numeric-semantics-and-model-quality') }),
    page('pllm.docs.learn.concepts.research-and-evidence', 'content/docs/learn/concepts/research-and-evidence.mdx', '/learn/concepts/research-and-evidence', 'concept', { aliases: legacyAliases('/learn/reading-research-and-evidence') }),

    page('pllm.docs.sdk', 'content/docs/sdk/index.mdx', '/sdk', 'guide', { aliases: legacyAliases('/sdk/build') }),
    page('pllm.docs.sdk.experiments.files', 'content/docs/sdk/experiments/files.mdx', '/sdk/experiments/files', 'guide', { aliases: legacyAliases('/sdk/configuration') }),
    page('pllm.docs.sdk.plans', 'content/docs/sdk/plans/index.mdx', '/sdk/plans', 'reference'),
    page('pllm.docs.sdk.components', 'content/docs/sdk/components/index.mdx', '/sdk/components', 'component'),

    page('pllm.docs.build.models', 'content/docs/sdk/extend/model-adapter.mdx', '/sdk/extend/model-adapter', 'guide', { aliases: legacyAliases('/sdk/build/model-adapters') }),
    page('pllm.docs.build.research', 'content/docs/recipes/build-method.mdx', '/research/recipes/build-method', 'research'),

    page('pllm.docs.understand', 'content/docs/learn/concepts/architecture-and-trust.mdx', '/learn/concepts/architecture-and-trust', 'concept', { aliases: legacyAliases('/learn/understand') }),
    page('pllm.docs.understand.architecture', 'content/docs/learn/concepts/architecture.mdx', '/learn/concepts/architecture', 'concept', { aliases: legacyAliases('/learn/understand/architecture') }),
    page('pllm.docs.understand.trust-boundary', 'content/docs/learn/concepts/trust-boundary.mdx', '/learn/concepts/trust-boundary', 'concept', { aliases: legacyAliases('/learn/understand/trust-boundary') }),
    page('pllm.docs.understand.privacy-assurance', 'content/docs/learn/concepts/privacy-assurance.mdx', '/learn/concepts/privacy-assurance', 'assurance', { aliases: legacyAliases('/learn/understand/privacy-assurance') }),
    page('pllm.docs.understand.evidence-claims', 'content/docs/learn/concepts/evidence-claims.mdx', '/learn/concepts/evidence-claims', 'evidence', { aliases: legacyAliases('/learn/understand/evidence-claims') }),

    page('pllm.docs.evaluate', 'content/docs/sdk/evaluate/index.mdx', '/sdk/evaluate', 'benchmark', { aliases: legacyAliases('/sdk/research') }),
    page('pllm.docs.measure.benchmark', 'content/docs/sdk/evaluate/benchmark.mdx', '/sdk/evaluate/benchmark', 'benchmark', { aliases: legacyAliases('/sdk/research/benchmark') }),
    page('pllm.docs.measure.compare', 'content/docs/recipes/compare.mdx', '/research/recipes/compare', 'benchmark'),
    page('pllm.docs.measure.reproduce', 'content/docs/recipes/reproduce.mdx', '/research/recipes/reproduce', 'research'),
    page('pllm.docs.measure.assure', 'content/docs/sdk/evaluate/assurance.mdx', '/sdk/evaluate/assurance', 'assurance', { aliases: [...legacyAliases('/sdk/research/assure'), ...legacyAliases('/sdk/research/assurance')] }),

    page('pllm.docs.operate', 'content/docs/sdk/run/index.mdx', '/sdk/run', 'deployment', { aliases: legacyAliases('/sdk/operate') }),
    page('pllm.docs.operate.client', 'content/docs/sdk/run/clients.mdx', '/sdk/run/clients', 'deployment', { aliases: legacyAliases('/sdk/operate/client') }),
    page('pllm.docs.operate.client-boundary', 'content/docs/sdk/run/embedded-gateway.mdx', '/sdk/run/embedded-gateway', 'deployment', { aliases: legacyAliases('/sdk/operate/client-boundary') }),
    page('pllm.docs.operate.provider-roles', 'content/docs/sdk/run/provider-roles.mdx', '/sdk/run/provider-roles', 'deployment', { aliases: legacyAliases('/sdk/operate/provider-roles') }),
    page('pllm.docs.operate.deployment', 'content/docs/sdk/run/lifecycle.mdx', '/sdk/run/lifecycle', 'deployment', { aliases: legacyAliases('/sdk/operate/deployment/status') }),
    page('pllm.docs.sdk.run.deployment', 'content/docs/sdk/run/deployment.mdx', '/sdk/run/deployment', 'deployment', { aliases: legacyAliases('/sdk/operate/deployment') }),

    page('pllm.docs.reference', 'content/docs/reference/index.mdx', '/reference', 'reference'),
    page('pllm.docs.reference.cli', 'content/docs/reference/cli/index.mdx', '/reference/cli', 'reference', cliProvenance('content/docs/reference/cli/index.mdx')),
    ...cliCommandPages,
    ...cliGuidePages,
    page('pllm.docs.reference.python.pllm', 'content/docs/reference/python/pllm/index.mdx', '/reference/python/pllm', 'reference', { publicModules: pythonModules, sourcePaths: ['content/docs/reference/python/pllm/index.mdx', '../python/pllm'], testPaths: ['../tests/test_developer_reference.py'] }),
    ...pythonModulePages,
    page('pllm.docs.reference.components', 'content/docs/reference/components.mdx', '/reference/components', 'reference', { publicModules: ['pllm.components', 'pllm.correlation', 'pllm.kernels', 'pllm.metrics', 'pllm.nonlinear', 'pllm.passes', 'pllm.preparation', 'pllm.protocols', 'pllm.roles', 'pllm.schedulers', 'pllm.state', 'pllm.verification'], publicSymbols: ['get', 'get_component', 'list_component_classes', 'list_components'], componentIds: ['pllm/accuracy', 'pllm/bfv-correlations/v1', 'pllm/binary-table/v1', 'pllm/blinded-linear/v1', 'pllm/chunked-independent-lanes/v1', 'pllm/cleartext-linear', 'pllm/client-local-kv', 'pllm/communication', 'pllm/cost', 'pllm/cpu', 'pllm/direct-fhe', 'pllm/energy', 'pllm/guarded-linear/v1', 'pllm/he-authenticated-preprocessing', 'pllm/independent-lanes/v1', 'pllm/inference', 'pllm/kv-cache-eviction', 'pllm/latency', 'pllm/linear-integrity', 'pllm/masked-linear', 'pllm/memory', 'pllm/model-aware-corrections', 'pllm/perplexity', 'pllm/r03-crt/v1', 'pllm/scalar/v1', 'pllm/secure-linear/v1', 'pllm/seeded-expansion', 'pllm/throughput'], sourcePaths: ['content/docs/reference/components.mdx', '../python/pllm/components/__init__.py'], testPaths: ['../tests/test_component_families.py', '../tests/test_developer_reference.py'] }),
    page('pllm.docs.reference.schemas', 'content/docs/reference/schemas/index.mdx', '/reference/schemas', 'reference'),
    page('pllm.docs.reference.status', 'content/docs/reference/status.mdx', '/reference/status', 'reference'),

    page('pllm.docs.research.sources', 'content/docs/research/sources.mdx', '/research/sources', 'research'),
    page('pllm.docs.research.methods', 'content/docs/research/methods.mdx', '/research/methods', 'research'),
    page('pllm.docs.research.compositions', 'content/docs/research/compositions.mdx', '/research/compositions', 'research'),
    page('pllm.docs.research.evidence', 'content/docs/research/evidence.mdx', '/research/evidence', 'research'),
    page('pllm.docs.research.publications', 'content/docs/research/publications.mdx', '/research/publications', 'research'),
    page('pllm.docs.research.clean-room', 'content/docs/research/clean-room.mdx', '/research/clean-room', 'research'),

    page('pllm.research', 'content/research.html', '/research', 'research-landing'),
    page('pllm.research.paper', 'content/research/paper.mdx', '/research/paper', 'research-paper'),
    {
      ...page('pllm.research.whitepaper', 'content/research/whitepaper.mdx', '/research/whitepaper', 'whitepaper'),
      citationLinks: ['/learn/concepts/architecture/', '/learn/concepts/privacy-assurance/'],
      evidenceLinks: ['/downloads/current-runtime-2026-09-11.json'],
    },
  ],
};

const legacySdkRoutes = new Map([
  ['sdk/inference/index', ['/sdk/pipeline']],
  ['sdk/inference/prepared-protocol', ['/sdk/pipeline/protocols/masked-linear']],
  ['sdk/components/protocols/index', ['/sdk/pipeline/protocols']],
  ['sdk/components/preparation/index', ['/sdk/pipeline/preparation']],
  ['sdk/components/kernels/native-matrix', ['/sdk/pipeline/kernels']],
  ['sdk/plans/compiler-internals', ['/sdk/pipeline/compiler']],
  ['sdk/run/runtime-internals', ['/sdk/pipeline/runtime']],
  ['sdk/plans/operators/index', ['/sdk/build/operators']],
  ['sdk/plans/numerics', ['/sdk/build/numerics']],
  ['sdk/plans/representations', ['/sdk/build/representations']],
  ['sdk/plans/conversions', ['/sdk/build/conversions']],
  ['sdk/evaluate/evidence', ['/sdk/research/benchmarks']],
  ['sdk/evaluate/search', ['/sdk/research/search', '/sdk/build/search']],
  ['sdk/extend/index', ['/sdk/contribute']],
  ['sdk/extend/component', ['/sdk/contribute/component-standard']],
  ['sdk/extend/provider', ['/sdk/contribute/publish-a-provider']],
  ['sdk/extend/add-a-category', ['/sdk/contribute/add-a-category']],
  ['sdk/extend/native-plugin', ['/sdk/contribute/native-plugin-abi']],
  ['sdk/extend/upstreaming', ['/sdk/contribute/upstreaming']],
  ['sdk/extend/agents/index', ['/sdk/contribute/agents']],
  ['sdk/extend/agents/implementation-map', ['/sdk/contribute/agents/implementation-map']],
]);

function discoveredDocsPages() {
  const declaredSources = new Set(declaredPublicationRegistry.pages.map((item) => item.sourcePath));
  return discoveredDocs.flatMap((relative) => {
    const sourcePath = `content/docs/${relative}.mdx`;
    if (declaredSources.has(sourcePath)) return [];
    const segments = relative.split('/');
    if (segments.at(-1) === 'index') segments.pop();
    const oldFamilySlugs = {
      'qwen3-5': 'qwen35',
      'phi-4-mini': 'phi4-mini',
      'gemma-4': 'gemma4',
    };
    const familySlug = relative.startsWith('sdk/models/families/')
      ? relative.slice('sdk/models/families/'.length)
      : null;
    const oldFamilyRoute = familySlug === null ? null
      : `/sdk/build/models${familySlug === 'index' ? '' : `/${oldFamilySlugs[familySlug] ?? familySlug}`}`;
    const oldGarblingRoute = relative.startsWith('sdk/components/nonlinear/garbling/')
      ? `/sdk/pipeline/protocols/garbling${relative.endsWith('/index') ? '' : `/${relative.split('/').at(-1)}`}`
      : null;
    const oldOperatorRoute = relative.startsWith('sdk/plans/operators/') && !relative.endsWith('/index')
      ? `/sdk/build/operators/${relative.split('/').at(-1)}` : null;
    const previousRoutes = [oldFamilyRoute, oldGarblingRoute, oldOperatorRoute,
      ...(legacySdkRoutes.get(relative) ?? [])].filter((route) => route !== null);
    return [page(
      ['pllm', 'docs', ...segments].join('.'),
      sourcePath,
      '/',
      'guide',
      previousRoutes.length === 0 ? {} : { aliases: previousRoutes.flatMap(legacyAliases) },
    )];
  });
}

export const publicationRegistry = {
  ...declaredPublicationRegistry,
  pages: [...declaredPublicationRegistry.pages, ...discoveredDocsPages()],
};
