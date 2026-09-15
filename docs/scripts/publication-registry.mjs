import { discoveredDocs } from './discovered-docs.mjs';
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
    markdownAliases: [],
    ...metadata,
  };
};

const cliCommands = [
  ['config'],
  ['components'],
  ['research'],
  ['benchmark'],
  ['dev'],
  ['config', 'show'],
  ['config', 'export'],
  ['components', 'list'],
  ['components', 'show'],
  ['research', 'sources'],
  ['research', 'methods'],
  ['research', 'recipes'],
  ['research', 'assess'],
  ['research', 'agents'],
  ['benchmark', 'run'],
  ['dev', 'dashboard'],
  ['research', 'sources', 'list'],
  ['research', 'sources', 'show'],
  ['research', 'methods', 'list'],
  ['research', 'methods', 'show'],
  ['research', 'recipes', 'list'],
  ['research', 'recipes', 'show'],
];
const cliParents = new Set(cliCommands.flatMap((words) =>
  words.slice(1).map((_word, index) => words.slice(0, index + 1).join('/'))));
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

const declaredPublicationRegistry = {
  schemaVersion: '1.0.0',
  release: '0.1.0',
  canonicalOrigin: 'https://pllm.run',
  pages: [
    page('pllm.home', 'content/home.html', '/', 'homepage'),

    page('pllm.docs.start', 'content/docs/start/index.mdx', '/start', 'guide'),
    page('pllm.docs.start.installation', 'content/docs/start/installation.mdx', '/start/installation', 'guide'),
    page('pllm.docs.start.first-private-request', 'content/docs/start/first-private-request.mdx', '/start/first-private-request', 'guide'),

    page('pllm.docs.sdk.configuration', 'content/docs/sdk/configuration.mdx', '/sdk/configuration', 'reference'),
    page('pllm.docs.sdk.plans', 'content/docs/sdk/plans.mdx', '/sdk/plans', 'reference'),
    page('pllm.docs.sdk.components', 'content/docs/sdk/components.mdx', '/sdk/components', 'component'),

    page('pllm.docs.build', 'content/docs/build/index.mdx', '/build', 'guide'),
    page('pllm.docs.build.models', 'content/docs/build/models.mdx', '/build/models', 'reference'),
    page('pllm.docs.build.research', 'content/docs/build/research.mdx', '/build/research', 'research'),

    page('pllm.docs.understand', 'content/docs/understand/index.mdx', '/understand', 'concept'),
    page('pllm.docs.understand.architecture', 'content/docs/understand/architecture.mdx', '/understand/architecture', 'concept'),
    page('pllm.docs.understand.trust-boundary', 'content/docs/understand/trust-boundary.mdx', '/understand/trust-boundary', 'concept'),
    page('pllm.docs.understand.privacy-assurance', 'content/docs/understand/privacy-assurance.mdx', '/understand/privacy-assurance', 'assurance'),
    page('pllm.docs.understand.evidence-claims', 'content/docs/understand/evidence-claims.mdx', '/understand/evidence-claims', 'evidence'),

    page('pllm.docs.measure', 'content/docs/measure/index.mdx', '/measure', 'benchmark'),
    page('pllm.docs.measure.benchmark', 'content/docs/measure/benchmark.mdx', '/measure/benchmark', 'benchmark'),
    page('pllm.docs.measure.compare', 'content/docs/measure/compare.mdx', '/measure/compare', 'benchmark'),
    page('pllm.docs.measure.reproduce', 'content/docs/measure/reproduce.mdx', '/measure/reproduce', 'research'),
    page('pllm.docs.measure.assure', 'content/docs/measure/assure.mdx', '/measure/assure', 'assurance'),

    page('pllm.docs.operate', 'content/docs/operate/index.mdx', '/operate', 'deployment'),
    page('pllm.docs.operate.client-boundary', 'content/docs/operate/client-boundary.mdx', '/operate/client-boundary', 'deployment'),
    page('pllm.docs.operate.provider-roles', 'content/docs/operate/provider-roles.mdx', '/operate/provider-roles', 'deployment'),
    page('pllm.docs.operate.deployment', 'content/docs/operate/deployment.mdx', '/operate/deployment', 'deployment'),

    page('pllm.docs.reference', 'content/docs/reference/index.mdx', '/reference', 'reference'),
    page('pllm.docs.reference.cli', 'content/docs/reference/cli/index.mdx', '/reference/cli', 'reference', cliProvenance('content/docs/reference/cli/index.mdx')),
    ...cliCommandPages,
    page('pllm.docs.reference.python.pllm', 'content/docs/reference/python/pllm/index.mdx', '/reference/python/pllm', 'reference', { publicModules: ['pllm', 'pllm.config', 'pllm.models', 'pllm.plan', 'pllm.components', 'pllm.kernels', 'pllm.protocols', 'pllm.protocols.masked_linear', 'pllm.preparation', 'pllm.pipeline', 'pllm.deployment', 'pllm.runtime', 'pllm.research', 'pllm.compiler', 'pllm.evidence'], sourcePaths: ['content/docs/reference/python/pllm/index.mdx', '../python/pllm'], testPaths: ['../tests/test_developer_reference.py'] }),
    page('pllm.docs.reference.components', 'content/docs/reference/components.mdx', '/reference/components', 'reference', { publicModules: ['pllm.components'], publicSymbols: ['list_components', 'get_component'], componentIds: ['pllm/cpu', 'pllm/kv-cache-eviction', 'pllm/masked-linear', 'pllm/model-aware-corrections'], sourcePaths: ['content/docs/reference/components.mdx', '../python/pllm/components/__init__.py'], testPaths: ['../tests/test_developer_reference.py'] }),
    page('pllm.docs.reference.research', 'content/docs/reference/research.mdx', '/reference/research', 'reference', { publicModules: ['pllm.research'], publicSymbols: ['list_sources', 'list_methods', 'list_recipes', 'get_source', 'get_method', 'get_recipe'], sourcePaths: ['content/docs/reference/research.mdx', '../research/methods', '../research/recipes'], testPaths: ['../tests/test_research.py', '../tests/test_developer_reference.py'] }),
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
      ...page('pllm.research.whitepaper', 'content/whitepaper.html', '/research/whitepaper', 'whitepaper'),
      citationLinks: ['/learn/understand/architecture/', '/learn/understand/privacy-assurance/'],
      evidenceLinks: ['/downloads/current-runtime-2026-09-11.json'],
    },
  ],
};

function discoveredDocsPages() {
  const declaredSources = new Set(declaredPublicationRegistry.pages.map((item) => item.sourcePath));
  return discoveredDocs.flatMap((relative) => {
    const sourcePath = `content/docs/${relative}.mdx`;
    if (declaredSources.has(sourcePath)) return [];
    const segments = relative.split('/');
    if (segments.at(-1) === 'index') segments.pop();
    return [page(
      ['pllm', 'docs', ...segments].join('.'),
      sourcePath,
      '/',
      'guide',
    )];
  });
}

export const publicationRegistry = {
  ...declaredPublicationRegistry,
  pages: [...declaredPublicationRegistry.pages, ...discoveredDocsPages()],
};
