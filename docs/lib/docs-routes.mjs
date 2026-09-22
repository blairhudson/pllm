const EXACT_ROUTES = new Map([
  ['agents/reproduction-checklist', 'research/recipes/reproduction-checklist'],
  ['agents/research-map', 'research/recipes/agent-map'],
  ['build/models', 'sdk/build/model-adapters'],
  ['build/research', 'research/recipes/build-method'],
  ['measure/compare', 'research/recipes/compare'],
  ['measure/reproduce', 'research/recipes/reproduce'],
  ['operate/deployment', 'sdk/operate/deployment/status'],
  ['reference/cli', 'cli/reference'],
]);

const PREFIX_ROUTES = [
  ['reference/cli', 'cli/reference'],
  ['research/experiments', 'research/recipes/experiments'],
  ['research/reproductions', 'research/recipes/reproductions'],
  ['reference', 'sdk/reference'],
  ['deployment', 'sdk/operate/deployment'],
  ['contribute', 'sdk/contribute'],
  ['preparation', 'sdk/pipeline/preparation'],
  ['representations', 'sdk/build/representations'],
  ['conversions', 'sdk/build/conversions'],
  ['benchmarks', 'sdk/research/benchmarks'],
  ['assurance', 'sdk/research/assurance'],
  ['protocols', 'sdk/pipeline/protocols'],
  ['compiler', 'sdk/pipeline/compiler'],
  ['operators', 'sdk/build/operators'],
  ['numerics', 'sdk/build/numerics'],
  ['runtime', 'sdk/pipeline/runtime'],
  ['kernels', 'sdk/pipeline/kernels'],
  ['models', 'sdk/build/models'],
  ['search', 'sdk/build/search'],
  ['pipeline', 'sdk/pipeline'],
  ['measure', 'sdk/research'],
  ['operate', 'sdk/operate'],
  ['agents', 'sdk/contribute/agents'],
  ['build', 'sdk/build'],
  ['research', 'research'],
  ['recipes', 'research/recipes'],
  ['metrics', 'research/records/metrics'],
  ['start', 'learn/start'],
  ['understand', 'learn/understand'],
  ['examples', 'learn/examples'],
  ['learn', 'learn'],
  ['cli', 'cli'],
  ['sdk', 'sdk'],
];

export function docsHrefForSlugs(input) {
  const slugs = [...input];
  if (slugs.at(-1) === 'index') slugs.pop();
  const sourceRoute = slugs.join('/');
  const exact = EXACT_ROUTES.get(sourceRoute);
  if (exact) return `/${exact}`;

  for (const [sourcePrefix, publicPrefix] of PREFIX_ROUTES) {
    if (sourceRoute === sourcePrefix) return `/${publicPrefix}`;
    if (sourceRoute.startsWith(`${sourcePrefix}/`)) {
      return `/${publicPrefix}${sourceRoute.slice(sourcePrefix.length)}`;
    }
  }

  throw new Error(`No public documentation route for source slug: ${sourceRoute}`);
}

export function docsHrefForSourcePath(sourcePath) {
  const match = /^content\/docs\/(.+)\.mdx$/.exec(sourcePath);
  if (!match) throw new Error(`Not a documentation source path: ${sourcePath}`);
  return docsHrefForSlugs(match[1].split('/'));
}

export function canonicalDocsUrl(sourcePath) {
  return `${docsHrefForSourcePath(sourcePath)}/`;
}

export function markdownUrlForCanonical(canonicalUrl) {
  if (canonicalUrl === '/') return '/index.md';
  return `${canonicalUrl.replace(/\/$/, '')}.md`;
}

export function fumadocsHref(canonicalUrl) {
  if (canonicalUrl === '/') return '/';
  return canonicalUrl.replace(/\/+$/, '');
}
