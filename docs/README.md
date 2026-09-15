# Fumadocs documentation

This directory contains the Next.js/Fumadocs static site. It uses repository-owned
fonts and assets, browser-only search, and no prompt submission endpoint or
analytics service.

## Develop

From `docs/`:

```bash
npm ci
npm run dev
```

The committed `package-lock.json` is the dependency source for repeatable builds.
Use `npm install` only when intentionally changing dependencies, then review the
lockfile diff.

## Validate current content

```bash
npm run generate
npm run check:content
npm test
npm run typecheck
npm run build
```

`generate` renders every registered page to Markdown and writes search, agent,
manifest, sitemap, robots, Cloudflare redirect, and Markdown MIME surfaces.
`check:content` validates metadata and local links. Tests reject stale generated
files, duplicate identities/routes, hash drift, missing twins, and orphan
outputs. Typecheck and build invoke Fumadocs generation.

A production static export is emitted to `out/`. Validate a repository-path Pages
deployment with:

```bash
NEXT_PUBLIC_BASE_PATH=/pllm npm run build
```

`NEXT_PUBLIC_BASE_PATH` prefixes downloads, browser search, repository-owned raw
HTML links, and framework routes. Pages reads the configured path at build time;
no owner or repository URL is hardcoded.

## Content ownership

- `content/home.html` is the current product landing content.
- `content/docs/` is operational MDX; `content/research/` supplies the independent
  `/research/*` publication tree.
- `components/site-chrome.tsx` owns shared search, theme, header, and footer UI.
- `navigation.json` owns maintained product navigation labels and canonical routes;
  `lib/navigation.ts` consumes them for shared UI.
- `public/brand.css` is the shared visual language.
- `content/research.html` owns the research landing page. Paper downloads,
  Markdown exports, search indexes, and `out/` are generated surfaces.
- `lib/docs-routes.mjs` defines canonical documentation routes from source slugs.
  Canonical HTML routes use static-export directory slashes but never an `index`
  leaf. Each Markdown twin removes that slash and adds `.md`; `/` maps to
  `/index.md`. Pre-launch docs have no compatibility aliases. Only slash
  normalization redirects are emitted. `public/releases/<release>/` holds
  immutable machine indexes.

Paper and evidence assets have a separate canonical source and build. Regenerate
them only for an intentional research/release change, not as a side effect of a
product docs edit. `scripts/generate.mjs` creates static search and Markdown
exports during a full site build.

Neither a local framework build nor configured GitHub workflow proves remote
deployment, model quality, protocol security, or benchmark performance.
