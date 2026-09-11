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
npm run check:content
npm test
npm run typecheck
npm run build
```

`check:content` validates MDX metadata and local links. Tests validate routes,
required user journeys, static export configuration, and privacy-sensitive
examples. Typecheck and build invoke Fumadocs generation; use them when generated
build state is allowed. Operational content-only work can run the first two
commands without rewriting checked public exports.

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
- `content/docs/` is operational MDX; `content/docs/research/` is historical
  research and must retain its recorded scope.
- `components/site-chrome.tsx` owns shared search, theme, header, and footer UI.
- `lib/navigation.ts` owns maintained product navigation labels and routes.
- `public/brand.css` is the shared visual language.
- `content/research.html`, paper downloads, Markdown exports, search indexes, and
  `out/` are generated or historical surfaces, not places to hand-edit current
  runtime claims.

Paper and evidence assets have a separate canonical source and build. Regenerate
them only for an intentional research/release change, not as a side effect of a
product docs edit. `scripts/generate.mjs` creates static search and Markdown
exports during a full site build.

Neither a local framework build nor configured GitHub workflow proves remote
deployment, model quality, protocol security, or benchmark performance.
