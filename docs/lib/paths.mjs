/** Normalize a Next.js base path without changing external URLs. */
export function normalizeBasePath(value = '') {
  const path = value.trim().replace(/\/+$/, '');
  if (!path || path === '/') return '';
  if (!/^\/[A-Za-z0-9._/-]+$/.test(path) || path.includes('..') || path.includes('//')) {
    throw new Error('NEXT_PUBLIC_BASE_PATH must be an absolute path such as /pllm');
  }
  return path;
}
export const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH ?? '');
/** @param {string} href */
export function withBasePath(href, prefix = basePath) {
  if (!href.startsWith('/') || href.startsWith('//') || !prefix) return href;
  if (href === prefix || href.startsWith(prefix + '/')) return href;
  return prefix + href;
}
/** Prefix repository-owned HTML links, not user-supplied content. */
export function prefixHtml(html, prefix = basePath) {
  return html.replace(/\b(href|src)="(\/[^"\n]*)"/g,
    (_match, attribute, href) => `${attribute}="${withBasePath(href, prefix)}"`);
}

function pageForRoute(route) {
  const normalized = route === '/' ? '/' : `${route.replace(/\/$/, '')}/`;
  return publicationRegistry.pages.find((candidate) =>
    candidate.canonicalUrl === normalized || candidate.aliases?.some((alias) =>
      (alias === '/' ? '/' : `${alias.replace(/\/$/, '')}/`) === normalized));
}

/** Stable Markdown alternate for each canonical public route. */
export function markdownPathForRoute(route) {
  const page = pageForRoute(route);
  if (!page) throw new Error(`Unregistered public route: ${route}`);
  return page.markdownUrl;
}

/** Repository-relative source path for a canonical public route, when known. */
export function sourcePathForRoute(route) {
  return pageForRoute(route)?.sourcePath;
}
import { publicationRegistry } from '../scripts/publication-registry.mjs';
