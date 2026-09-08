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
