import { createMDX } from 'fumadocs-mdx/next';
import { normalizeBasePath } from './lib/paths.mjs';
const withMDX = createMDX();
const config = {
  trailingSlash: true,
  reactStrictMode: true,
  allowedDevOrigins: ['127.0.0.1'],
  basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH ?? ''),
  images: { unoptimized: true },
  poweredByHeader: false,
};

if (process.env.NODE_ENV !== 'development') config.output = 'export';

export default withMDX(config);
