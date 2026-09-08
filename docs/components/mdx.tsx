import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import Link from 'next/link';
import { withBasePath } from '@/lib/paths.mjs';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    a: ({ href = '', children, ...props }) => {
      if (href.startsWith('/docs') || href === '/' || href.startsWith('/research')) {
        return <Link href={href} {...props}>{children}</Link>;
      }
      return <a href={withBasePath(href)} {...props}>{children}</a>;
    },
    ...components,
  };
}
