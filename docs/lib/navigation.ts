export type NavigationItem = {
  href: string;
  label: string;
  activePrefix?: string;
};

export const mainNavigation: readonly NavigationItem[] = [
  { href: '/docs', label: 'Documentation', activePrefix: '/docs' },
  {
    href: '/docs/reference/dashboard',
    label: 'Live dashboard',
    activePrefix: '/docs/reference/dashboard',
  },
  {
    href: '/docs/deployment/overview',
    label: 'Deploy',
    activePrefix: '/docs/deployment',
  },
];

export const footerNavigation: readonly NavigationItem[] = [
  { href: '/docs/security', label: 'Security' },
  { href: '/docs/reference/compatibility', label: 'Compatibility' },
  { href: '/research', label: 'Historical research' },
];

export const searchSuggestions = ['inventory', 'preparation', 'dashboard'];
