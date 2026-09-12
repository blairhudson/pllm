export type NavigationItem = {
  href: string;
  label: string;
  activePrefix?: string;
};

export const mainNavigation: readonly NavigationItem[] = [
  { href: '/docs', label: 'Documentation', activePrefix: '/docs' },
  { href: '/research', label: 'Research', activePrefix: '/research' },
];

export const footerNavigation: readonly NavigationItem[] = [
  { href: '/docs/security', label: 'Security' },
  { href: '/docs/reference/compatibility', label: 'Compatibility' },
  { href: '/research', label: 'Research' },
];

export const searchSuggestions = ['inventory', 'preparation', 'dashboard'];
