import type { ReactNode } from 'react';

const statuses = ['Available', 'Experimental', 'Planned', 'Not supported', 'Not evaluated', 'Not applicable'] as const;

type Status = (typeof statuses)[number];

export function CapabilityStatus({ status, children }: { status: Status; children: ReactNode }) {
  const kind = status.toLowerCase().replaceAll(' ', '-');
  return (
    <div className="capability-status" data-status={kind}>
      <strong>{status}</strong>
      <div>{children}</div>
    </div>
  );
}
