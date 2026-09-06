/**
 * The frame every page shares: a header with the site's name and a way back to
 * the games, and a footer for the fine print.
 *
 * Kept deliberately small. The board and its controls are the page; this only
 * has to say where you are and how to leave.
 */

import type { ReactNode } from "react";

export const SITE_NAME = "reversi-zero";

export function Shell({
  title,
  lead,
  children,
  footer,
}: {
  /** The page's own heading. Omit on the hub, whose heading is the site name. */
  title?: string;
  lead?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-12 pt-6">
      <header className="mb-6 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <div>
          {title === undefined ? (
            <h1 className="text-2xl font-bold tracking-tight">{SITE_NAME}</h1>
          ) : (
            <>
              <a
                href="/"
                className="text-xs font-semibold uppercase tracking-wider text-muted hover:text-ink"
              >
                {SITE_NAME} · all games
              </a>
              <h1 className="mt-1 text-2xl font-bold tracking-tight">{title}</h1>
            </>
          )}
          {lead && <p className="mt-1 max-w-prose text-sm text-muted">{lead}</p>}
        </div>
      </header>

      {children}

      {footer && (
        <footer className="mt-8 max-w-prose text-xs leading-relaxed text-muted">{footer}</footer>
      )}
    </div>
  );
}
