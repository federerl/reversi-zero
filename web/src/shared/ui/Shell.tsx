/**
 * The frame every page shares: a header with the site's name, a way back to the
 * games, the theme and sound switches, and a footer for the fine print.
 *
 * Kept deliberately small. The board and its controls are the page; this only
 * has to say where you are, how to leave, and how you like it to look.
 */

import { useEffect, useState, type ReactNode } from "react";

import { sounds } from "../sound";
import {
  applyChoice,
  readChoice,
  resolveTheme,
  safeStorage,
  systemPrefersDark,
  toggledChoice,
  type Theme,
  type ThemeChoice,
} from "../theme";

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
      <header className="mb-6 flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
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

        <div className="flex items-center gap-1">
          <ThemeSwitch />
          <SoundSwitch />
        </div>
      </header>

      {children}

      {footer && (
        <footer className="mt-8 max-w-prose text-xs leading-relaxed text-muted">{footer}</footer>
      )}
    </div>
  );
}

function useTheme(): [Theme, () => void] {
  const [choice, setChoice] = useState<ThemeChoice>(() => readChoice(safeStorage()));
  const [prefersDark, setPrefersDark] = useState(systemPrefersDark);

  useEffect(() => {
    if (typeof matchMedia !== "function") return;
    const query = matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => setPrefersDark(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  const theme = resolveTheme(choice, prefersDark);
  const toggle = () => {
    const next = toggledChoice(choice, prefersDark);
    applyChoice(next);
    setChoice(next);
  };
  return [theme, toggle];
}

function ThemeSwitch() {
  const [theme, toggle] = useTheme();
  const toDark = theme === "light";
  return (
    <IconButton
      label={toDark ? "Switch to the dark theme" : "Switch to the light theme"}
      onClick={toggle}
    >
      {toDark ? (
        // a moon
        <path d="M14.5 3.5a7 7 0 1 0 6 10.5A8 8 0 0 1 14.5 3.5z" />
      ) : (
        // a sun
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" />
        </>
      )}
    </IconButton>
  );
}

function SoundSwitch() {
  const [muted, setMuted] = useState(sounds.muted);
  const toggle = () => {
    sounds.setMuted(!muted);
    setMuted(!muted);
  };
  return (
    <IconButton label={muted ? "Unmute sounds" : "Mute sounds"} onClick={toggle} pressed={muted}>
      <path d="M4 9v6h3.5L12 19V5L7.5 9H4z" />
      {muted ? (
        <path d="M16 9l5 6M21 9l-5 6" />
      ) : (
        <path d="M15.5 8.5a5 5 0 0 1 0 7M18 6a8.5 8.5 0 0 1 0 12" />
      )}
    </IconButton>
  );
}

function IconButton({
  label,
  onClick,
  pressed,
  children,
}: {
  label: string;
  onClick: () => void;
  pressed?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      className="grid size-9 place-items-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink"
    >
      <svg
        viewBox="0 0 24 24"
        width="18"
        height="18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {children}
      </svg>
    </button>
  );
}
