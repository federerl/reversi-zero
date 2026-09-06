/**
 * The frame every page shares: a thin bar with the site's name or the way back
 * to the games, the page's title, and the theme and sound switches.
 *
 * The bar is one line tall on purpose. A game page is a landscape screen with a
 * square in it, and every line of chrome above the board is a line taken from
 * the board.
 */

import { useState, type ReactNode } from "react";

import { sounds } from "../sound";
import {
  applyChoice,
  readChoice,
  resolveTheme,
  safeStorage,
  toggledChoice,
  type Theme,
  type ThemeChoice,
} from "../theme";

export const SITE_NAME = "reversi-zero";

export function Shell({
  title,
  children,
  wide = false,
}: {
  /** The page's own heading. Omit on the hub, whose heading is the site name. */
  title?: string;
  children: ReactNode;
  /** Use the whole window width. Game pages do; the hub reads better narrower. */
  wide?: boolean;
}) {
  return (
    <div className={`flex min-h-dvh flex-col ${wide ? "" : "mx-auto max-w-5xl"}`}>
      <header className="flex h-12 shrink-0 items-center gap-4 px-4 sm:px-6">
        {title === undefined ? (
          <p className="font-display text-2xl font-semibold tracking-tight">{SITE_NAME}</p>
        ) : (
          <>
            <a href="/" className="text-sm text-muted hover:text-ink">
              All games
            </a>
            <span aria-hidden="true" className="h-4 w-px bg-line-strong" />
            <h1 className="font-display text-2xl font-semibold leading-none tracking-tight">
              {title}
            </h1>
          </>
        )}
        <div className="ml-auto flex items-center gap-1">
          <ThemeSwitch />
          <SoundSwitch />
        </div>
      </header>

      <div className="flex-1 px-4 pb-6 sm:px-6">{children}</div>
    </div>
  );
}

function useTheme(): [Theme, () => void] {
  const [choice, setChoice] = useState<ThemeChoice>(() => readChoice(safeStorage()));
  const theme = resolveTheme(choice);
  const toggle = () => {
    const next = toggledChoice(choice);
    applyChoice(next);
    setChoice(next);
  };
  return [theme, toggle];
}

function ThemeSwitch() {
  const [theme, toggle] = useTheme();
  const toLight = theme === "dark";
  return (
    <IconButton
      label={toLight ? "Switch to the light theme" : "Switch to the dark theme"}
      onClick={toggle}
    >
      {toLight ? (
        <>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" />
        </>
      ) : (
        <path d="M14.5 3.5a7 7 0 1 0 6 10.5A8 8 0 0 1 14.5 3.5z" />
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
