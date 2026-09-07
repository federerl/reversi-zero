/**
 * Buttons, a switch, and the error toast.
 *
 * Three weights of button and no more: brass for the one action a screen is
 * about, a plain surface for everything else, and quiet text for the things a
 * player uses once. Every one is at least 44 pixels tall, which is the smallest
 * target a thumb finds reliably.
 */

import type { ReactNode } from "react";

export function Button({
  children,
  onClick,
  disabled,
  variant = "secondary",
  size = "md",
  title,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
  size?: "md" | "lg";
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`btn ${variant === "primary" ? "btn-primary" : ""} ${size === "lg" ? "btn-lg" : ""}`}
    >
      {children}
    </button>
  );
}

/**
 * A labelled checkbox that reads as part of the game rather than as a form.
 *
 * Still a real `<input type="checkbox">`: it is what a screen reader and a
 * keyboard already know how to operate, and hiding it behind a div would mean
 * reimplementing both.
 */
export function Toggle({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: () => void;
  children: ReactNode;
}) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-2.5 text-[0.95rem] text-ink-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="size-4 accent-accent"
      />
      {children}
    </label>
  );
}

export function Toast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="fixed bottom-4 left-1/2 z-50 flex max-w-[90vw] -translate-x-1/2 items-center gap-3 rounded-lg border border-bad/40 bg-surface px-4 py-3 shadow-2xl"
    >
      <span className="text-[0.95rem] text-bad">{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="shrink-0 text-sm text-muted underline underline-offset-2 hover:text-ink"
      >
        Dismiss
      </button>
    </div>
  );
}
