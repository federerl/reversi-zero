/**
 * The controls beside the board, and the parts of the agent's thinking worth
 * showing.
 *
 * The two selectors are separate on purpose. *Which generation* you play sets
 * how good the agent's intuition is; *how long it thinks* sets how much search
 * it does on top of that. Collapsing them into one "difficulty" slider would
 * hide the thing this project is actually about.
 *
 * Every opponent is labelled with a measured rating and its interval, never an
 * adjective. That is a standing rule in the repository, and it is also simply
 * more interesting to read.
 */

import { useId } from "react";

import { LEVELS, type Level } from "../engine/levels";
import type { ModelDescriptor } from "../engine/onnx";
import { BLACK, WHITE, type Player } from "../engine/rules";

/**
 * What to promise a player about a level.
 *
 * A time-capped level must not advertise a simulation count, because on most
 * devices it will not reach it -- on the machine this was written on, "Max"
 * gets about 560 of its 800 before the clock stops it. Naming the time is both
 * honest and the thing a player actually wants to know: how long they will be
 * waiting.
 */
function describeBudget(level: Level): string {
  if (level.maxMillis === undefined) return `${level.simulations} simulations`;
  return `up to ${(level.maxMillis / 1000).toFixed(1)} s per move`;
}

// ---------------------------------------------------------------------------

export function Score({
  black,
  white,
  toMove,
  humanColor,
}: {
  black: number;
  white: number;
  toMove: Player;
  humanColor: Player;
}) {
  return (
    <div className="flex items-center justify-between">
      <Side count={black} colour="black" active={toMove === BLACK} you={humanColor === BLACK} />
      <Side count={white} colour="white" active={toMove === WHITE} you={humanColor === WHITE} />
    </div>
  );
}

function Side({
  count,
  colour,
  active,
  you,
}: {
  count: number;
  colour: "black" | "white";
  active: boolean;
  you: boolean;
}) {
  return (
    <div className={`flex items-center gap-2 ${active ? "text-ink" : "text-muted"}`}>
      <span
        aria-hidden="true"
        className={`size-4 rounded-full border border-line ${
          colour === "black" ? "bg-disc-black" : "bg-disc-white"
        }`}
      />
      <span className="font-mono text-2xl font-semibold tabular-nums">{count}</span>
      <span className="text-xs uppercase tracking-wider text-muted">{you ? "you" : "agent"}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------

export function OpponentPicker({
  models,
  value,
  onChange,
  disabled,
}: {
  models: readonly ModelDescriptor[];
  value: string;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const selected = models.find((model) => model.id === value);
  const id = useId();

  return (
    <Field label="Opponent" htmlFor={id} hint={selected?.note}>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded border border-line bg-surface px-2 py-1.5 text-sm text-ink disabled:opacity-50"
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
            {model.elo !== undefined ? ` — ${Math.round(model.elo)} Elo` : ""}
          </option>
        ))}
      </select>

      {selected?.eloInterval && (
        <p className="mt-1 font-mono text-[0.7rem] text-muted tabular-nums">
          95% interval {Math.round(selected.eloInterval[0])}–{Math.round(selected.eloInterval[1])},
          random play = 0
        </p>
      )}
    </Field>
  );
}

export function LevelPicker({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const selected = LEVELS.find((level) => level.id === value);
  const id = useId();

  return (
    <Field label="Thinking time" htmlFor={id} hint={selected?.description}>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded border border-line bg-surface px-2 py-1.5 text-sm text-ink disabled:opacity-50"
      >
        {LEVELS.map((level) => (
          <option key={level.id} value={level.id}>
            {level.label} — {describeBudget(level)}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function SidePicker({
  value,
  onChange,
  disabled,
}: {
  value: Player;
  onChange: (player: Player) => void;
  disabled: boolean;
}) {
  const id = useId();

  return (
    <Field label="You play" htmlFor={id}>
      <select
        id={id}
        value={value === BLACK ? "black" : "white"}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value === "black" ? BLACK : WHITE)}
        className="w-full rounded border border-line bg-surface px-2 py-1.5 text-sm text-ink disabled:opacity-50"
      >
        <option value="black">Black (moves first)</option>
        <option value="white">White</option>
      </select>
    </Field>
  );
}

/**
 * A labelled control.
 *
 * The label is tied to its control by id rather than merely sitting above it.
 * Without that the two are unrelated as far as assistive technology is
 * concerned: a screen reader announces an unlabelled combo box, and clicking
 * the word "Opponent" does nothing. The child is cloned to receive the id
 * because that keeps every caller from having to invent one.
 */
function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string | undefined;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        htmlFor={htmlFor}
        className="mb-1 block text-[0.7rem] font-semibold uppercase tracking-wider text-muted"
      >
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs leading-snug text-muted">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------

export function Status({ line, thinking }: { line: string; thinking: boolean }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex min-h-14 items-center gap-2 border-l-2 border-accent-2 bg-surface-2 px-3 py-2 text-sm"
    >
      {thinking && <Spinner />}
      <span>{line}</span>
    </div>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      className="size-3.5 shrink-0 animate-spin rounded-full border-2 border-line-strong border-t-accent-2"
    />
  );
}

/**
 * How the agent rates the position, from the player's side of the board.
 *
 * Stated as "your chances" rather than the raw value, and the direction is
 * chosen deliberately: the agent's value is from whoever is about to move, so
 * showing it unflipped would make the bar swing wildly every turn for reasons
 * that have nothing to do with the game.
 */
export function WinProbability({ probability }: { probability: number }) {
  const percent = Math.round(probability * 100);

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted">
          Agent&rsquo;s estimate
        </span>
        <span className="font-mono text-sm tabular-nums">{percent}% you</span>
      </div>
      <div
        role="meter"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Your estimated chance of winning"
        className="h-2 w-full overflow-hidden rounded-full bg-surface-3"
      >
        <div
          className="h-full rounded-full bg-accent-2 transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "secondary",
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}) {
  const styles =
    variant === "primary"
      ? "bg-ink text-ground hover:opacity-90"
      : "border border-line bg-surface text-ink hover:bg-surface-2";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded px-3 py-1.5 text-sm font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Toast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="fixed bottom-4 left-1/2 z-50 max-w-[90vw] -translate-x-1/2 rounded border border-bad/40 bg-surface px-4 py-2 text-sm shadow-lg"
    >
      <span className="text-bad">{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="ml-3 text-muted underline underline-offset-2"
      >
        dismiss
      </button>
    </div>
  );
}
