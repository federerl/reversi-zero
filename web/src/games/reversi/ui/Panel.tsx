/**
 * The scoreboard, the controls, and the parts of the agent's thinking worth
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

import { useId, type ReactNode } from "react";

import { LEVELS, type Level } from "../engine/levels";
import modelsManifest from "../engine/models.json";
import type { ModelDescriptor } from "../engine/onnx";
import { BLACK, WHITE, type Player } from "../engine/rules";

const LEVEL_RATINGS = modelsManifest.levels ?? [];

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
  return `up to ${(level.maxMillis / 1000).toFixed(1)} s a move`;
}

/**
 * A level's measured rating, if it has one.
 *
 * From the calibration report, never typed here. Reading the numbers from the
 * file that measured them means the interface cannot claim a separation nobody
 * checked.
 */
function ratingFor(id: string): { elo: number; interval: [number, number] } | undefined {
  const found = (
    LEVEL_RATINGS as Array<{ id: string; elo: number; eloInterval: [number, number] }>
  ).find((entry) => entry.id === id);
  return found ? { elo: found.elo, interval: found.eloInterval } : undefined;
}

function signed(elo: number): string {
  const rounded = Math.round(elo);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

// ---------------------------------------------------------------------------
// The scoreboard
// ---------------------------------------------------------------------------

/**
 * Two numerals and two discs, the way a counter sits beside a board. The side
 * to move carries the brass mark; nothing else on the strip is coloured.
 */
export function Scoreboard({
  black,
  white,
  toMove,
  humanColor,
  opponentLabel,
  terminal,
}: {
  black: number;
  white: number;
  toMove: Player;
  humanColor: Player;
  opponentLabel: string;
  terminal: boolean;
}) {
  const nameFor = (colour: Player) => (colour === humanColor ? "You" : opponentLabel);
  return (
    <div className="flex items-end justify-between gap-6">
      <Side
        count={black}
        colour="black"
        name={nameFor(BLACK)}
        active={!terminal && toMove === BLACK}
        align="left"
      />
      <Side
        count={white}
        colour="white"
        name={nameFor(WHITE)}
        active={!terminal && toMove === WHITE}
        align="right"
      />
    </div>
  );
}

function Side({
  count,
  colour,
  name,
  active,
  align,
}: {
  count: number;
  colour: "black" | "white";
  name: string;
  active: boolean;
  align: "left" | "right";
}) {
  const disc = (
    <span
      aria-hidden="true"
      className={`size-7 shrink-0 rounded-full shadow ${
        colour === "black" ? "bg-disc-black ring-1 ring-line-strong" : "bg-disc-white"
      }`}
    />
  );
  return (
    <div
      className={`flex items-center gap-3 ${align === "right" ? "flex-row-reverse text-right" : ""}`}
    >
      {disc}
      <div>
        <div className={`score-number text-5xl sm:text-6xl ${active ? "text-ink" : "text-ink-2"}`}>
          {count}
        </div>
        <div
          className={`mt-1 flex items-center gap-1.5 text-sm ${
            align === "right" ? "flex-row-reverse" : ""
          } ${active ? "text-ink" : "text-muted"}`}
        >
          {active && <span aria-hidden="true" className="size-1.5 rounded-full bg-accent" />}
          <span>{active ? `${name} to move` : name}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The controls
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
    <Control label="Opponent" htmlFor={id} hint={selected?.note}>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="control-select"
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
            {model.elo !== undefined ? `, ${signed(model.elo)} Elo` : ""}
          </option>
        ))}
      </select>
      {selected?.eloInterval && (
        <span className="text-sm text-muted">
          95% interval {Math.round(selected.eloInterval[0])} to{" "}
          {Math.round(selected.eloInterval[1])}. Random play is 0.
        </span>
      )}
    </Control>
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
  const selectedRating = ratingFor(value);
  const id = useId();

  return (
    <Control label="Thinking time" htmlFor={id} hint={selected?.description}>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="control-select"
      >
        {LEVELS.map((level) => {
          const rated = ratingFor(level.id);
          return (
            <option key={level.id} value={level.id}>
              {level.label}
              {rated ? `, ${signed(rated.elo)} Elo` : ""}, {describeBudget(level)}
            </option>
          );
        })}
      </select>
      {selectedRating && (
        <span className="text-sm text-muted">
          95% interval {Math.round(selectedRating.interval[0])} to{" "}
          {Math.round(selectedRating.interval[1])}. Random play is 0.
        </span>
      )}
    </Control>
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
    <Control label="You play" htmlFor={id}>
      <select
        id={id}
        value={value === BLACK ? "black" : "white"}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value === "black" ? BLACK : WHITE)}
        className="control-select"
      >
        <option value="black">Black, moves first</option>
        <option value="white">White</option>
      </select>
    </Control>
  );
}

/**
 * A control with its name beside it.
 *
 * The label is tied to its control by id rather than merely sitting near it.
 * Without that the two are unrelated as far as assistive technology is
 * concerned: a screen reader announces an unlabelled combo box, and clicking
 * the word "Opponent" does nothing. The hint, when there is one, is a second
 * line under the control, in plain words.
 */
function Control({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string | undefined;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <label htmlFor={htmlFor} className="w-28 shrink-0 text-[0.95rem] text-ink-2">
          {label}
        </label>
        <div className="flex min-w-0 flex-col gap-1">{children}</div>
      </div>
      {hint && <p className="ml-0 text-sm leading-snug text-muted sm:ml-31">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// What is happening
// ---------------------------------------------------------------------------

export function Status({ line, thinking }: { line: string; thinking: boolean }) {
  return (
    <div role="status" aria-live="polite" className="flex min-h-7 items-center gap-2 text-[1.05rem]">
      {thinking && <Spinner />}
      <span>{line}</span>
    </div>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      className="size-3.5 shrink-0 animate-spin rounded-full border-2 border-line-strong border-t-accent"
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
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-[0.95rem] text-ink-2">Your chances</span>
      <div
        role="meter"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Your estimated chance of winning"
        className="h-1.5 w-40 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className="h-full rounded-full bg-ink-2 transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="score-number text-xl">{percent}%</span>
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "secondary",
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}) {
  const styles =
    variant === "primary"
      ? "bg-ink text-ground hover:opacity-90"
      : "border border-line-strong bg-surface text-ink hover:bg-surface-2";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-3.5 py-1.5 text-[0.95rem] font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Toast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="fixed bottom-4 left-1/2 z-50 max-w-[90vw] -translate-x-1/2 rounded-md border border-bad/40 bg-surface px-4 py-2 text-sm shadow-lg"
    >
      <span className="text-bad">{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="ml-3 text-muted underline underline-offset-2"
      >
        Dismiss
      </button>
    </div>
  );
}
