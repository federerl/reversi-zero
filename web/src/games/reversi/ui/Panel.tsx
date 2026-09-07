/**
 * The two player plates, the turn indicator, and the control panel.
 *
 * The two selectors are separate on purpose. *Which generation* you play sets
 * how good the agent's intuition is; *how long it thinks* sets how much search
 * it does on top of that. Collapsing them into one difficulty slider would hide
 * the thing this project is actually about.
 *
 * Every opponent is labelled with a measured rating, never an adjective. That
 * is a standing rule in the repository. The interval and the description of what
 * an opponent *is* sit in a disclosure: true, worth reading once, and not what a
 * player needs while it is their move.
 */

import { useId } from "react";

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

export function signedElo(elo: number): string {
  const rounded = Math.round(elo);
  return `${rounded > 0 ? "+" : ""}${rounded}`;
}

// ---------------------------------------------------------------------------
// The players
// ---------------------------------------------------------------------------

/**
 * One side of the game: the disc, who it is, their rating if they have one, and
 * their count. One plate sits above the board and one below, the way a name and
 * a clock frame a chess board -- so a score is never floating at a corner with
 * nothing to attach it to.
 */
export function PlayerPlate({
  colour,
  name,
  rating,
  count,
  active,
  thinking,
}: {
  colour: "black" | "white";
  name: string;
  /** The opponent's measured strength. The human has none, and gets none. */
  rating?: number | undefined;
  count: number;
  active: boolean;
  thinking?: boolean;
}) {
  return (
    <div className={`plate ${active ? "text-ink" : "text-muted"}`}>
      <span
        aria-hidden="true"
        className={`chip ${colour === "black" ? "chip-black" : "chip-white"} size-8 shrink-0`}
      />

      <span className="flex min-w-0 flex-col leading-tight">
        <span className="flex items-center gap-2">
          {active && !thinking && (
            <span aria-hidden="true" className="size-2 shrink-0 rounded-full bg-accent" />
          )}
          {thinking && <Spinner />}
          <span className="truncate text-[1.05rem]">{name}</span>
        </span>
        {rating !== undefined && <span className="text-sm text-muted">{signedElo(rating)} Elo</span>}
      </span>

      <span className="display ml-auto text-4xl sm:text-5xl">{count}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// What is happening
// ---------------------------------------------------------------------------

/**
 * The turn indicator: the one thing a player checks between moves.
 *
 * It keeps the live region, so a screen reader hears every change, and it holds
 * the detailed sentence as well -- "You win, 34 to 30" says more than "Game
 * over", and the reducer already writes it.
 */
export function StatusPill({
  headline,
  detail,
  tone,
  thinking,
}: {
  headline: string;
  detail?: string | undefined;
  tone: "you" | "quiet";
  thinking: boolean;
}) {
  return (
    <div role="status" aria-live="polite" className="status-pill" data-tone={tone}>
      {thinking && <Spinner />}
      <span className="min-w-0">
        <span className="font-medium">{headline}</span>
        {detail && <span className="text-muted"> {detail}</span>}
      </span>
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
      <span className="shrink-0 text-[0.95rem] text-ink-2">Your chances</span>
      <div
        role="meter"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Your estimated chance of winning"
        className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className="h-full rounded-full bg-ink-2 transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="display shrink-0 text-xl">{percent}%</span>
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
  const id = useId();
  return (
    <div className="control-row">
      <label htmlFor={id}>Opponent</label>
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
            {model.elo !== undefined ? `, ${signedElo(model.elo)} Elo` : ""}
          </option>
        ))}
      </select>
    </div>
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
  const id = useId();
  return (
    <div className="control-row">
      <label htmlFor={id}>Thinking time</label>
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
              {rated ? `, ${signedElo(rated.elo)} Elo` : ""}, {describeBudget(level)}
            </option>
          );
        })}
      </select>
    </div>
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
    <div className="control-row">
      <label htmlFor={id}>You play</label>
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
    </div>
  );
}

/**
 * What this opponent is, and how sure the rating is.
 *
 * A disclosure rather than a paragraph: the interval matters to a reader who
 * wants to check the claim, and it is noise to a player who wants a game. Both
 * are served, in that order.
 */
export function OpponentDetails({
  model,
  levelId,
  showLevel,
}: {
  model: ModelDescriptor | undefined;
  levelId: string;
  showLevel: boolean;
}) {
  const level = LEVELS.find((entry) => entry.id === levelId);
  const levelRating = ratingFor(levelId);
  if (model === undefined) return null;

  return (
    <details className="disclosure">
      <summary>About this opponent</summary>
      <div className="flex flex-col gap-2 pb-1 text-sm leading-relaxed text-muted">
        {model.note && <p>{model.note}</p>}
        {model.eloInterval && (
          <p>
            Rated {signedElo(model.elo ?? 0)}, with a 95% interval from{" "}
            {Math.round(model.eloInterval[0])} to {Math.round(model.eloInterval[1])}. Random play is
            0.
          </p>
        )}
        {showLevel && level && (
          <p>
            {level.label} searches {describeBudget(level)}
            {levelRating ? `, and rates ${signedElo(levelRating.elo)} on the same scale` : ""}.{" "}
            {level.description}
          </p>
        )}
      </div>
    </details>
  );
}
