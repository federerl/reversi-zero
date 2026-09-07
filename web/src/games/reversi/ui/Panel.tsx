/**
 * The two player plates, the turn indicator, and the control panel.
 *
 * What a player picks is a **level**: Level 1 to Level 6, with a word for how
 * hard it is. What a level actually *is* -- which checkpoint of the training run,
 * or which hand-written baseline, and what it measured against everything else
 * -- is one click away under "About this level", because it is true and worth
 * reading once and it is not what anybody needs while it is their move.
 *
 * Thinking time stays a separate control for the levels a network plays. It is
 * the second dial this project is about: *which* network sets how good the
 * intuition is, and *how long it thinks* sets how much search runs on top. A
 * single difficulty slider would hide that, and the levels would stop being
 * measurable things.
 */

import { useId } from "react";

import { LADDER, rungName, type Rung } from "../ladder";
import { LEVELS, type Level } from "../engine/levels";
import modelsManifest from "../engine/models.json";
import { BLACK, WHITE, type Player } from "../engine/rules";

const LEVEL_RATINGS = modelsManifest.levels ?? [];

/**
 * What to promise a player about a thinking time.
 *
 * A time-capped level must not advertise a simulation count, because on most
 * devices it will not reach it -- on the machine this was written on, "Max"
 * gets about 560 of its 800 before the clock stops it. Naming the time is both
 * honest and the thing a player actually wants to know: how long they will be
 * waiting.
 */
function describeBudget(level: Level): string {
  if (level.maxMillis === undefined) return `${level.simulations} simulations a move`;
  return `up to ${(level.maxMillis / 1000).toFixed(1)} seconds a move`;
}

/**
 * The same promise, short enough for the select.
 *
 * A native select does not wrap and does not grow, so an option longer than the
 * control is silently cut off mid-word -- "Strong, up to 1.2 second". The prose
 * version above is for the note, where there is room for a sentence.
 */
function shortBudget(level: Level): string {
  if (level.maxMillis === undefined) return `${level.simulations} simulations`;
  return `up to ${(level.maxMillis / 1000).toFixed(1)} s`;
}

/**
 * A thinking time's measured rating, if it has one.
 *
 * From the calibration report, never typed here. That report is its own
 * tournament on one checkpoint, so its numbers are only ever shown against the
 * thinking time they belong to -- never mixed into the ladder, which is ordered
 * by a different tournament.
 */
function ratingForBudget(id: string): { elo: number; interval: [number, number] } | undefined {
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
 * One side of the game: the disc, who it is, and their count. One plate sits
 * above the board and one below, the way a name and a clock frame a chess board
 * -- so a score is never floating at a corner with nothing to attach it to.
 */
export function PlayerPlate({
  colour,
  name,
  detail,
  count,
  active,
  thinking,
}: {
  colour: "black" | "white";
  name: string;
  /** A word under the name: how hard this level is. The human gets none. */
  detail?: string | undefined;
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
        {detail !== undefined && <span className="text-sm text-muted">{detail}</span>}
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
 * This is also the live region, so a screen reader hears every change, and it
 * carries the result sentence at the end of a game.
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

/**
 * The ladder, as a player meets it.
 *
 * The option values are opponent ids rather than level numbers, because the
 * level *is* a view of which opponent is playing -- there is no second piece of
 * state to keep in step, and the ladder can gain a rung without anything here
 * changing.
 */
export function LevelPicker({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (modelId: string) => void;
  disabled: boolean;
}) {
  const id = useId();
  return (
    <div className="control-row">
      <label htmlFor={id}>Level</label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="control-select"
      >
        {LADDER.map((rung) => (
          <option key={rung.modelId} value={rung.modelId}>
            {rungName(rung)}, {rung.word}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ThinkingTimePicker({
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
        {LEVELS.map((level) => (
          <option key={level.id} value={level.id}>
            {level.label}, {shortBudget(level)}
          </option>
        ))}
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
 * What this level really is, for anyone who wants to know.
 *
 * Everything technical about an opponent lives here: which checkpoint of the
 * self-play run it is, what it measured, and how sure that measurement is. A
 * player who never opens it loses nothing; a reader who does gets the whole
 * claim, including its error bar.
 */
export function LevelDetails({
  rung,
  levelId,
}: {
  rung: Rung;
  levelId: string;
}) {
  const budget = LEVELS.find((entry) => entry.id === levelId);
  const budgetRating = ratingForBudget(levelId);

  return (
    <details className="disclosure">
      <summary>About {rungName(rung).toLowerCase()}</summary>
      <div className="flex flex-col gap-2 pb-1 text-sm leading-relaxed text-muted">
        <p>
          <span className="text-ink-2">{rung.opponentLabel}.</span> {rung.note}
        </p>
        <p>
          It rates {signedElo(rung.elo)} against the other levels, where random play is 0
          {rung.interval && (
            <>
              , with a 95% interval from {Math.round(rung.interval[0])} to{" "}
              {Math.round(rung.interval[1])}
            </>
          )}
          . Neighbouring levels can overlap, which is the honest way to say they are close.
        </p>
        {rung.usesNetwork && budget && (
          <p>
            It is searching {describeBudget(budget)}
            {budgetRating && (
              <>
                , a setting that rates {signedElo(budgetRating.elo)} in its own tournament on the
                final network
              </>
            )}
            . {budget.description}
          </p>
        )}
      </div>
    </details>
  );
}
