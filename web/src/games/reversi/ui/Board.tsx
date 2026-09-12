/**
 * The board.
 *
 * Real `<button>` elements, one per square, rather than a canvas or a grid of
 * divs. That single choice gives keyboard focus, screen-reader labels and
 * testability for free instead of requiring three separate implementations of
 * them, and it costs nothing at 64 elements.
 *
 * Legality is never encoded in colour alone: a legal square gets a visible dot,
 * and its accessible name says so outright. A player who cannot distinguish the
 * dot from the felt still gets told.
 *
 * **How a disc turns over.** Each disc is one element with two faces, black on
 * the front and white on the back, and changing colour rotates it half a turn.
 * React keeps the same element on a square across renders, so a flip is a CSS
 * transition on that element rather than a swap, and the eye sees a disc turn
 * rather than a colour change. Discs further from the placed disc start turning
 * later, which is how the flips in a real game happen: the placed disc lands,
 * and the line it captures turns over away from it.
 */

import { useEffect, useRef } from "react";

import { indices, testBit } from "../engine/bitboard";
import { legalActions, passAction, type Action, type State } from "../engine/rules";

const FILES = "abcdefgh";

/** Milliseconds between one ring of flips and the next, measured from the placed disc. */
export const FLIP_STAGGER_MS = 60;

interface BoardProps {
  state: State;
  /** Squares the player may click. Empty while the agent is thinking. */
  interactive: boolean;
  lastMove: Action | null;
  /** Visits per action from the agent's last search, for the heat map. */
  visits?: readonly number[] | undefined;
  onPlay: (action: Action) => void;
}

export function Board({ state, interactive, lastMove, visits, onPlay }: BoardProps) {
  const size = state.size;
  const legal = new Set(legalActions(state).filter((a) => a !== passAction(size)));
  const gridRef = useRef<HTMLDivElement>(null);

  const black = new Set(indices(state.black));
  const white = new Set(indices(state.white));

  const peakVisits = visits ? Math.max(1, ...visits.slice(0, size * size)) : 1;
  const placed = lastMove !== null && lastMove !== passAction(size) ? lastMove : null;

  // Arrow keys move between squares. Without this the only way around a
  // 64-button grid is 64 presses of Tab.
  useEffect(() => {
    const grid = gridRef.current;
    if (grid === null) return;

    const onKeyDown = (event: KeyboardEvent) => {
      const deltas: Record<string, number> = {
        ArrowUp: -size,
        ArrowDown: size,
        ArrowLeft: -1,
        ArrowRight: 1,
      };
      const delta = deltas[event.key];
      if (delta === undefined) return;

      const focused = document.activeElement;
      if (!(focused instanceof HTMLElement) || focused.dataset["square"] === undefined) return;

      const from = Number(focused.dataset["square"]);
      const row = Math.floor(from / size);
      const to = from + delta;

      // Left and right must not run off one row onto the next.
      if (Math.abs(delta) === 1 && Math.floor(to / size) !== row) return;
      if (to < 0 || to >= size * size) return;

      event.preventDefault();
      grid.querySelector<HTMLElement>(`[data-square="${to}"]`)?.focus();
    };

    grid.addEventListener("keydown", onKeyDown);
    return () => grid.removeEventListener("keydown", onKeyDown);
  }, [size]);

  // The coordinates around the board, and the board itself, share one grid so
  // the labels line up with the squares whatever size the board is drawn at.
  // The template lives in CSS, because on a phone the labels are dropped and
  // their tracks have to collapse with them.
  return (
    <div className="board-frame">
      <ol className="board-ranks" aria-hidden="true">
        {Array.from({ length: size }, (_, row) => (
          <li key={row}>{row + 1}</li>
        ))}
      </ol>

      <div className="board-well">
        <div
          ref={gridRef}
          role="grid"
          aria-label={`Othello board, ${size} by ${size}`}
          className="board-grid"
          // Both axes, explicitly. Naming only the columns leaves the rows as
          // implicit tracks, and an implicit track is sized by its content -- so a
          // row holding a disc grew taller than an empty one and the squares
          // stopped being square.
          //
          // `minmax(0, 1fr)` rather than `1fr`: a bare `1fr` is `minmax(auto, 1fr)`,
          // whose floor is the content's minimum size, which would let a disc push
          // its row open again on a small board. The zero floor is what keeps the
          // grid in charge of the track sizes rather than what is sitting in them.
          style={{
            gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${size}, minmax(0, 1fr))`,
          }}
        >
          {Array.from({ length: size * size }, (_, square) => {
            const row = Math.floor(square / size);
            const column = square % size;
            const name = `${FILES[column]}${row + 1}`;

            const hasBlack = black.has(square);
            const hasWhite = white.has(square);
            const playable = interactive && legal.has(square);
            const share = visits ? (visits[square] ?? 0) / peakVisits : 0;

            // How far this square is from the disc just placed, in king moves.
            // Only discs that actually changed colour animate; the delay simply
            // makes those further along the captured line turn later.
            const ring =
              placed === null
                ? 0
                : Math.max(
                    Math.abs(row - Math.floor(placed / size)),
                    Math.abs(column - (placed % size)),
                  );

            return (
              <button
                key={square}
                type="button"
                data-square={square}
                disabled={!playable}
                onClick={() => playable && onPlay(square)}
                aria-label={describeSquare(name, hasBlack, hasWhite, playable)}
                className={[
                  "board-square",
                  playable ? "cursor-pointer" : "cursor-default",
                  placed === square ? "board-square-last" : "",
                ].join(" ")}
              >
                {/* The heat map sits under the disc so it never obscures the
                    position itself -- it is commentary, not state. */}
                {share > 0.02 && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-0 bg-accent"
                    style={{ opacity: Math.min(0.5, share * 0.5) }}
                  />
                )}

                {(hasBlack || hasWhite) && (
                  <span
                    aria-hidden="true"
                    data-disc={hasBlack ? "black" : "white"}
                    className={["disc", placed === square ? "disc-placed" : ""].join(" ")}
                    style={{ "--flip-delay": `${ring * FLIP_STAGGER_MS}ms` } as React.CSSProperties}
                  >
                    <span className="disc-face disc-face-black" />
                    <span className="disc-face disc-face-white" />
                  </span>
                )}

                {playable && !hasBlack && !hasWhite && (
                  <span aria-hidden="true" className="board-hint" />
                )}
              </button>
            );
          })}

          {/* Star points at the four classic intersections. Decoration only,
              positioned over the grid lines rather than inside any square. */}
          {size === 8 &&
            [
              [2, 2],
              [2, 6],
              [6, 2],
              [6, 6],
            ].map(([x, y]) => (
              <span
                key={`${x}${y}`}
                aria-hidden="true"
                className="board-star"
                style={{ left: `${(x! / size) * 100}%`, top: `${(y! / size) * 100}%` }}
              />
            ))}
        </div>
      </div>

      <span aria-hidden="true" className="board-spacer" />
      <ol className="board-files" aria-hidden="true">
        {Array.from({ length: size }, (_, column) => (
          <li key={column}>{FILES[column]}</li>
        ))}
      </ol>
    </div>
  );
}

function describeSquare(name: string, black: boolean, white: boolean, playable: boolean): string {
  if (black) return `${name}, black disc`;
  if (white) return `${name}, white disc`;
  return playable ? `${name}, empty — you can play here` : `${name}, empty`;
}

export function squareName(action: Action, size: number): string {
  if (action === passAction(size)) return "pass";
  return `${FILES[action % size]}${Math.floor(action / size) + 1}`;
}

export function occupiedBy(state: State, square: number): "black" | "white" | null {
  if (testBit(state.black, square)) return "black";
  if (testBit(state.white, square)) return "white";
  return null;
}
