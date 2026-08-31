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
 */

import { useEffect, useRef } from "react";

import { indices, testBit } from "../engine/bitboard";
import { legalActions, passAction, type Action, type State } from "../engine/rules";

const FILES = "abcdefgh";

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

  return (
    <div className="rounded-md bg-board-edge p-2 shadow-[0_1px_2px_rgba(0,0,0,.12),0_10px_26px_rgba(0,0,0,.10)]">
      <div
        ref={gridRef}
        role="grid"
        aria-label={`Reversi board, ${size} by ${size}`}
        className="grid aspect-square w-full gap-[2px] border-2 border-board-line bg-board-line"
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

          return (
            <button
              key={square}
              type="button"
              data-square={square}
              disabled={!playable}
              onClick={() => playable && onPlay(square)}
              aria-label={describeSquare(name, hasBlack, hasWhite, playable)}
              className={[
                "relative grid place-items-center bg-board p-0",
                playable ? "cursor-pointer" : "cursor-default",
                lastMove === square ? "outline outline-2 -outline-offset-2 outline-accent-2" : "",
              ].join(" ")}
            >
              {/* The heat map sits under the disc so it never obscures the
                  position itself -- it is commentary, not state. */}
              {share > 0.02 && (
                <span
                  aria-hidden="true"
                  className="absolute inset-0 bg-accent-2"
                  style={{ opacity: Math.min(0.55, share * 0.55) }}
                />
              )}

              {(hasBlack || hasWhite) && (
                <span
                  aria-hidden="true"
                  data-disc={hasBlack ? "black" : "white"}
                  className={[
                    "relative aspect-square w-4/5 rounded-full transition-colors duration-300",
                    "shadow-[0_1px_2px_rgba(0,0,0,.45)]",
                    hasBlack ? "bg-disc-black" : "bg-disc-white",
                  ].join(" ")}
                />
              )}

              {playable && !hasBlack && !hasWhite && (
                <span
                  aria-hidden="true"
                  className="relative aspect-square w-[22%] rounded-full bg-white/35"
                />
              )}
            </button>
          );
        })}
      </div>
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
