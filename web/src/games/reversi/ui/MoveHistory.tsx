/**
 * Stepping back through a finished or in-progress game.
 *
 * A board game you cannot look back at is a board game you cannot learn from.
 * The interesting question after losing is never "what was the score" -- it is
 * "where did that go wrong", and answering it needs the position from twelve
 * moves ago on the screen, not in memory.
 *
 * This is a scrubber rather than a move list, and the choice is about the space
 * it has to live in. A list of sixty moves needs a scrolling box beside the
 * board, which is the largest thing in the panel and pushes everything a player
 * acts on below the fold. A slider is one row: drag it, or step with the arrows,
 * and the board follows. The move that was played is named next to it, so the
 * position and its notation are legible together.
 *
 * The control is deliberately absent until there is something to review. An
 * inert slider on the opening position is a promise the interface has not yet
 * earned.
 */

import { squareName } from "./Board";
import { Button } from "../../../shared/ui/Button";
import { passAction } from "../engine/rules";
import { isLive, moveCount, viewedPly, type Game } from "../state/game";

/** "d3, you" -- which square, and whose move it was. */
function describeMove(game: Game, ply: number): string {
  const turn = game.history[ply];
  const before = game.history[ply - 1];
  if (turn === undefined || turn.move === null || before === undefined) {
    return "the opening position";
  }

  // Whoever was to move in the position *before* this one is the mover. Reading
  // it off the position afterwards would be wrong: when the opponent has no
  // legal reply the same player moves twice in a row, so `toMove` after a move
  // is not reliably the other side.
  const who = before.state.toMove === game.humanColor ? "you" : "the AI";

  const size = turn.state.size;
  if (turn.move === passAction(size)) return `pass, ${who}`;
  return `${squareName(turn.move, size)}, ${who}`;
}

export function MoveHistory({
  game,
  onView,
  onLive,
}: {
  game: Game;
  onView: (ply: number) => void;
  onLive: () => void;
}) {
  const total = moveCount(game);
  if (total < 1) return null;

  const ply = viewedPly(game);
  const live = isLive(game);

  return (
    <div className="flex flex-col gap-2 border-t border-line pt-3">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor="move-scrubber" className="text-[0.95rem] text-ink-2">
          Review
        </label>
        <span className="text-sm text-muted">
          {ply === 0 ? "Start" : `Move ${ply} of ${total}`} &middot; {describeMove(game, ply)}
        </span>
      </div>

      <input
        id="move-scrubber"
        type="range"
        min={0}
        max={total}
        step={1}
        value={ply}
        aria-label="Move to review"
        // The slider reports the position, not the move number, so a screen
        // reader hears the same thing the sighted label says.
        aria-valuetext={`${ply === 0 ? "start" : `move ${ply}`}, ${describeMove(game, ply)}`}
        onChange={(event) => onView(Number(event.target.value))}
        className="move-scrubber"
      />

      <div className="flex flex-wrap items-center gap-2">
        {/* "Previous" and "Next", not "Back" and "Forward": "Take back" is a
            button in the same panel, and two controls whose names start the same
            way are two controls a player has to read twice. */}
        <Button onClick={() => onView(ply - 1)} disabled={ply <= 0}>
          Previous
        </Button>
        <Button onClick={() => onView(ply + 1)} disabled={ply >= total}>
          Next
        </Button>
        {!live && (
          <Button variant="primary" onClick={onLive}>
            Latest
          </Button>
        )}
      </div>
    </div>
  );
}
