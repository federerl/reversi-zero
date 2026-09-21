/**
 * The end of a puzzle, as something that happens rather than something you
 * notice.
 *
 * A result that appears as changed text in a side panel is not a result. This is
 * a real `<dialog>`, which brings focus trapping and Escape-to-dismiss with it,
 * and dismissing leaves the finished board on screen to count — which is the
 * whole reason the ending is played out at all.
 *
 * It carries the explanation because nothing else does. The page stays silent
 * while the ending is in progress, so this is the only place that can say *which
 * move* lost it, and a loss with no explanation is the complaint this feature
 * exists to answer.
 */

import { useEffect, useRef } from "react";

import { squareName } from "../ui/Board";
import type { Outcome } from "../state/puzzle";

export function ResultDialog({
  outcome,
  size,
  hasNext,
  onRetry,
  onNext,
  onShowLine,
  onDismiss,
}: {
  outcome: Outcome | null;
  size: number;
  hasNext: boolean;
  onRetry: () => void;
  onNext: () => void;
  onShowLine: () => void;
  onDismiss: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const open = outcome !== null;

  useEffect(() => {
    const dialog = ref.current;
    if (dialog === null) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  if (outcome === null) return null;

  const { discs, margin, won } = outcome;

  return (
    <dialog
      ref={ref}
      // The same frame as the game's own result dialog: a centred card, not a
      // full-bleed box. `.game-over` carries only the entrance animation.
      className="game-over panel m-auto w-[min(92vw,30rem)] p-0 text-ink backdrop:bg-black/65"
      aria-labelledby="puzzle-result"
      onClose={onDismiss}
      onCancel={onDismiss}
    >
      <div className="flex flex-col gap-4 p-7">
        <div>
          <h2 id="puzzle-result" className="display text-3xl">
            {won ? "Won it." : margin === 0 ? "Drawn." : "Lost it."}
          </h2>
          <p className="mt-1 text-lg text-ink-2">
            {discs.mine}&ndash;{discs.theirs}
            {margin !== 0 && <>, by {Math.abs(margin)}</>}.
          </p>
        </div>

        <div className="flex flex-col gap-2 text-[0.95rem] leading-relaxed text-muted">
          <Explanation outcome={outcome} size={size} won={won} />
        </div>

        <div className="flex flex-wrap gap-2">
          {won && hasNext ? (
            <button type="button" className="btn btn-primary" onClick={onNext}>
              Next puzzle
            </button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={onRetry}>
              Try again
            </button>
          )}
          <button type="button" className="btn" onClick={onShowLine}>
            Show the winning line
          </button>
          {won && !hasNext && (
            <button type="button" className="btn" onClick={onDismiss}>
              Look at the board
            </button>
          )}
          {!won && hasNext && (
            <button type="button" className="btn" onClick={onNext}>
              Skip to the next
            </button>
          )}
        </div>
      </div>
    </dialog>
  );
}

function Explanation({
  outcome,
  size,
  won,
}: {
  outcome: Outcome;
  size: number;
  won: boolean;
}) {
  const opening = squareName(outcome.openingMove, size);

  if (won) {
    return (
      <>
        <p>
          You held it from a position worth {outcome.best > 0 ? "+" : ""}
          {outcome.best}, and every reply was the best one available &mdash; the
          opponent here does not make mistakes.
        </p>
        {outcome.openingMargin < outcome.best && (
          <p>
            Your opening move {opening} was worth {outcome.openingMargin > 0 ? "+" : ""}
            {outcome.openingMargin}, where the best on offer was {outcome.best > 0 ? "+" : ""}
            {outcome.best}. Still a win, and a narrower one than it had to be.
          </p>
        )}
      </>
    );
  }

  return (
    <>
      {outcome.openingMargin <= 0 ? (
        <p>
          <span className="text-ink-2">{opening} was already the mistake.</span> It
          gives up a position worth {outcome.best > 0 ? "+" : ""}
          {outcome.best} and leaves you{" "}
          {outcome.openingMargin === 0
            ? "level"
            : `losing by ${Math.abs(outcome.openingMargin)}`}{" "}
          with best play from there. Nothing later could have recovered it.
        </p>
      ) : outcome.lostItAt !== null ? (
        <p>
          <span className="text-ink-2">
            {squareName(outcome.lostItAt.move, size)} is where it went.
          </span>{" "}
          Before it you were winning by {outcome.lostItAt.wasWorth}; after it the best
          available was no longer a win. Your opening move {opening} was fine &mdash; it
          was worth {outcome.openingMargin > 0 ? "+" : ""}
          {outcome.openingMargin}.
        </p>
      ) : (
        <p>
          Your opening move {opening} was worth {outcome.openingMargin > 0 ? "+" : ""}
          {outcome.openingMargin}, and the win was there to hold.
        </p>
      )}
      <p>
        The opponent played perfectly throughout, so nothing here was luck. The
        winning line is one click away.
      </p>
    </>
  );
}
