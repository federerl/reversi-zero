/**
 * The end of a game.
 *
 * Who won, by how much, and the two things a player wants next: the same game
 * again, or the same game from the other side. The native `<dialog>` element
 * gives focus trapping and Escape-to-dismiss for free, and dismissing leaves the
 * finished board on screen to look at, which is the third thing a player wants.
 */

import { useEffect, useRef, type ReactNode } from "react";

export interface GameOverDialogProps {
  open: boolean;
  /** "win", "loss" or "draw", from the player's side. */
  result: "win" | "loss" | "draw";
  black: number;
  white: number;
  /** Which colour the player was. */
  humanIsBlack: boolean;
  /** What to call the other side, so the result names a real opponent. */
  opponentName: string;
  children?: ReactNode;
  onRematch: () => void;
  onSwapSides: () => void;
  onReview: () => void;
}

export function GameOverDialog({
  open,
  result,
  black,
  white,
  humanIsBlack,
  opponentName,
  children,
  onRematch,
  onSwapSides,
  onReview,
}: GameOverDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (dialog === null) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const title = result === "win" ? "You win" : result === "loss" ? `${opponentName} wins` : "A draw";
  const margin = Math.abs(black - white);
  const detail =
    result === "draw"
      ? `${black} discs each.`
      : `${Math.max(black, white)} to ${Math.min(black, white)}, by ${margin} ${
          margin === 1 ? "disc" : "discs"
        }.`;

  return (
    <dialog
      ref={ref}
      onClose={onReview}
      aria-labelledby="game-over-title"
      className="game-over panel m-auto w-[min(92vw,27rem)] p-0 text-ink backdrop:bg-black/65"
    >
      <div className="p-7">
        <h2 id="game-over-title" className="display text-5xl">
          {title}
        </h2>
        <p className="mt-2 text-[0.95rem] text-muted">{detail}</p>

        <div className="mt-6 flex items-end justify-center gap-10">
          <Count colour="black" count={black} name={humanIsBlack ? "You" : opponentName} />
          <Count colour="white" count={white} name={humanIsBlack ? opponentName : "You"} />
        </div>

        {children && <div className="mt-6">{children}</div>}

        <div className="mt-7 flex flex-wrap items-center gap-2">
          <button type="button" onClick={onRematch} className="btn btn-primary">
            Rematch
          </button>
          <button type="button" onClick={onSwapSides} className="btn">
            Swap sides
          </button>
          <button type="button" onClick={onReview} className="btn">
            Look at the board
          </button>
          <a href="/" className="ml-auto text-sm text-muted underline-offset-2 hover:underline">
            All games
          </a>
        </div>
      </div>
    </dialog>
  );
}

function Count({
  colour,
  count,
  name,
}: {
  colour: "black" | "white";
  count: number;
  name: string;
}) {
  return (
    <div className="flex min-w-0 flex-col items-center gap-2">
      <span
        aria-hidden="true"
        className={`chip ${colour === "black" ? "chip-black" : "chip-white"} size-10`}
      />
      {/* Named for the tests: the dialog's heading contains the opponent's
          level number, so reading the score out of the dialog's text picks up a
          digit that is not part of it. */}
      <span data-count={colour} className="display text-5xl">
        {count}
      </span>
      <span className="max-w-[9rem] truncate text-sm text-muted">{name}</span>
    </div>
  );
}
