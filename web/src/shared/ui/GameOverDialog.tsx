/**
 * The end of a game, said properly.
 *
 * A finished game used to be one sentence in the status box. This is a real
 * ending: who won and by how much, the final count with the discs drawn, and
 * the things a player does next. The native `<dialog>` element gives focus
 * trapping and Escape-to-dismiss for free, and dismissing leaves the finished
 * board on screen to look at.
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
  /** Anything else worth showing under the count: a chart, a record line. */
  children?: ReactNode;
  onPlayAgain: () => void;
  onSwapSides: () => void;
  onReview: () => void;
}

export function GameOverDialog({
  open,
  result,
  black,
  white,
  humanIsBlack,
  children,
  onPlayAgain,
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

  const title = result === "win" ? "You win" : result === "loss" ? "The agent wins" : "A draw";
  const margin = Math.abs(black - white);
  const detail =
    result === "draw"
      ? `${black} discs each.`
      : `${Math.max(black, white)} to ${Math.min(black, white)}, by ${margin} ${margin === 1 ? "disc" : "discs"}.`;

  return (
    <dialog
      ref={ref}
      onClose={onReview}
      aria-labelledby="game-over-title"
      className="game-over m-auto w-[min(92vw,26rem)] rounded-lg border border-line bg-surface p-0 text-ink shadow-2xl backdrop:bg-black/60"
    >
      <div className="p-7">
        <h2 id="game-over-title" className="font-display text-5xl font-bold leading-none tracking-tight">
          {title}
        </h2>
        <p className="mt-2 text-[0.95rem] text-muted">{detail}</p>

        <div className="mt-6 flex items-end justify-center gap-10">
          <Count colour="black" count={black} you={humanIsBlack} />
          <Count colour="white" count={white} you={!humanIsBlack} />
        </div>

        {children && <div className="mt-6">{children}</div>}

        <div className="mt-7 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onPlayAgain}
            className="rounded-md bg-ink px-3.5 py-1.5 text-[0.95rem] font-medium text-ground hover:opacity-90"
          >
            Play again
          </button>
          <button
            type="button"
            onClick={onSwapSides}
            className="rounded-md border border-line-strong bg-surface px-3.5 py-1.5 text-[0.95rem] font-medium text-ink hover:bg-surface-2"
          >
            Swap sides
          </button>
          <button
            type="button"
            onClick={onReview}
            className="rounded-md border border-line-strong bg-surface px-3.5 py-1.5 text-[0.95rem] font-medium text-ink hover:bg-surface-2"
          >
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

function Count({ colour, count, you }: { colour: "black" | "white"; count: number; you: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <span
        aria-hidden="true"
        className={`size-9 rounded-full shadow ${
          colour === "black" ? "bg-disc-black ring-1 ring-line-strong" : "bg-disc-white"
        }`}
      />
      <span className="score-number text-5xl">{count}</span>
      <span className="text-sm text-muted">{you ? "You" : "Agent"}</span>
    </div>
  );
}
