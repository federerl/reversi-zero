/**
 * The launcher.
 *
 * One game to play, one coming, and the research behind them folded away. The
 * board is the hero: a position from a real self-play game, lit from behind, big
 * enough to be the thing you look at first. Everything a visitor needs to decide
 * fits on one screen -- what it is, how strong it is, and the button that starts
 * it.
 *
 * The methodology that used to run down the page now sits in a disclosure beside
 * the game it explains. It is the honest part of the project and it belongs one
 * click away, not between a player and the board.
 */

import { Board } from "../games/reversi/ui/Board";
import { Shell } from "../shared/ui/Shell";
import { previewPosition } from "./preview";
import { GAMES, strongestElo, type GameEntry } from "./registry";

export function HubPage() {
  const reversi = GAMES.find((game) => game.id === "reversi");
  const soon = GAMES.filter((game) => game.status === "planned");

  return (
    <Shell>
      <div className="flex min-h-[calc(100dvh-8rem)] flex-col justify-center gap-14 py-6">
        {reversi && <Hero game={reversi} />}

        {soon.length > 0 && (
          <section aria-labelledby="soon" className="border-t border-line pt-6">
            <h2 id="soon" className="text-[0.95rem] text-muted">
              Coming soon
            </h2>
            <ul className="mt-3 flex flex-col gap-3">
              {soon.map((game) => (
                <li key={game.id} data-game={game.id} className="flex items-center gap-4">
                  <GomokuTile />
                  <div>
                    <p className="display text-xl text-ink-2">{game.title}</p>
                    <p className="text-[0.95rem] text-muted">{game.tagline}</p>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </Shell>
  );
}

function Hero({ game }: { game: GameEntry }) {
  const strongest = strongestElo(game);

  return (
    <section data-game={game.id} className="hero">
      <div className="hero-preview">
        <a href={game.path} aria-label={`Start a game of ${game.title}`} className="block">
          <Board state={previewPosition()} interactive={false} lastMove={null} onPlay={() => {}} />
        </a>
      </div>

      <div>
        <h2 className="display text-6xl leading-[0.85] sm:text-7xl">Reversi</h2>
        <p className="mt-4 max-w-[34ch] text-lg text-ink-2">
          Play against an AI that taught itself the game, right here in your browser.
        </p>

        <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-3">
          <a href={game.path} className="btn btn-primary btn-lg">
            Play Reversi
          </a>
          <p className="text-[0.95rem] text-muted">
            {game.opponents.length} opponents
            {strongest !== undefined && (
              <>
                , up to <span className="display text-xl text-ink-2">+{Math.round(strongest)}</span>
              </>
            )}
          </p>
        </div>

        <details className="disclosure mt-6 max-w-[46ch]">
          <summary>How the AI learned</summary>
          <div className="pb-2 text-[0.95rem] leading-relaxed text-muted">
            <p>
              Nobody taught it Reversi. It started from random weights and played itself sixty
              thousand times, keeping what worked: a small neural network guesses which moves look
              promising and who is winning, a search checks those guesses a few hundred positions
              deep, and the result of every finished game trains the network that plays the next
              one.
            </p>
            <p className="mt-3">
              Each opponent here is a checkpoint from that run, and the number beside it was
              measured, not chosen. Every agent played every other in a round robin, and the results
              were fitted into one rating scale anchored so that random play sits at 0. The interval
              beside a rating is where the true strength probably lies; neighbouring generations
              overlap, which is the honest way to say they are close.
            </p>
          </div>
        </details>
      </div>
    </section>
  );
}

/** A quiet stand-in for a game that has no board yet: the grid it will be played on. */
function GomokuTile() {
  const lines = Array.from({ length: 9 }, (_, i) => 8 + i * 10.5);
  return (
    <span
      aria-hidden="true"
      className="grid size-14 shrink-0 place-items-center rounded-md border border-line bg-surface"
    >
      <svg viewBox="0 0 100 100" className="size-9 opacity-45">
        {lines.map((p) => (
          <g key={p} stroke="var(--color-muted)" strokeWidth="2">
            <line x1={p} y1="8" x2={p} y2="92" />
            <line x1="8" y1={p} x2="92" y2={p} />
          </g>
        ))}
      </svg>
    </span>
  );
}
