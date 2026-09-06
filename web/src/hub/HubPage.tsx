/**
 * The front page: the games this site offers.
 *
 * The board is the entrance. Reversi is shown as its own board in the opening
 * position, large, with one line about what the agent is and the two numbers a
 * visitor needs to choose: how many opponents, and how strong the strongest is
 * on the scale every rating on this site uses. A game that is planned but not
 * yet playable is drawn as an empty board, dimmed, and says so.
 */

import { initialState } from "../games/reversi/engine/rules";
import { Board } from "../games/reversi/ui/Board";
import { Shell } from "../shared/ui/Shell";
import { GAMES, strongestElo, type GameEntry } from "./registry";

export function HubPage() {
  return (
    <Shell>
      <p className="mb-8 max-w-[60ch] text-[0.95rem] text-muted">
        Board-game agents that learned by playing themselves, each opponent with a measured rating.
        Everything runs in your browser and nothing is sent anywhere.
      </p>

      <ul className="flex flex-col gap-12" aria-label="Games">
        {GAMES.map((game) => (
          <li key={game.id}>
            <GameEntry game={game} />
          </li>
        ))}
      </ul>

      <p className="mt-12 max-w-[68ch] text-sm leading-relaxed text-muted">
        Ratings are Bradley&ndash;Terry fits over round-robin tournaments, anchored so that random
        play is 0, with 95% bootstrap intervals. The numbers on each game&rsquo;s page come from the
        same tables.
      </p>
    </Shell>
  );
}

function GameEntry({ game }: { game: GameEntry }) {
  const playable = game.status === "playable";
  const strongest = strongestElo(game);

  return (
    <article
      data-game={game.id}
      className={`grid items-center gap-6 ${
        playable
          ? "sm:grid-cols-[minmax(0,20rem)_1fr]"
          : "opacity-60 sm:grid-cols-[minmax(0,11rem)_1fr]"
      }`}
    >
      {playable ? (
        <a href={game.path} aria-label={`Play ${game.title}`} className="block">
          <Preview game={game} />
        </a>
      ) : (
        <Preview game={game} />
      )}

      <div className="max-w-[46ch]">
        <h2
          className={`font-display font-bold leading-none tracking-tight ${playable ? "text-4xl" : "text-3xl"}`}
        >
          {game.title}
        </h2>
        <p className="mt-2 text-[0.95rem] text-muted">{game.tagline}</p>

        {playable ? (
          <>
            <p className="mt-3 text-[0.95rem] text-ink-2">
              {game.opponents.length} opponents
              {strongest !== undefined && (
                <>
                  , the strongest rated{" "}
                  <span className="score-number text-xl">+{Math.round(strongest)} Elo</span>
                </>
              )}
              .
            </p>
            <a
              href={game.path}
              className="mt-5 inline-block rounded-md bg-ink px-4 py-2 text-[0.95rem] font-medium text-ground hover:opacity-90"
            >
              Play
            </a>
          </>
        ) : (
          <p className="mt-3 text-[0.95rem] text-muted">Next. Not playable yet.</p>
        )}
      </div>
    </article>
  );
}

function Preview({ game }: { game: GameEntry }) {
  if (game.id === "reversi") {
    return (
      <div aria-hidden="true" className="pointer-events-none select-none">
        <Board state={initialState(8)} interactive={false} lastMove={null} onPlay={() => {}} />
      </div>
    );
  }
  // An empty board for a game that is not here yet: a 15-line grid, as Gomoku is played.
  const lines = Array.from({ length: 15 }, (_, i) => 5 + i * (90 / 14));
  return (
    <div
      className="board-frame"
      aria-hidden="true"
      style={{ gridTemplateColumns: "1fr", padding: "0.35rem" }}
    >
      <svg viewBox="0 0 100 100" className="block aspect-square w-full rounded-sm bg-board">
        {lines.map((p) => (
          <g key={p} stroke="rgba(16,17,18,0.45)" strokeWidth="0.35">
            <line x1={p} y1="5" x2={p} y2="95" />
            <line x1="5" y1={p} x2="95" y2={p} />
          </g>
        ))}
        {[3, 7, 11].flatMap((x) =>
          [3, 7, 11].map((y) => (
            <circle key={`${x}-${y}`} cx={lines[x]} cy={lines[y]} r="1" fill="rgba(16,17,18,0.6)" />
          )),
        )}
      </svg>
    </div>
  );
}
