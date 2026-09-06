/**
 * The front page: the games this site offers, one card each.
 *
 * A card shows what a visitor needs to choose: what the agent is, how many
 * opponents it offers, and how strong the strongest is on the same scale every
 * rating on this site uses. A game that is planned but not yet playable is shown
 * as exactly that, so the page reads as a collection from the first day without
 * pretending to more than it has.
 */

import { Shell } from "../shared/ui/Shell";
import { GAMES, strongestElo, type GameEntry } from "./registry";

export function HubPage() {
  return (
    <Shell
      lead={
        <>
          Board-game agents that learned by playing themselves, with a measured rating on every
          opponent. Everything runs in your browser &mdash; nothing is sent anywhere.
        </>
      }
      footer={
        <>
          Ratings are Bradley&ndash;Terry fits over round-robin tournaments, anchored so that random
          play is 0, with 95% bootstrap intervals. The numbers on each game&rsquo;s page come from
          the same tables.
        </>
      }
    >
      <ul className="grid gap-4 sm:grid-cols-2" aria-label="Games">
        {GAMES.map((game) => (
          <li key={game.id}>
            <GameCard game={game} />
          </li>
        ))}
      </ul>
    </Shell>
  );
}

function GameCard({ game }: { game: GameEntry }) {
  const playable = game.status === "playable";
  const strongest = strongestElo(game);

  return (
    <article
      data-game={game.id}
      className={[
        "flex h-full flex-col gap-3 rounded-md border border-line bg-surface p-5",
        playable ? "" : "opacity-70",
      ].join(" ")}
    >
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-xl font-semibold tracking-tight">{game.title}</h2>
        {!playable && (
          <span className="rounded-full border border-line px-2 py-0.5 text-[0.7rem] font-semibold uppercase tracking-wider text-muted">
            coming in 1.1
          </span>
        )}
      </div>

      <p className="text-sm leading-snug text-muted">{game.tagline}</p>

      {playable && (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs text-muted tabular-nums">
          <dt>opponents</dt>
          <dd className="text-ink">{game.opponents.length}</dd>
          {strongest !== undefined && (
            <>
              <dt>strongest</dt>
              <dd className="text-ink">+{Math.round(strongest)} Elo</dd>
            </>
          )}
        </dl>
      )}

      <div className="mt-auto pt-2">
        {playable ? (
          <a
            href={game.path}
            className="inline-block rounded bg-ink px-3 py-1.5 text-sm font-medium text-ground hover:opacity-90"
          >
            Play
          </a>
        ) : (
          <span className="text-xs text-muted">Not playable yet.</span>
        )}
      </div>
    </article>
  );
}
