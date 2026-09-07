/**
 * The Reversi screen: a table with a board on it, and one panel of controls.
 *
 * The shape worth noticing: the agent's turn is driven by an effect that fires
 * whenever it becomes the agent's move, not by the click handler. A click makes
 * the player's move and nothing else. That keeps one rule -- "if it is the
 * agent's turn and nobody is thinking, think" -- in one place, and it means
 * passes chain correctly without a special case. If the agent passes and it is
 * still the agent's turn, the same effect simply runs again.
 *
 * The layout has two regions rather than three columns. The table is the board
 * with a player plate above and below it, so a disc count belongs to a face
 * instead of floating at a corner. The panel runs the height of the board and
 * holds what a player acts on, in the order they need it: what is happening, who
 * they are playing, and what they can do about it. Everything fits a laptop
 * window without scrolling.
 *
 * The opponent is named by its rung on the ladder -- "Level 4" -- everywhere a
 * player reads it: the plate, the turn indicator, the game-over dialog. Which
 * checkpoint that is, and what it measured, sits under a disclosure in the panel.
 * See `../ladder.ts` for why the ladder is derived from the ratings rather than
 * written down.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { LocalEngine } from "../engine/local";
import {
  BLACK,
  WHITE,
  isTerminal,
  legalActions,
  mustPass,
  passAction,
  winner,
  type Action,
  type Player,
} from "../engine/rules";
import type { Engine } from "../engine/types";
import { rungFor, rungName } from "../ladder";
import {
  current,
  isHumanTurn,
  lastTurn,
  newGame,
  reduce,
  score,
  type Game,
} from "../state/game";
import { sounds } from "../../../shared/sound";
import { Button, Toast, Toggle } from "../../../shared/ui/Button";
import { GameOverDialog } from "../../../shared/ui/GameOverDialog";
import { Shell } from "../../../shared/ui/Shell";
import { Board, FLIP_STAGGER_MS, squareName } from "./Board";
import {
  LevelDetails,
  LevelPicker,
  PlayerPlate,
  SidePicker,
  StatusPill,
  ThinkingTimePicker,
  WinProbability,
} from "./Panel";

/**
 * Start on level 2.
 *
 * A first game you lose is a bad demonstration, and every network on the ladder
 * beats a casual player comfortably -- the weakest checkpoint rates above the
 * depth-4 search it was measured against. Level 2 is the disc-counting baseline:
 * beatable, still a measured opponent rather than a hobbled agent, and the rest
 * of the ladder is one select away from the first second.
 *
 * It also means the page is playable immediately, because a baseline needs no
 * download -- no network is fetched until somebody picks a level that uses one.
 */
const DEFAULT_MODEL = "greedy";

/**
 * The soonest the agent is allowed to answer, in milliseconds.
 *
 * At the quicker levels a search finishes in under 300 ms, which is faster than
 * a person can follow. You click, your discs start turning over, and before that
 * finishes the agent has moved and turned some of them back -- so the move you
 * just made is never actually visible, and the board appears to change by
 * itself.
 *
 * Waiting out the difference costs nothing: the search has already produced its
 * answer, and this only delays showing it. Levels that genuinely take longer
 * than this are unaffected, so the pause exists exactly where it is needed and
 * nowhere else.
 */
const MIN_REPLY_MS = 650;

export function ReversiScreen() {
  const [game, dispatch] = useReducer(reduce, newGame(BLACK, "club", DEFAULT_MODEL));
  const [loading, setLoading] = useState(true);

  // One engine for the life of the page. Recreating it per move would reload
  // the network every time, which is the single most expensive thing here.
  const engineRef = useRef<Engine | null>(null);
  const searchRef = useRef<AbortController | null>(null);

  if (engineRef.current === null) engineRef.current = new LocalEngine(DEFAULT_MODEL);

  useEffect(() => {
    const engine = engineRef.current;
    return () => engine?.dispose();
  }, []);

  // Load the chosen opponent. Switching generation is a download the first
  // time and free after, because the browser caches it.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    engineRef.current
      ?.useModel(game.modelId)
      .then(() => {
        if (!cancelled) setLoading(false);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setLoading(false);
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : "could not load the agent",
        });
      });

    return () => {
      cancelled = true;
    };
  }, [game.modelId]);

  const state = current(game);
  const terminal = isTerminal(state);
  const humanTurn = isHumanTurn(game);

  // The agent's turn. One rule, one place: if it is the agent's move and
  // nothing is in flight, think.
  //
  // `game.thinking` is deliberately *not* a dependency, and the reason is worth
  // writing down because the bug it causes looks like a hung search rather than
  // a mistake in a dependency list. The effect's first act is to set `thinking`.
  // If it also depended on it, React would tear the effect down and run its
  // cleanup -- aborting the search that had just started -- then re-run the body,
  // which would bail out because `thinking` was now true. The agent would sit at
  // "Thinking…" forever, having cancelled itself.
  //
  // What the effect keys on instead is the thing that genuinely identifies a
  // search: this position, this opponent, this budget.
  const ply = game.history.length;
  useEffect(() => {
    if (loading || terminal || humanTurn) return;

    const engine = engineRef.current;
    if (engine === null) return;

    const controller = new AbortController();
    searchRef.current = controller;
    dispatch({ type: "thinking", value: true });

    const askedAt = performance.now();

    engine
      .think(state, game.levelId, controller.signal)
      .then(async (thought) => {
        // Let the player's own move finish landing before answering it.
        const remaining = MIN_REPLY_MS - (performance.now() - askedAt);
        if (remaining > 0) {
          await new Promise((resolve) => setTimeout(resolve, remaining));
        }
        if (controller.signal.aborted) return;
        dispatch({ type: "agentPlayed", action: thought.action, thought });
        dispatch({ type: "thinking", value: false });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        dispatch({
          type: "error",
          message: error instanceof Error ? error.message : "the agent could not move",
        });
      });

    // Abandon the search if the position changes underneath it -- a take-back,
    // a new game, or a different opponent. Finishing it would play a move into
    // a game that no longer exists.
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, terminal, humanTurn, ply, game.levelId, game.modelId]);

  const play = useCallback((action: Action) => {
    sounds.place();
    dispatch({ type: "play", action });
  }, []);

  // The agent's disc makes the same sound as yours, and the flips tick outward
  // in step with the animation. The agent's move is the only one that arrives
  // here after the fact, so its sound plays when the history grows by its move
  // rather than when it was decided.
  const lastMover = lastTurn(game).thought;
  useEffect(() => {
    if (lastMover === null || ply <= 1) return;
    sounds.place();
    sounds.flip(3, FLIP_STAGGER_MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ply]);

  // The ending, once, when the game reaches it.
  const outcome = terminal ? winner(state) : null;
  const playerResult: "win" | "loss" | "draw" | null =
    outcome === null
      ? null
      : outcome === "draw"
        ? "draw"
        : (outcome === "black") === (game.humanColor === BLACK)
          ? "win"
          : "loss";
  const [dismissedAt, setDismissedAt] = useState<number | null>(null);
  useEffect(() => {
    if (playerResult !== null) sounds.gameOver(playerResult);
  }, [playerResult]);

  const stopThinking = useCallback(() => {
    searchRef.current?.abort();
    dispatch({ type: "thinking", value: false });
  }, []);

  const { black, white } = score(game);
  const turn = lastTurn(game);
  const humanMustPass = humanTurn && mustPass(state);
  const rung = rungFor(game.modelId);
  const levelName = rungName(rung);
  const playsNetwork = rung.usesNetwork;

  const heatmap = useMemo(
    () => (game.showAnalysis ? turn.thought?.visits : undefined),
    [game.showAnalysis, turn.thought],
  );

  // The agent's estimate, read from the player's side of the board. Random and
  // Greedy hold no opinion about who is winning, so there is nothing to show.
  const agentChances = turn.thought?.winProbability;
  const yourChances = agentChances === undefined ? null : 1 - agentChances;

  // Whose plate goes above the board: the opponent's, as across a real table.
  const humanIsBlack = game.humanColor === BLACK;
  const plateFor = (colour: Player) => {
    const isHuman = colour === game.humanColor;
    return (
      <PlayerPlate
        colour={colour === BLACK ? "black" : "white"}
        name={isHuman ? "You" : levelName}
        detail={isHuman ? undefined : rung.word}
        count={colour === BLACK ? black : white}
        active={!terminal && state.toMove === colour}
        thinking={!isHuman && (game.thinking || loading)}
      />
    );
  };

  const status = describeTurn({
    loading,
    thinking: game.thinking,
    humanTurn,
    humanMustPass,
    levelName,
    result: playerResult,
    black,
    white,
  });

  return (
    <Shell breadcrumb="Reversi" wide>
      <div className="game-layout">
        <div className="game-table">
          {plateFor(humanIsBlack ? WHITE : BLACK)}
          <Board
            state={state}
            interactive={humanTurn && !game.thinking && !loading}
            lastMove={turn.move}
            visits={heatmap}
            onPlay={play}
          />
          {plateFor(game.humanColor)}
        </div>

        <aside className="panel flex flex-col gap-5 p-4">
          <div className="flex flex-col gap-3">
            <StatusPill
              headline={status.headline}
              detail={status.detail}
              tone={status.tone}
              thinking={game.thinking || loading}
            />

            {(humanMustPass || game.thinking) && (
              <div className="flex flex-wrap gap-2">
                {humanMustPass && (
                  <Button variant="primary" onClick={() => play(passAction(state.size))}>
                    Pass
                  </Button>
                )}
                {game.thinking && <Button onClick={stopThinking}>Stop thinking</Button>}
              </div>
            )}

            {yourChances !== null && !terminal && <WinProbability probability={yourChances} />}
          </div>

          <div className="flex flex-col gap-2">
            <LevelPicker
              value={game.modelId}
              onChange={(id) => dispatch({ type: "setModel", modelId: id })}
              disabled={game.thinking}
            />
            {playsNetwork && (
              <ThinkingTimePicker
                value={game.levelId}
                onChange={(id) => dispatch({ type: "setLevel", levelId: id })}
                disabled={game.thinking}
              />
            )}
            <SidePicker
              value={game.humanColor}
              onChange={(player) => dispatch({ type: "newGame", humanColor: player })}
              disabled={game.thinking}
            />
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              <Button variant="primary" onClick={() => dispatch({ type: "newGame" })}>
                New game
              </Button>
              <Button onClick={() => dispatch({ type: "swapSides" })}>Swap sides</Button>
              <Button
                onClick={() => dispatch({ type: "undo" })}
                disabled={game.history.length <= 1 || game.thinking}
              >
                Take back
              </Button>
            </div>
            <Toggle
              checked={game.showAnalysis}
              onChange={() => dispatch({ type: "toggleAnalysis" })}
            >
              Show where the AI searched
            </Toggle>
          </div>

          <div className="flex flex-col gap-1 border-t border-line pt-2">
            {turn.thought && (
              <p className="text-sm text-muted">
                Played {squareName(turn.thought.action, state.size)}
                {turn.thought.simulations > 0 && <> after {turn.thought.simulations} simulations</>}{" "}
                in {Math.round(turn.thought.elapsedMs)} ms
              </p>
            )}
            <LevelDetails rung={rung} levelId={game.levelId} />
          </div>
        </aside>
      </div>

      {game.error && (
        <Toast message={game.error} onDismiss={() => dispatch({ type: "error", message: null })} />
      )}

      {playerResult !== null && (
        <GameOverDialog
          open={dismissedAt !== ply}
          result={playerResult}
          black={black}
          white={white}
          humanIsBlack={humanIsBlack}
          opponentName={levelName}
          onRematch={() => {
            setDismissedAt(null);
            dispatch({ type: "newGame" });
          }}
          onSwapSides={() => {
            setDismissedAt(null);
            dispatch({ type: "swapSides" });
          }}
          onReview={() => setDismissedAt(ply)}
        />
      )}
    </Shell>
  );
}

/**
 * What to put in the turn indicator.
 *
 * The headline is the state in two or three words, because that is what a player
 * checks between moves. At the end it is the result, worded exactly as the
 * dialog words it: the interface must not say "the agent wins" in one place and
 * "Level 2 wins" in another about the same game.
 *
 * This is also the live region, so a screen reader hears the whole sentence.
 */
function describeTurn({
  loading,
  thinking,
  humanTurn,
  humanMustPass,
  levelName,
  result,
  black,
  white,
}: {
  loading: boolean;
  thinking: boolean;
  humanTurn: boolean;
  humanMustPass: boolean;
  /** How the opponent is named to the player: "Level 4". */
  levelName: string;
  result: "win" | "loss" | "draw" | null;
  black: number;
  white: number;
}): { headline: string; detail?: string; tone: "you" | "quiet" } {
  if (loading) return { headline: "Loading the AI…", tone: "quiet" };

  if (result !== null) {
    const high = Math.max(black, white);
    const low = Math.min(black, white);
    if (result === "draw") {
      return { headline: "A draw", detail: `${black} discs each.`, tone: "quiet" };
    }
    return {
      headline: result === "win" ? "You win" : `${levelName} wins`,
      detail: `${high} to ${low}.`,
      tone: result === "win" ? "you" : "quiet",
    };
  }

  if (thinking) return { headline: `${levelName} is thinking…`, tone: "quiet" };
  if (humanMustPass) return { headline: "You must pass", tone: "you" };
  if (humanTurn) return { headline: "Your turn", tone: "you" };
  return { headline: `${levelName} to move`, tone: "quiet" };
}

/** Exported for the tests: which squares the board should be offering. */
export function playableSquares(game: Game): Action[] {
  if (!isHumanTurn(game)) return [];
  const state = current(game);
  return legalActions(state).filter((action) => action !== passAction(state.size));
}
