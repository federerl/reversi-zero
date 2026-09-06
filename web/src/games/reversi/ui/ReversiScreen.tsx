/**
 * The Reversi screen: the board with a player on each side of it, the controls
 * in a column beside it, and the agent's turn.
 *
 * The shape worth noticing: the agent's turn is driven by an effect that fires
 * whenever it becomes the agent's move, not by the click handler. A click makes
 * the player's move and nothing else. That keeps one rule -- "if it is the
 * agent's turn and nobody is thinking, think" -- in one place, and it means
 * passes chain correctly without a special case. If the agent passes and it is
 * still the agent's turn, the same effect simply runs again.
 *
 * The layout is the one every serious play site converges on: a thin bar, a
 * board sized to the window's height, a player plate above and below it, and
 * the controls in a column that is always on screen. Nothing needs scrolling on
 * a laptop, and nothing competes with a disc turning over.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { BASELINES, isBaseline } from "../engine/baselines";
import { LocalEngine } from "../engine/local";
import modelsManifest from "../engine/models.json";
import type { ModelDescriptor } from "../engine/onnx";
import {
  BLACK,
  WHITE,
  isTerminal,
  legalActions,
  mustPass,
  passAction,
  winner,
  type Action,
} from "../engine/rules";
import type { Engine } from "../engine/types";
import {
  current,
  isHumanTurn,
  lastTurn,
  newGame,
  reduce,
  score,
  statusLine,
  type Game,
} from "../state/game";
import { sounds } from "../../../shared/sound";
import { GameOverDialog } from "../../../shared/ui/GameOverDialog";
import { Shell } from "../../../shared/ui/Shell";
import { Board, FLIP_STAGGER_MS, squareName } from "./Board";
import {
  Button,
  LevelPicker,
  OpponentPicker,
  PlayerPlate,
  SidePicker,
  Status,
  Toast,
  WinProbability,
} from "./Panel";

interface RatedBaseline {
  readonly name: string;
  readonly elo: number;
  readonly interval: [number, number];
}

const NETWORKS = modelsManifest.models as unknown as ModelDescriptor[];
const RATED = modelsManifest.baselines as unknown as RatedBaseline[];

/**
 * Every opponent, weakest first.
 *
 * The baselines come first deliberately. The network is strong even at its
 * earliest checkpoint -- generation 5 rates above the depth-4 search it was
 * measured against -- so before these were offered there was nothing on the
 * ladder a new player could actually beat, and no simulation budget makes a
 * +547 network into a beginner's opponent.
 *
 * The order is by measured rating, not by what feels like it should be easier.
 */
const OPPONENTS: readonly ModelDescriptor[] = [
  ...BASELINES.map((baseline) => {
    const rated = RATED.find((entry) => entry.name === baseline.ratingName);
    return {
      id: baseline.id,
      label: baseline.label,
      generation: -1,
      url: "",
      boardSize: 8,
      ...(rated ? { elo: rated.elo, eloInterval: rated.interval } : {}),
      note: baseline.note,
    } satisfies ModelDescriptor;
  }),
  ...[...NETWORKS].sort((a, b) => a.generation - b.generation),
];

/**
 * Start on Greedy.
 *
 * A first game you lose is a bad demonstration, and every network on this ladder
 * beats a casual player comfortably -- the weakest of them rates above the
 * depth-4 search it was measured against. Greedy is beatable, it is a measured
 * baseline rather than a hobbled agent, and the ladder above it is visible in
 * the picker from the first second.
 *
 * It also means the page is playable immediately: no network is fetched until
 * somebody chooses one.
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
  const opponent = OPPONENTS.find((model) => model.id === game.modelId);
  const opponentName =
    opponent === undefined
      ? "Agent"
      : opponent.elo === undefined
        ? opponent.label
        : `${opponent.label}, ${opponent.elo > 0 ? "+" : ""}${Math.round(opponent.elo)} Elo`;

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
  const topColour = humanIsBlack ? WHITE : BLACK;
  const plate = (colour: typeof BLACK | typeof WHITE) => (
    <PlayerPlate
      colour={colour === BLACK ? "black" : "white"}
      name={colour === game.humanColor ? "You" : opponentName}
      count={colour === BLACK ? black : white}
      active={!terminal && state.toMove === colour}
    />
  );

  return (
    <Shell title="Reversi" wide>
      <div className="game-layout">
        <aside className="game-aside-left flex flex-col gap-4 text-[0.95rem] leading-relaxed text-muted">
          <p>
            An agent that learned Reversi from scratch by playing against itself. It runs entirely
            in your browser and nothing is sent anywhere.
          </p>
          <p>
            Ratings come from a round robin of 210 games per entrant, fit with a Bradley&ndash;Terry
            model and anchored so that random play is 0. The intervals are 95% bootstrap intervals,
            and they overlap between neighbouring generations, which is the honest way to say that
            generation 40 and generation 60 are close.
          </p>
        </aside>

        <div className="game-board-column flex flex-col gap-2">
          {plate(topColour)}
          <Board
            state={state}
            interactive={humanTurn && !game.thinking && !loading}
            lastMove={turn.move}
            visits={heatmap}
            onPlay={play}
          />
          {plate(game.humanColor)}
        </div>

        <aside className="game-aside-right flex flex-col gap-5">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <Status
                line={loading ? "Loading the agent…" : statusLine(game)}
                thinking={game.thinking || loading}
              />
              {humanMustPass && (
                <Button variant="primary" onClick={() => play(passAction(state.size))}>
                  Pass
                </Button>
              )}
              {game.thinking && <Button onClick={stopThinking}>Stop thinking</Button>}
            </div>
            {yourChances !== null && !terminal && <WinProbability probability={yourChances} />}
            {turn.thought && (
              <p className="text-sm text-muted">
                Played {squareName(turn.thought.action, state.size)}
                {turn.thought.simulations > 0 && <> after {turn.thought.simulations} simulations</>}{" "}
                in {Math.round(turn.thought.elapsedMs)} ms
              </p>
            )}
          </div>

          <div className="flex flex-col gap-3 border-t border-line pt-4">
            <OpponentPicker
              models={OPPONENTS}
              value={game.modelId}
              onChange={(id) => dispatch({ type: "setModel", modelId: id })}
              disabled={game.thinking}
            />

            {!isBaseline(game.modelId) && (
              <LevelPicker
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

          <div className="flex flex-col gap-3 border-t border-line pt-4">
            <div className="flex flex-wrap items-center gap-2">
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
            <label className="flex cursor-pointer items-center gap-2 text-sm text-muted">
              <input
                type="checkbox"
                checked={game.showAnalysis}
                onChange={() => dispatch({ type: "toggleAnalysis" })}
                className="accent-accent"
              />
              Show where the agent searched
            </label>
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
          onPlayAgain={() => {
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

/** Exported for the tests: which squares the board should be offering. */
export function playableSquares(game: Game): Action[] {
  if (!isHumanTurn(game)) return [];
  const state = current(game);
  return legalActions(state).filter((action) => action !== passAction(state.size));
}
