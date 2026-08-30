/**
 * The whole application.
 *
 * The shape worth noticing: the agent's turn is driven by an effect that fires
 * whenever it becomes the agent's move, not by the click handler. A click makes
 * the player's move and nothing else. That keeps one rule -- "if it is the
 * agent's turn and nobody is thinking, think" -- in one place, and it means
 * passes chain correctly without a special case. If the agent passes and it is
 * still the agent's turn, the same effect simply runs again.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import modelsManifest from "../engine/models.json";
import { LocalEngine } from "../engine/local";
import type { ModelDescriptor } from "../engine/onnx";
import { isTerminal, legalActions, mustPass, passAction, type Action } from "../engine/rules";
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
import { Board, squareName } from "./Board";
import {
  Button,
  LevelPicker,
  OpponentPicker,
  Score,
  SidePicker,
  Status,
  Toast,
  WinProbability,
} from "./Panel";
import { BLACK } from "../engine/rules";

const MODELS = modelsManifest.models as unknown as ModelDescriptor[];
const DEFAULT_MODEL = MODELS[0]!.id;

export function App() {
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

    engine
      .think(state, game.levelId, controller.signal)
      .then((thought) => {
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
    dispatch({ type: "play", action });
  }, []);

  const stopThinking = useCallback(() => {
    searchRef.current?.abort();
    dispatch({ type: "thinking", value: false });
  }, []);

  const { black, white } = score(game);
  const turn = lastTurn(game);
  const humanMustPass = humanTurn && mustPass(state);

  const heatmap = useMemo(
    () => (game.showAnalysis ? turn.thought?.visits : undefined),
    [game.showAnalysis, turn.thought],
  );

  // The agent's estimate, read from the player's side of the board.
  const yourChances = turn.thought ? 1 - turn.thought.winProbability : null;

  return (
    <div className="mx-auto max-w-4xl px-4 pb-16 pt-8">
      <header className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight">reversi-zero</h1>
        <p className="mt-1 max-w-prose text-sm text-muted">
          An agent that learned Reversi from scratch by playing against itself. It runs entirely in
          your browser &mdash; nothing is sent anywhere.
        </p>
      </header>

      <div className="grid items-start gap-7 md:grid-cols-[minmax(0,1fr)_17rem]">
        <Board
          state={state}
          interactive={humanTurn && !game.thinking && !loading}
          lastMove={turn.move}
          visits={heatmap}
          onPlay={play}
        />

        <aside className="flex flex-col gap-4 rounded-md border border-line bg-surface p-4">
          <Score black={black} white={white} toMove={state.toMove} humanColor={game.humanColor} />

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

          {yourChances !== null && !terminal && <WinProbability probability={yourChances} />}

          <OpponentPicker
            models={MODELS}
            value={game.modelId}
            onChange={(id) => dispatch({ type: "setModel", modelId: id })}
            disabled={game.thinking}
          />

          <LevelPicker
            value={game.levelId}
            onChange={(id) => dispatch({ type: "setLevel", levelId: id })}
            disabled={game.thinking}
          />

          <SidePicker
            value={game.humanColor}
            onChange={(player) => dispatch({ type: "newGame", humanColor: player })}
            disabled={game.thinking}
          />

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

          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={game.showAnalysis}
              onChange={() => dispatch({ type: "toggleAnalysis" })}
              className="accent-accent-2"
            />
            Show where the agent searched
          </label>

          {turn.thought && (
            <p className="font-mono text-[0.7rem] leading-relaxed text-muted tabular-nums">
              played {squareName(turn.thought.action, state.size)} &middot;{" "}
              {turn.thought.simulations} sims &middot; {Math.round(turn.thought.elapsedMs)} ms
            </p>
          )}
        </aside>
      </div>

      <footer className="mt-8 max-w-prose text-xs leading-relaxed text-muted">
        Ratings come from a round robin of 210 games per entrant, fit with a Bradley&ndash;Terry
        model and anchored so that random play is 0. The intervals are 95% bootstrap intervals, and
        they overlap between neighbouring generations &mdash; which is the honest way to say that
        generation 40 and generation 60 are close.
      </footer>

      {game.error && (
        <Toast message={game.error} onDismiss={() => dispatch({ type: "error", message: null })} />
      )}
    </div>
  );
}

/** Exported for the tests: which squares the board should be offering. */
export function playableSquares(game: Game): Action[] {
  if (!isHumanTurn(game)) return [];
  const state = current(game);
  return legalActions(state).filter((action) => action !== passAction(state.size));
}
