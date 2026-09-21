/**
 * The puzzle page.
 *
 * You are handed an ending you are winning, and you play it out to the last
 * square against an opponent that cannot be improved on. Win and the puzzle is
 * solved; lose and it is not. The score you were promised either appears on the
 * board or it does not, and either way you can count it.
 *
 * That is a change from the first version, which graded one move and stopped.
 * Being told "+6" asks you to trust a number. This shows you one.
 *
 * **The page says nothing about the position while the ending is in progress.**
 * The exact value of every move is milliseconds away, and putting it on screen
 * would turn the page into a cheat sheet you could play by watching rather than
 * by calculating. The whole account arrives at the end, in a dialog that can
 * name the move that lost it.
 *
 * Nothing is locked. A gate would be the obvious reading of a staged
 * progression and the wrong call for a page strangers land on: hiding stage 4
 * behind twelve solved puzzles mostly stops a visitor seeing the interesting
 * ones. The order is carried by arrangement and labelling instead.
 */

import { useCallback, useEffect, useReducer, useRef } from "react";

import { Board, squareName } from "../ui/Board";
import { Button } from "../../../shared/ui/Button";
import { Shell } from "../../../shared/ui/Shell";
import { StatusPill } from "../ui/Panel";
import { safeStorage } from "../../../shared/theme";
import {
  BLACK,
  isTerminal,
  mustPass,
  passAction,
  type Action,
  type State,
} from "../engine/rules";
import { STAGES, puzzlesInStage, stageOf, type Puzzle } from "./data";
import { readSolved, writeSolved } from "./progress";
import { ResultDialog } from "./ResultDialog";
import { replyTo, valuesOf } from "./solver";
import {
  accepting,
  boardOf,
  currentPuzzle,
  describeOutcome,
  newSession,
  playerColour,
  playableSquares,
  reduce,
  solvedInStage,
  type Session,
  type SessionAction,
} from "../state/puzzle";

export function PuzzlesScreen() {
  const [session, dispatch] = useReducer(
    (state: Session, action: SessionAction) => reduce(state, action, puzzlesInStage(state.stage)),
    undefined,
    () => newSession(STAGES[0]?.number ?? 1, readSolved(safeStorage())),
  );

  useEffect(() => {
    writeSolved(safeStorage(), session.solvedIds);
  }, [session.solvedIds]);

  const stage = stageOf(session.stage);
  const inStage = puzzlesInStage(session.stage);
  const puzzle = currentPuzzle(session, inStage);

  const board = puzzle === null ? null : boardOf(session, puzzle);
  const reviewing = session.line !== null;
  const over = board !== null && isTerminal(board);
  const myTurn = board !== null && puzzle !== null && board.toMove === playerColour(puzzle);

  // Which position a search is out for. A ref rather than a flag in the
  // session, and this is load-bearing: the effect below dispatches `thinking`,
  // so if `session.thinking` were one of its dependencies React would tear the
  // effect down and its cleanup would mark the reply stale before it arrived.
  // The opponent would cancel itself on every move and sit there forever. This
  // project has shipped that bug once already, in the game page's own effect.
  const asked = useRef<string | null>(null);
  const positionKey = (state: State): string =>
    `${state.black.lo}:${state.black.hi}:${state.white.lo}:${state.white.hi}:${state.toMove}`;

  // The opponent answers, in a worker, whenever it is its turn. Also where a
  // forced pass is taken -- for either side -- so the board never sits waiting
  // on a move that does not exist.
  useEffect(() => {
    if (puzzle === null || board === null) return;
    if (reviewing || session.outcome !== null || over) return;

    if (myTurn) {
      if (mustPass(board)) dispatch({ type: "play", action: passAction(board.size) });
      return;
    }

    const key = positionKey(board);
    if (asked.current === key) return;
    asked.current = key;
    dispatch({ type: "thinking" });

    replyTo(board)
      .then((action) => {
        // Compared against the key rather than a closed-over flag, so a reply
        // is applied when it answers the position still on the board and
        // discarded when the page has moved on.
        if (asked.current === key) dispatch({ type: "opponentPlayed", action });
      })
      .catch((error: Error) => {
        if (asked.current === key) dispatch({ type: "failed", message: error.message });
      });
  }, [puzzle, board, myTurn, over, reviewing, session.outcome]);

  // Once the ending is over, work out the account: the exact value of every
  // position along the line, which is what lets the dialog name the move that
  // lost it rather than only reporting the score.
  const scored = useRef<string | null>(null);
  useEffect(() => {
    if (puzzle === null || board === null || !over || session.outcome !== null || reviewing) return;

    const key = `${puzzle.id}@${session.history.length}`;
    if (scored.current === key) return;
    scored.current = key;

    valuesOf(session.history.map((turn) => turn.state))
      .then((values) => {
        if (scored.current === key) {
          dispatch({ type: "finished", outcome: describeOutcome(session, puzzle, values) });
        }
      })
      .catch((error: Error) => {
        if (scored.current === key) dispatch({ type: "failed", message: error.message });
      });
  }, [puzzle, board, over, reviewing, session]);

  const goNext = useCallback(() => dispatch({ type: "next" }), []);
  const goRetry = useCallback(() => dispatch({ type: "retry" }), []);
  const showLine = useCallback(() => dispatch({ type: "showLine" }), []);

  if (puzzle === null || board === null) {
    return (
      <Shell breadcrumb="Endgame puzzles">
        <p className="panel p-4 text-ink-2">This stage has no puzzles in it yet.</p>
      </Shell>
    );
  }

  const line = session.line ?? 0;
  const lineDone = line >= puzzle.principalVariation.length;
  const lastTurn = session.history[session.history.length - 1];
  const lastMove: Action | null = reviewing
    ? (puzzle.principalVariation[line - 1] ?? null)
    : (lastTurn?.move ?? null);

  const status = describe(session, puzzle, { reviewing, line, myTurn, over });

  return (
    <Shell breadcrumb="Endgame puzzles" wide>
      <StagePicker session={session} dispatch={dispatch} />

      <div className="game-layout puzzles-layout">
        <div className="puzzle-prompt panel flex flex-col gap-3 p-4">
          <p className="text-sm text-muted">
            {stage.title} · puzzle {session.index + 1} of {inStage.length} · {puzzle.empties} empty
            squares at the start
          </p>
          <StatusPill
            headline={status.headline}
            detail={status.detail}
            tone={status.tone}
            thinking={session.thinking}
          />
          {session.error !== null && (
            <p role="alert" className="text-sm text-ink-2">
              {session.error}
            </p>
          )}
        </div>

        <div className="game-table">
          <Board
            state={board}
            interactive={accepting(session, puzzle) && playableSquares(board).length > 0}
            lastMove={lastMove}
            onPlay={(action) => dispatch({ type: "play", action })}
          />
        </div>

        <aside className="puzzle-panel panel flex flex-col gap-5 p-4">
          <div className="flex flex-wrap gap-2">
            {reviewing && !lineDone && (
              <Button variant="primary" onClick={() => dispatch({ type: "stepLine" })}>
                Next move
              </Button>
            )}
            {reviewing && (
              <Button onClick={() => dispatch({ type: "hideLine" })}>Back to the puzzle</Button>
            )}
            {!reviewing && <Button onClick={goRetry}>Start again</Button>}
            {!reviewing && session.outcome !== null && (
              <Button onClick={showLine}>Show the winning line</Button>
            )}
            {!reviewing && session.index < inStage.length - 1 && (
              <Button
                variant={session.outcome?.won === true ? "primary" : "secondary"}
                onClick={goNext}
              >
                Next puzzle
              </Button>
            )}
          </div>

          <p className="text-[0.95rem] leading-relaxed text-muted">{stage.teaches}</p>

          <MoveLog session={session} size={puzzle.state.size} />

          {session.outcome !== null && <MoveTable puzzle={puzzle} />}
        </aside>
      </div>

      <ResultDialog
        outcome={session.dismissed ? null : session.outcome}
        size={puzzle.state.size}
        hasNext={session.index < inStage.length - 1}
        onRetry={goRetry}
        onNext={goNext}
        onShowLine={showLine}
        onDismiss={() => dispatch({ type: "dismiss" })}
      />
    </Shell>
  );
}

function StagePicker({
  session,
  dispatch,
}: {
  session: Session;
  dispatch: (action: SessionAction) => void;
}) {
  return (
    <nav aria-label="Stages" className="mb-4 flex flex-wrap gap-2">
      {STAGES.map((stage) => {
        const inStage = puzzlesInStage(stage.number);
        const done = solvedInStage(session, inStage);
        const here = stage.number === session.stage;
        return (
          <button
            key={stage.number}
            type="button"
            aria-current={here ? "true" : undefined}
            // The full name is on the button where there is room and in the
            // accessible name always, so a narrow screen loses the width rather
            // than the meaning. Five buttons carrying "Stage 3 · Two regions at
            // once" wrap to five lines on a phone and push the board off it.
            aria-label={`Stage ${stage.number}, ${stage.title}, ${done} of ${inStage.length} won`}
            onClick={() => dispatch({ type: "pickStage", stage: stage.number })}
            className={`btn ${here ? "btn-primary" : ""}`}
          >
            <span aria-hidden="true">
              Stage {stage.number}
              <span className="hidden lg:inline"> · {stage.title}</span>
              {/* Dimmed by opacity rather than by the muted colour: the selected
                  stage sits on brass, where a grey meant for the page background
                  is close to unreadable. */}
              <span className="opacity-65">
                {" "}
                {done}/{inStage.length}
              </span>
            </span>
          </button>
        );
      })}
    </nav>
  );
}

/** The moves so far, so a player can see the ending they are in. */
function MoveLog({ session, size }: { session: Session; size: number }) {
  const played = session.history.filter((turn) => turn.move !== null);
  if (played.length === 0) return null;

  return (
    <ol className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
      {played.map((turn, index) => (
        <li key={index} className={turn.mine ? "text-ink-2" : "text-muted"}>
          <span className="tabular-nums">{squareName(turn.move!, size)}</span>
        </li>
      ))}
    </ol>
  );
}

/**
 * Every legal move at the *starting* position, and where it led.
 *
 * Only after the ending is over. Shown while it was in progress this would be
 * the answer key.
 */
function MoveTable({ puzzle }: { puzzle: Puzzle }) {
  const size = puzzle.state.size;
  const ranked = [...puzzle.margins].sort((a, b) => b[1] - a[1] || a[0] - b[0]);

  return (
    <details className="disclosure">
      <summary>Every opening move, and where it leads</summary>
      <div className="flex flex-col gap-3 pb-1 text-sm text-muted">
        <ul className="flex flex-col gap-1">
          {ranked.map(([move, margin]) => (
            <li key={move} className="flex items-baseline gap-2">
              <span className="w-8 shrink-0 font-medium text-ink-2">{squareName(move, size)}</span>
              <span className="w-24 shrink-0 tabular-nums">
                {margin > 0 ? `wins by ${margin}` : margin < 0 ? `loses by ${-margin}` : "draws"}
              </span>
              <span className="min-w-0">
                {[
                  margin === puzzle.best ? "best" : "",
                  move === puzzle.tempting ? "the move the network likes" : "",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </li>
          ))}
        </ul>
        <p className="leading-relaxed">
          These are not estimates. With {puzzle.empties} squares empty the position is small
          enough to search to the last move, so each number is the disc difference the game
          really ends on when both sides play perfectly.
          {puzzle.greedyFalls && " Here, taking the most discs available loses."}
        </p>
      </div>
    </details>
  );
}

function describe(
  session: Session,
  puzzle: Puzzle,
  where: { reviewing: boolean; line: number; myTurn: boolean; over: boolean },
): { headline: string; detail: string; tone: "you" | "quiet" } {
  const size = puzzle.state.size;
  const mover = playerColour(puzzle) === BLACK ? "Black" : "White";

  if (where.reviewing) {
    const total = puzzle.principalVariation.length;
    const shown = puzzle.principalVariation
      .slice(0, where.line)
      .map((action) => (action === passAction(size) ? "pass" : squareName(action, size)))
      .join(" ");
    return {
      headline: where.line === 0 ? "Perfect play from the start" : shown,
      detail:
        where.line >= total
          ? `That is the whole ending. ${mover} wins by ${puzzle.best}.`
          : `${where.line} of ${total} moves.`,
      tone: "quiet",
    };
  }

  if (session.outcome !== null) {
    const { discs, won } = session.outcome;
    return {
      headline: won ? `You won, ${discs.mine}–${discs.theirs}.` : `You lost, ${discs.mine}–${discs.theirs}.`,
      detail: "The board is finished. Start again, or look at the winning line.",
      tone: won ? "you" : "quiet",
    };
  }

  if (session.thinking) {
    return { headline: "Thinking…", detail: "It is solving the ending exactly.", tone: "quiet" };
  }

  if (session.history.length <= 1) {
    return {
      headline: `You are ${mover.toLowerCase()}, and winning`,
      detail: `Play it out and win it. ${puzzle.empties} squares left, and the opponent will not slip.`,
      tone: "you",
    };
  }

  return {
    headline: where.myTurn ? "Your move" : "Its move",
    detail: `${puzzle.empties - session.history.length + 1} squares left.`,
    tone: where.myTurn ? "you" : "quiet",
  };
}
