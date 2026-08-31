/**
 * The opponent, running on the visitor's own machine.
 *
 * This is a thin wrapper that turns the worker's message protocol back into
 * ordinary promises, so the rest of the app can `await engine.think(...)` and
 * never think about workers at all.
 *
 * The only real logic here is matching replies to requests. Messages arrive in
 * whatever order they finish, and a player who changes their mind twice while
 * the agent is thinking would otherwise get an answer to a question they have
 * already withdrawn -- and it would look like the board playing its own moves.
 */

import type { State } from "./rules";
import { EngineError, type Engine, type Thought, type WorkerRequest, type WorkerResponse } from "./types";

interface Pending {
  resolve: (thought: Thought) => void;
  reject: (error: Error) => void;
}

export class LocalEngine implements Engine {
  readonly kind = "local" as const;

  private readonly worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private readonly loads = new Map<number, { resolve: () => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private loading: Promise<void> | null = null;

  constructor(private modelId: string) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.receive(event.data);
    this.worker.onerror = (event) => this.failEverything(new EngineError(event.message));
  }

  ready(): Promise<void> {
    this.loading ??= this.load(this.modelId);
    return this.loading;
  }

  async useModel(modelId: string): Promise<void> {
    if (modelId === this.modelId && this.loading !== null) return this.loading;
    this.modelId = modelId;
    this.loading = this.load(modelId);
    return this.loading;
  }

  think(state: State, levelId: string, signal?: AbortSignal): Promise<Thought> {
    return this.ready().then(
      () =>
        new Promise<Thought>((resolve, reject) => {
          const id = this.nextId++;
          this.pending.set(id, { resolve, reject });

          const abort = () => {
            this.send({ type: "cancel", id });
            this.pending.delete(id);
            reject(new DOMException("the search was cancelled", "AbortError"));
          };

          if (signal?.aborted) return abort();
          signal?.addEventListener("abort", abort, { once: true });

          this.send({ type: "think", id, state, levelId, opponentId: this.modelId });
        }),
    );
  }

  dispose(): void {
    this.failEverything(new EngineError("the engine was shut down"));
    this.worker.terminate();
  }

  // -----------------------------------------------------------------

  private load(modelId: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const id = this.nextId++;
      this.loads.set(id, { resolve, reject });
      this.send({ type: "load", id, modelId });
    });
  }

  private send(request: WorkerRequest): void {
    this.worker.postMessage(request);
  }

  private receive(response: WorkerResponse): void {
    if (response.type === "loaded") {
      this.loads.get(response.id)?.resolve();
      this.loads.delete(response.id);
      return;
    }

    if (response.type === "thought") {
      this.pending.get(response.id)?.resolve(response.thought);
      this.pending.delete(response.id);
      return;
    }

    if (response.type === "cancelled") {
      this.pending.delete(response.id);
      return;
    }

    const error = new EngineError(response.message);
    this.loads.get(response.id)?.reject(error);
    this.loads.delete(response.id);
    this.pending.get(response.id)?.reject(error);
    this.pending.delete(response.id);
  }

  private failEverything(error: Error): void {
    for (const { reject } of this.pending.values()) reject(error);
    for (const { reject } of this.loads.values()) reject(error);
    this.pending.clear();
    this.loads.clear();
    this.loading = null;
  }
}
