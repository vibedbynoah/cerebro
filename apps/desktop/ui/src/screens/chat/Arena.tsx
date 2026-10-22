import "../../styles/arena.css";
import { createEffect, createMemo, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { SettingSwitch } from "../../components/SettingSwitch";
import {
  ARENA_CATEGORIES,
  arenaFilename,
  categoryLabel,
  exportPayload,
  formatRating,
  leaderboardRows,
  loadLeaderboard,
  recordVote,
  saveLeaderboard,
  type ArenaCategory,
  type ArenaReplyExport,
} from "../../lib/arena";
import { saveText } from "../../lib/download";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import { DEFAULT_SAMPLER, type ChatStats } from "../../service/chat-types";
import type { LibraryModel } from "../../service/types";

interface ArenaReply {
  modelId: string;
  modelName: string;
  label: string;
  text: string;
  tokens: number;
  stats: ChatStats | null;
  firstTokenMs: number | null;
  liveTps: number | null;
  error: string | null;
  streaming: boolean;
}

interface ArenaRound {
  prompt: string;
  category: ArenaCategory;
  blind: boolean;
  replies: ArenaReply[];
  /** A model id, "tie", or null before the vote. */
  vote: string | null;
  stopped: boolean;
}

const MAX_MODELS = 4;
const MAX_PROMPT = 4_000;
const LABELS = ["A", "B", "C", "D"] as const;

/**
 * Arena: one prompt, two to four models, answers shown side by side with their measured speed.
 * Local models run in turn because one graphics card runs one model at a time. Blind mode hides the
 * names until a vote; votes update a leaderboard kept only on this computer.
 */
export function Arena(props: { models: LibraryModel[]; active?: boolean }): JSX.Element {
  const service = useService();
  const [selected, setSelected] = createSignal<string[]>([]);
  const [category, setCategory] = createSignal<ArenaCategory>("general");
  const [boardCategory, setBoardCategory] = createSignal<ArenaCategory>("general");
  const [blind, setBlind] = createSignal(true);
  const [query, setQuery] = createSignal("");
  const [prompt, setPrompt] = createSignal("");
  const [maxTokens, setMaxTokens] = createSignal(512);
  const [temperature, setTemperature] = createSignal(0.7);
  const [round, setRound] = createSignal<ArenaRound | null>(null);
  const [running, setRunning] = createSignal(false);
  const [leaderboard, setLeaderboard] = createSignal(loadLeaderboard());
  let abort: AbortController | null = null;

  const models = createMemo(() => props.models.filter((m) => m.kind === "text" || m.kind === "vision"));
  /** The library can hold hundreds of files: the search keeps the picker one screen tall. */
  const visibleModels = createMemo(() => {
    const needle = query().trim().toLowerCase();
    if (!needle) return models();
    return models().filter((m) => selected().includes(m.id) || `${m.name} ${m.quant} ${m.params} ${m.family}`.toLowerCase().includes(needle));
  });
  const selectedModels = createMemo(() => {
    const byId = new Map(models().map((m) => [m.id, m]));
    return selected()
      .map((id) => byId.get(id))
      .filter((m): m is LibraryModel => m !== undefined);
  });
  const canRun = () => !running() && prompt().trim().length > 0 && selectedModels().length >= 2;

  // Two models are picked for the first visit; a model that leaves the library is dropped.
  createEffect(() => {
    const list = models();
    if (!list.length) return;
    const valid = selected().filter((id) => list.some((m) => m.id === id));
    if (valid.length !== selected().length) setSelected(valid);
    if (valid.length === 0) {
      // Compare different models where possible, not two quants of the same one.
      const first = list[0]!;
      const second = list.find((m) => m.name !== first.name);
      setSelected(second ? [first.id, second.id] : [first.id]);
    }
  });

  createEffect(() => {
    // The view stays mounted so a finished round survives a look at Chat, but a run in progress
    // stops when it is hidden: one graphics card cannot serve a hidden Arena and Chat at once.
    if (props.active === false) abort?.abort();
  });

  onCleanup(() => {
    abort?.abort();
  });

  const toggleModel = (id: string) => {
    if (running()) return;
    setSelected((current) => {
      if (current.includes(id)) return current.filter((x) => x !== id);
      if (current.length >= MAX_MODELS) return current;
      return [...current, id];
    });
  };

  const updateReply = (index: number, change: (reply: ArenaReply) => void) => {
    setRound((current) => {
      if (!current) return current;
      return {
        ...current,
        replies: current.replies.map((reply, i) => {
          if (i !== index) return reply;
          const copy = { ...reply };
          change(copy);
          return copy;
        }),
      };
    });
  };

  const run = async () => {
    if (!canRun()) return;
    const snapshot = selectedModels();
    const text = prompt().trim();
    const chosen = category();
    const isBlind = blind();
    const replies: ArenaReply[] = snapshot.map((model, i) => ({
      modelId: model.id,
      modelName: model.name,
      label: LABELS[i] ?? String(i + 1),
      text: "",
      tokens: 0,
      stats: null,
      firstTokenMs: null,
      liveTps: null,
      error: null,
      streaming: false,
    }));
    setRound({ prompt: text, category: chosen, blind: isBlind, replies, vote: null, stopped: false });
    setBoardCategory(chosen);
    setRunning(true);
    const controller = new AbortController();
    abort = controller;

    for (let i = 0; i < snapshot.length; i++) {
      if (controller.signal.aborted) break;
      const model = snapshot[i]!;
      const label = LABELS[i] ?? String(i + 1);
      let pending = "";
      let pendingFrame = 0;
      let tokens = 0;
      let firstAt = 0;
      const flush = () => {
        pendingFrame = 0;
        if (!pending) return;
        const add = pending;
        pending = "";
        updateReply(i, (reply) => {
          reply.text += add;
        });
      };
      updateReply(i, (reply) => {
        reply.streaming = true;
      });
      try {
        await service.loadModel(model.id, null);
        if (controller.signal.aborted) break;
        await service.chat(
          {
            modelId: model.id,
            systemPrompt: "",
            turns: [{ role: "user", text }],
            sampler: { ...DEFAULT_SAMPLER, temperature: temperature(), maxTokens: maxTokens() },
            native: null,
            passes: 0,
            effort: "low",
            verbosity: "normal",
            source: "none",
            keepCache: false,
          },
          (event) => {
            switch (event.type) {
              case "start":
                updateReply(i, (reply) => {
                  reply.firstTokenMs = event.ttftMs;
                });
                break;
              case "token": {
                tokens += 1;
                if (!firstAt) firstAt = performance.now();
                pending += event.text;
                if (!pendingFrame) pendingFrame = requestAnimationFrame(flush);
                updateReply(i, (reply) => {
                  reply.tokens = tokens;
                  reply.liveTps = tokens / Math.max(0.05, (performance.now() - firstAt) / 1000);
                });
                break;
              }
              case "done":
                cancelAnimationFrame(pendingFrame);
                flush();
                updateReply(i, (reply) => {
                  reply.stats = event.stats;
                  reply.streaming = false;
                });
                break;
              case "error":
                updateReply(i, (reply) => {
                  reply.error = event.message;
                  reply.streaming = false;
                });
                break;
              default:
                break;
            }
          },
          controller.signal,
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          const message = error instanceof Error ? error.message : "The model stopped answering.";
          updateReply(i, (reply) => {
            reply.error = message;
          });
          notify(`${label} could not answer`, message, "error");
        }
      } finally {
        cancelAnimationFrame(pendingFrame);
        flush();
        updateReply(i, (reply) => {
          reply.streaming = false;
        });
      }
    }

    setRunning(false);
    setRound((current) => (current ? { ...current, stopped: controller.signal.aborted } : current));
    if (abort === controller) abort = null;
  };

  const stop = () => {
    abort?.abort();
  };

  const revealed = () => {
    const current = round();
    return !current || !current.blind || current.vote !== null;
  };

  const castVote = (choice: string) => {
    const current = round();
    if (!current || current.vote !== null || running()) return;
    const winnerId = choice === "tie" ? null : choice;
    const next = recordVote(
      leaderboard(),
      current.category,
      winnerId,
      current.replies.map((r) => r.modelId),
    );
    setLeaderboard(next);
    saveLeaderboard(next);
    setRound({ ...current, vote: winnerId ?? "tie" });
    setBoardCategory(current.category);
    notify("Vote counted", winnerId ? `The leaderboard for ${categoryLabel(current.category).toLowerCase()} is updated.` : "A tie adds half a point to every answer.", "success");
  };

  const exportResults = () => {
    const current = round();
    if (!current) return;
    const at = new Date();
    const replies: ArenaReplyExport[] = current.replies.map((reply) => ({
      label: reply.label,
      modelId: reply.modelId,
      modelName: reply.modelName,
      text: reply.text,
      ttftMs: reply.stats?.ttftMs ?? reply.firstTokenMs,
      tokPerSec: reply.stats?.tokPerSec ?? reply.liveTps,
      outputTokens: reply.stats?.outputTokens ?? reply.tokens,
      chars: reply.text.length,
      error: reply.error,
      // A reply without final stats when the run stopped did not finish, even if the stream had
      // already been switched off by the time the round settled.
      stopped: current.stopped && reply.stats === null && reply.error === null,
    }));
    const payload = exportPayload({
      at,
      category: current.category,
      prompt: current.prompt,
      blind: current.blind,
      vote: current.vote ?? "none",
      replies,
      leaderboard: leaderboardRows(leaderboard(), current.category),
    });
    const name = arenaFilename(at);
    if (saveText(name, "application/json", JSON.stringify(payload, null, 2))) notify("Results exported", `${name} was saved.`, "success");
    else notify("Export failed", "The file could not be created. Nothing was saved.", "error");
  };

  const modelName = (id: string) => models().find((m) => m.id === id)?.name ?? id;

  const metric = (reply: ArenaReply, key: "ttft" | "tps" | "tokens"): string => {
    if (key === "ttft") {
      const value = reply.stats?.ttftMs ?? reply.firstTokenMs;
      return value === null ? "—" : `${Math.round(value).toLocaleString()} ms`;
    }
    if (key === "tps") {
      const value = reply.stats?.tokPerSec ?? reply.liveTps;
      return value === null ? "—" : `${value.toFixed(1)} tok/s`;
    }
    return (reply.stats?.outputTokens ?? reply.tokens).toLocaleString();
  };

  return (
    <div class="arena">
      <p class="arena__note">
        <Icon name="info" size={16} />
        <span>Models take turns. The engine keeps one model loaded at a time, so each answer starts after the one before it.</span>
      </p>

      <form
        class="card arena-setup"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <div class="arena-setup__head">
          <div>
            <h2 class="arena-setup__title">Compare answers</h2>
            <p class="hint">Pick two to four models, ask one prompt, then vote for the answer you prefer.</p>
          </div>
          <Show when={!__DESKTOP__}>
            <IllustrativeFlag />
          </Show>
        </div>

        <fieldset class="arena-models" disabled={running()}>
          <legend>Models</legend>
          <div class="arena-models__filter">
            <label class="arena-search">
              <Icon name="search" size={15} />
              <span class="sr-only">Search models</span>
              <input type="search" placeholder="Search models" value={query()} onInput={(event) => setQuery(event.currentTarget.value)} disabled={running()} />
            </label>
            <span class="hint" role="status">
              {visibleModels().length} of {models().length} models.
            </span>
          </div>
          <div class="arena-models__grid">
            <For each={visibleModels()}>
              {(model) => (
                <label class="arena-model" data-selected={selected().includes(model.id)}>
                  <input
                    type="checkbox"
                    checked={selected().includes(model.id)}
                    disabled={running() || (!selected().includes(model.id) && selected().length >= MAX_MODELS)}
                    onChange={() => toggleModel(model.id)}
                    data-testid={`arena-model-${model.id}`}
                  />
                  <span class="arena-model__text">
                    <span class="arena-model__name">{model.name}</span>
                    <span class="arena-model__meta">
                      {model.quant} · {model.params || "size not reported"}
                    </span>
                  </span>
                </label>
              )}
            </For>
          </div>
          <Show when={!visibleModels().length}>
            <p class="hint">No model matches that search.</p>
          </Show>
          <Show when={models().length < 2}>
            <p class="hint">Add another text model on the Models screen to compare two answers.</p>
          </Show>
        </fieldset>

        <div class="arena-setup__row">
          <label class="arena-field">
            <span>Task category</span>
            <select value={category()} onChange={(event) => setCategory(event.currentTarget.value as ArenaCategory)} disabled={running()}>
              <For each={ARENA_CATEGORIES}>{(c) => <option value={c.id}>{c.label}</option>}</For>
            </select>
          </label>
          <div class="arena-blind">
            <SettingSwitch id="arena-blind" label="Blind mode" note="Hide model names until you vote. Names appear after." checked={blind()} disabled={running()} onChange={setBlind} />
          </div>
        </div>

        <label class="arena-field arena-field--prompt">
          <span>Prompt</span>
          <textarea
            rows="3"
            maxlength={MAX_PROMPT}
            value={prompt()}
            placeholder="Ask the models the same question."
            disabled={running()}
            onInput={(event) => setPrompt(event.currentTarget.value)}
            data-testid="arena-prompt"
          />
          <span class="hint">
            {prompt().length.toLocaleString()} of {MAX_PROMPT.toLocaleString()} characters.
          </span>
        </label>

        <details class="arena-advanced">
          <summary>Advanced reply settings</summary>
          <div class="arena-advanced__body">
            <label class="arena-field">
              <span>Longest reply</span>
              <input
                type="number"
                min="64"
                max="2048"
                step="64"
                value={maxTokens()}
                disabled={running()}
                onChange={(event) => setMaxTokens(Math.min(2048, Math.max(64, Number(event.currentTarget.value) || 512)))}
              />
              <span class="hint">Tokens. A shorter limit keeps a comparison quick.</span>
            </label>
            <label class="arena-field">
              <span>Temperature</span>
              <input
                type="number"
                min="0"
                max="1.5"
                step="0.05"
                value={temperature()}
                disabled={running()}
                onChange={(event) => setTemperature(Math.min(1.5, Math.max(0, Number(event.currentTarget.value) || 0)))}
              />
              <span class="hint">0 is the most predictable; higher values vary more.</span>
            </label>
          </div>
        </details>

        <div class="arena-setup__actions">
          <button class="btn btn--primary" type="submit" disabled={!canRun()} data-press data-testid="arena-run">
            <Icon name="play" size={15} />
            {running() ? "Comparing…" : "Compare answers"}
          </button>
          <Show when={running()}>
            <button class="btn btn--secondary" type="button" onClick={stop} data-press data-testid="arena-stop">
              <Icon name="stop" size={14} />
              Stop
            </button>
          </Show>
          <Show when={selectedModels().length < 2 && models().length >= 2}>
            <span class="hint">Choose at least two models.</span>
          </Show>
        </div>
      </form>

      <Show when={round()}>
        {(current) => (
          <section class="arena-results" aria-label="Answers">
            <div class="arena-results__head">
              <div>
                <h2 class="arena-results__title">Answers</h2>
                <p class="hint">
                  {categoryLabel(current().category)} · {current().prompt}
                </p>
              </div>
              <div class="arena-results__actions">
                <Show when={!revealed()}>
                  <span class="chip" data-tone="neutral">
                    Names hidden until you vote
                  </span>
                </Show>
                <Show when={current().stopped}>
                  <span class="chip" data-tone="off">
                    Stopped
                  </span>
                </Show>
                <button class="btn btn--secondary btn--sm" type="button" onClick={exportResults} data-press data-testid="arena-export">
                  <Icon name="download" size={14} />
                  Export results
                </button>
              </div>
            </div>

            <div class="arena-replies" data-count={current().replies.length}>
              <For each={current().replies}>
                {(reply) => (
                  <article class="card arena-reply" data-testid="arena-reply" data-label={reply.label}>
                    <header class="arena-reply__head">
                      <div class="arena-reply__who">
                        <span class="arena-reply__letter" aria-hidden="true">
                          {reply.label}
                        </span>
                        <div>
                          <h3 class="arena-reply__label">Answer {reply.label}</h3>
                          <Show when={revealed()} fallback={<p class="hint">Model name hidden until the vote.</p>}>
                            <p class="arena-reply__model">{reply.modelName}</p>
                          </Show>
                        </div>
                      </div>
                      <span class="chip" data-tone={reply.error || (!reply.streaming && !reply.stats && current().stopped) ? "off" : reply.streaming ? "active" : reply.stats ? "on" : "neutral"}>
                        {reply.error ? "Failed" : reply.streaming ? "Writing" : reply.stats ? "Done" : current().stopped ? "Stopped" : "Waiting"}
                      </span>
                    </header>

                    <dl class="arena-reply__metrics">
                      <div>
                        <dt>First token</dt>
                        <dd>{metric(reply, "ttft")}</dd>
                      </div>
                      <div>
                        <dt>Speed</dt>
                        <dd>{metric(reply, "tps")}</dd>
                      </div>
                      <div>
                        <dt>Length</dt>
                        <dd>{metric(reply, "tokens")} tokens</dd>
                      </div>
                    </dl>

                    <Show when={reply.error}>
                      <p class="notice notice--inline" role="alert">
                        {reply.error}
                      </p>
                    </Show>
                    <div class="arena-reply__body" role="log" aria-live="polite" aria-label={`Answer ${reply.label}`}>
                      <Show when={reply.text} fallback={<p class="hint">{reply.streaming ? "Waiting for the first token…" : "No answer."}</p>}>
                        <p class="arena-reply__text">{reply.text}</p>
                      </Show>
                    </div>
                  </article>
                )}
              </For>
            </div>

            <div class="card arena-vote">
              <Show
                when={current().vote === null}
                fallback={
                  <p class="arena-vote__done" role="status">
                    <Icon name="check" size={17} />
                    <span>{current().vote === "tie" ? "Vote saved as a tie." : `Vote saved for ${modelName(current().vote!)}.`} The leaderboard is updated.</span>
                  </p>
                }
              >
                <h3 class="arena-vote__title">Which answer is better?</h3>
                <p class="hint">One vote per comparison. The ratings below change only from votes on this computer.</p>
                <div class="arena-vote__choices" role="group" aria-label="Vote for an answer">
                  <For each={current().replies}>
                    {(reply) => (
                      <button
                        class="btn btn--secondary arena-vote__choice"
                        type="button"
                        disabled={running() || current().vote !== null || reply.text.length === 0}
                        onClick={() => castVote(reply.modelId)}
                        data-press
                        data-testid={`arena-vote-${reply.label.toLowerCase()}`}
                      >
                        <span class="arena-vote__letter" aria-hidden="true">
                          {reply.label}
                        </span>
                        <span>Answer {reply.label}</span>
                        <Show when={revealed()}>
                          <span class="arena-vote__name">{reply.modelName}</span>
                        </Show>
                      </button>
                    )}
                  </For>
                  <button
                    class="btn btn--tertiary arena-vote__choice"
                    type="button"
                    disabled={running() || current().vote !== null || current().replies.filter((r) => r.text.length > 0).length < 2}
                    onClick={() => castVote("tie")}
                    data-press
                    data-testid="arena-vote-tie"
                  >
                    Tie
                  </button>
                </div>
              </Show>
            </div>
          </section>
        )}
      </Show>

      <section class="card arena-board" aria-labelledby="arena-board-title">
        <div class="arena-board__head">
          <div>
            <h2 class="arena-board__title" id="arena-board-title">
              Leaderboard
            </h2>
            <p class="hint">Elo ratings from your votes here. They are a ranking aid, not an accuracy score.</p>
          </div>
          <label class="arena-field arena-field--compact">
            <span class="sr-only">Leaderboard category</span>
            <select value={boardCategory()} onChange={(event) => setBoardCategory(event.currentTarget.value as ArenaCategory)}>
              <For each={ARENA_CATEGORIES}>{(c) => <option value={c.id}>{c.label}</option>}</For>
            </select>
          </label>
        </div>
        <Show
          when={leaderboardRows(leaderboard(), boardCategory()).length}
          fallback={<p class="hint">No votes in {categoryLabel(boardCategory()).toLowerCase()} yet. Run a comparison and vote to start the board.</p>}
        >
          <div class="arena-board__scroll">
            <table class="arena-board__table">
              <caption class="sr-only">Ratings for {categoryLabel(boardCategory())}</caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Model</th>
                  <th scope="col">Rating</th>
                  <th scope="col">Wins</th>
                  <th scope="col">Losses</th>
                  <th scope="col">Ties</th>
                </tr>
              </thead>
              <tbody>
                <For each={leaderboardRows(leaderboard(), boardCategory())}>
                  {(row, index) => (
                    <tr data-testid="arena-leader-row">
                      <td class="mono">{index() + 1}</td>
                      <th scope="row">{modelName(row.modelId)}</th>
                      <td class="mono arena-board__rating">{formatRating(row.rating)}</td>
                      <td class="mono">{row.wins}</td>
                      <td class="mono">{row.losses}</td>
                      <td class="mono">{row.ties}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </Show>
      </section>
    </div>
  );
}
