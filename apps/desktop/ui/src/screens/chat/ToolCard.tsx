import { createSignal, For, Index, Show, type JSX } from "solid-js";
import "../../styles/toolcard.css";
import { Icon } from "../../components/icons";
import { Markdown } from "./Markdown";

import type { ToolActivity } from "../../service/chat-types";

export type ToolDecision = "allow-once" | "always" | "deny";
type Decision = ToolDecision;
/** What the user sends back for a question (ask_user) or a plan (exit_plan_mode). */
export type ToolAnswer = { answers: Array<{ question: string; answer: string | string[] }> } | { approved: boolean; feedback?: string };

const TRUNCATE_AT = 2000;

function chipLabel(state: ToolActivity["state"]): string {
  switch (state) {
    case "approval": return "Waiting for you";
    case "question": return "Question for you";
    case "plan": return "Plan to review";
    case "running": return "Running";
    case "done": return "Done";
    case "denied": return "Denied";
    case "error": return "Failed";
  }
}

interface Todo {
  content: string;
  status: "pending" | "in_progress" | "completed";
}
interface Question {
  question: string;
  options: string[];
  multi: boolean;
}

/** The tasks of a todo_write call, or null when the arguments are not a task list. */
export function todosOf(args: unknown): Todo[] | null {
  const list = (args as { todos?: unknown } | null)?.todos;
  if (!Array.isArray(list)) return null;
  const out: Todo[] = [];
  for (const t of list.slice(0, 50)) {
    const c = (t as { content?: unknown })?.content;
    const s = (t as { status?: unknown })?.status;
    if (typeof c !== "string" || (s !== "pending" && s !== "in_progress" && s !== "completed")) return null;
    out.push({ content: c.slice(0, 500), status: s });
  }
  return out;
}

function questionsOf(args: unknown): Question[] {
  const list = (args as { questions?: unknown } | null)?.questions;
  if (!Array.isArray(list)) return [];
  return list.slice(0, 4).flatMap((q) => {
    const x = q as { question?: unknown; options?: unknown; multi?: unknown };
    if (typeof x?.question !== "string") return [];
    const options = Array.isArray(x.options) ? x.options.filter((o): o is string => typeof o === "string").slice(0, 6) : [];
    return [{ question: x.question.slice(0, 1000), options, multi: x.multi === true }];
  });
}

const planOf = (args: unknown): string => {
  const p = (args as { plan?: unknown } | null)?.plan;
  return typeof p === "string" ? p.slice(0, 20000) : "";
};

const isAgentTool = (t: ToolActivity, name: string) => t.server === "cerebro" && t.name === name;

/** ask_user: one fieldset per question; options as radios or checkboxes, plus a free answer. */
function QuestionForm(props: { id: string; questions: Question[]; onAnswer: (a: ToolAnswer) => void }): JSX.Element {
  const [picked, setPicked] = createSignal<string[][]>(props.questions.map(() => []));
  const [other, setOther] = createSignal<string[]>(props.questions.map(() => ""));
  const answerFor = (i: number): string | string[] => {
    const q = props.questions[i]!;
    const free = other()[i]!.trim();
    const chosen = [...picked()[i]!, ...(free ? [free] : [])];
    return q.multi ? chosen : (chosen[chosen.length - 1] ?? "");
  };
  const complete = () => props.questions.every((_, i) => (Array.isArray(answerFor(i)) ? (answerFor(i) as string[]).length > 0 : answerFor(i) !== ""));
  const toggle = (i: number, opt: string, multi: boolean, on: boolean) =>
    setPicked((p) => p.map((list, j) => (j !== i ? list : multi ? (on ? [...list, opt] : list.filter((x) => x !== opt)) : on ? [opt] : [])));
  return (
    <form
      class="toolcard__question"
      onSubmit={(e) => {
        e.preventDefault();
        if (complete()) props.onAnswer({ answers: props.questions.map((q, i) => ({ question: q.question, answer: answerFor(i) })) });
      }}
    >
      <Index each={props.questions}>
        {(q, i) => (
          <fieldset class="toolcard__fieldset">
            <legend class="toolcard__prompt">{q().question}</legend>
            <For each={q().options}>
              {(opt) => (
                <label class="toolcard__option">
                  <input
                    type={q().multi ? "checkbox" : "radio"}
                    name={`${props.id}-q${i}`}
                    checked={picked()[i]!.includes(opt)}
                    onChange={(e) => {
                      toggle(i, opt, q().multi, e.currentTarget.checked);
                      if (!q().multi) setOther((o) => o.map((x, j) => (j === i ? "" : x)));
                    }}
                  />
                  <span>{opt}</span>
                </label>
              )}
            </For>
            <input
              class="input toolcard__other"
              type="text"
              aria-label={q().options.length ? `Another answer to: ${q().question}` : `Answer to: ${q().question}`}
              placeholder={q().options.length ? "Or type another answer" : "Your answer"}
              value={other()[i]}
              onInput={(e) => {
                const v = e.currentTarget.value;
                setOther((o) => o.map((x, j) => (j === i ? v : x)));
                if (!q().multi && v.trim()) setPicked((p) => p.map((l, j) => (j === i ? [] : l)));
              }}
            />
          </fieldset>
        )}
      </Index>
      <div class="toolcard__actions">
        <button class="btn btn--primary btn--sm" type="submit" disabled={!complete()}>
          Send answer
        </button>
        <button class="btn btn--tertiary btn--sm" type="button" onClick={() => props.onAnswer({ answers: props.questions.map((q) => ({ question: q.question, answer: "(the user chose not to answer)" })) })}>
          Skip
        </button>
      </div>
    </form>
  );
}

/** exit_plan_mode: the plan, then Approve or Keep planning with what to change. */
function PlanReview(props: { plan: string; onAnswer: (a: ToolAnswer) => void }): JSX.Element {
  const [feedback, setFeedback] = createSignal("");
  return (
    <div class="toolcard__plan" role="group" aria-label="Review the plan">
      <div class="toolcard__plan-body">
        <Markdown text={props.plan} />
      </div>
      <textarea class="input toolcard__feedback" rows={2} aria-label="What should change in the plan" placeholder="What should change (for Keep planning)" value={feedback()} onInput={(e) => setFeedback(e.currentTarget.value)} />
      <div class="toolcard__actions">
        <button class="btn btn--primary btn--sm" type="button" onClick={() => props.onAnswer({ approved: true })}>
          Approve plan
        </button>
        <button class="btn btn--secondary btn--sm" type="button" onClick={() => props.onAnswer({ approved: false, feedback: feedback().trim() })}>
          Keep planning
        </button>
      </div>
    </div>
  );
}

export function ToolCard(props: {
  tool: ToolActivity;
  onDecide?: ((id: string, decision: Decision) => void) | undefined;
  onAnswer?: ((id: string, answer: ToolAnswer) => void) | undefined;
}): JSX.Element {
  const tool = () => props.tool;
  const [showAll, setShowAll] = createSignal(false);

  const argsText = () => JSON.stringify(tool().arguments ?? {}, null, 2);
  const resultText = () => tool().result ?? "";
  const isLong = () => resultText().length > TRUNCATE_AT;
  const shownResult = () => (showAll() || !isLong() ? resultText() : resultText().slice(0, TRUNCATE_AT));
  const chipId = () => `toolcard-${tool().id}-state`;
  const descId = () => `toolcard-${tool().id}-desc`;
  const todos = () => (isAgentTool(tool(), "todo_write") ? todosOf(tool().arguments) : null);
  const answer = (a: ToolAnswer) => props.onAnswer?.(tool().id, a);

  const decide = (d: Decision) => () => props.onDecide?.(tool().id, d);

  return (
    <div class="toolcard" data-state={tool().state}>
      <div class="toolcard__head">
        <Icon name="tools" size={16} />
        <span class="toolcard__name mono">{tool().name}</span>
        <span class="toolcard__server">from {tool().server}</span>
        <span class={`toolcard__chip toolcard__chip--${tool().state}`} id={chipId()} aria-live="polite">
          <Show when={tool().state === "running"}>
            <span class="toolcard__spinner" aria-hidden="true" />
          </Show>
          {chipLabel(tool().state)}
        </span>
      </div>

      <Show
        when={todos()}
        fallback={
          <Show when={tool().state !== "question" && tool().state !== "plan" && !isAgentTool(tool(), "exit_plan_mode")}>
            <details class="toolcard__fold">
              <summary>Arguments</summary>
              <pre class="toolcard__pre">{argsText()}</pre>
            </details>
          </Show>
        }
      >
        {(list) => (
          <ul class="toolcard__todos" aria-label="Task list">
            <For each={list()}>
              {(t) => (
                <li data-status={t.status}>
                  <span class="toolcard__todo-mark" aria-hidden="true">
                    {t.status === "completed" ? "✓" : t.status === "in_progress" ? "▸" : "○"}
                  </span>
                  <span>{t.content}</span>
                  <span class="sr-only">{t.status === "completed" ? "(done)" : t.status === "in_progress" ? "(in progress)" : "(to do)"}</span>
                </li>
              )}
            </For>
          </ul>
        )}
      </Show>

      <Show when={tool().state === "approval"}>
        <div class="toolcard__approval" role="group" aria-label={`Approve ${tool().name}`} aria-describedby={descId()}>
          <p id={descId()} class="toolcard__prompt">
            The model wants to run this tool.
          </p>
          <div class="toolcard__actions">
            <button class="btn btn--primary btn--sm" type="button" onClick={decide("allow-once")}>
              Allow once
            </button>
            <button class="btn btn--secondary btn--sm" type="button" onClick={decide("always")}>
              Always allow
            </button>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={decide("deny")}>
              Deny
            </button>
          </div>
        </div>
      </Show>

      <Show when={tool().state === "question"}>
        <QuestionForm id={tool().id} questions={questionsOf(tool().arguments)} onAnswer={answer} />
      </Show>
      <Show when={tool().state === "plan"}>
        <PlanReview plan={planOf(tool().arguments)} onAnswer={answer} />
      </Show>
      <Show when={isAgentTool(tool(), "exit_plan_mode") && tool().state !== "plan" && planOf(tool().arguments)}>
        <details class="toolcard__fold">
          <summary>Plan</summary>
          <div class="toolcard__plan-body">
            <Markdown text={planOf(tool().arguments)} />
          </div>
        </details>
      </Show>

      <Show when={tool().result && !todos()}>
        <details class="toolcard__fold">
          <summary>Result</summary>
          <pre class="toolcard__pre">{shownResult()}</pre>
          <Show when={isLong()}>
            <button class="btn btn--tertiary btn--sm toolcard__showall" type="button" onClick={() => setShowAll(!showAll())} aria-expanded={showAll()}>
              {showAll() ? "Show less" : "Show all"}
            </button>
          </Show>
        </details>
      </Show>

      <Show when={tool().error}>
        <p class="toolcard__error" role={tool().state === "error" ? "alert" : undefined}>
          {tool().error}
        </p>
      </Show>
    </div>
  );
}
