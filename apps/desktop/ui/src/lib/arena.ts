/**
 * Arena helpers: the task categories, the local Elo leaderboard kept per category, and the export
 * payload. Elo is a ranking aid, not a measurement of quality; it only reflects the votes cast on
 * this computer.
 */
import { readPref, writePref } from "./storage";

export const ARENA_CATEGORIES = [
  { id: "general", label: "General", note: "Everyday questions and explanations." },
  { id: "coding", label: "Coding", note: "Code, debugging and technical answers." },
  { id: "writing", label: "Writing", note: "Drafting, editing and tone." },
  { id: "facts", label: "Facts and reasoning", note: "Factual answers, arithmetic and logic." },
] as const;

export type ArenaCategory = (typeof ARENA_CATEGORIES)[number]["id"];

export interface ArenaRating {
  rating: number;
  wins: number;
  losses: number;
  ties: number;
}

export type ArenaLeaderboard = Partial<Record<ArenaCategory, Record<string, ArenaRating>>>;

export const ARENA_ELO_KEY = "cerebro-arena-elo-v1";
export const ARENA_BASE_RATING = 1000;
export const ARENA_K = 32;

export function isArenaCategory(value: unknown): value is ArenaCategory {
  return typeof value === "string" && ARENA_CATEGORIES.some((c) => c.id === value);
}

export function categoryLabel(category: ArenaCategory): string {
  return ARENA_CATEGORIES.find((c) => c.id === category)?.label ?? "General";
}

export function expectedScore(a: number, b: number): number {
  return 1 / (1 + 10 ** ((b - a) / 400));
}

/** One side's new rating after a game: score 1 win, 0.5 tie, 0 loss. */
export function nextRating(rating: number, opponent: number, score: 0 | 0.5 | 1): number {
  return Math.round(rating + ARENA_K * (score - expectedScore(rating, opponent)));
}

/**
 * Records one vote. A winner beats every other answer; a tie gives every pair half a point. The
 * board is copied, never mutated, so Solid sees the change and storage never sees a shared object.
 */
export function recordVote(board: ArenaLeaderboard, category: ArenaCategory, winnerId: string | null, competitorIds: readonly string[]): ArenaLeaderboard {
  const ids = [...new Set(competitorIds)].filter(Boolean);
  if (ids.length < 2) return board;
  const current = board[category] ?? {};
  const next: Record<string, ArenaRating> = Object.fromEntries(ids.map((id) => [id, { ...(current[id] ?? { rating: ARENA_BASE_RATING, wins: 0, losses: 0, ties: 0 }) }]));
  const pairs: Array<[string, string, 0 | 0.5 | 1]> = [];
  if (winnerId && ids.includes(winnerId)) {
    for (const id of ids) if (id !== winnerId) pairs.push([winnerId, id, 1]);
  } else {
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) pairs.push([ids[i]!, ids[j]!, 0.5]);
  }
  for (const [a, b, score] of pairs) {
    const ra = next[a]!;
    const rb = next[b]!;
    const beforeA = ra.rating;
    const beforeB = rb.rating;
    ra.rating = nextRating(beforeA, beforeB, score);
    rb.rating = nextRating(beforeB, beforeA, (1 - score) as 0 | 0.5 | 1);
    if (score === 1) {
      ra.wins += 1;
      rb.losses += 1;
    } else {
      ra.ties += 1;
      rb.ties += 1;
    }
  }
  const stored: Record<string, ArenaRating> = { ...current, ...next };
  return { ...board, [category]: stored };
}

export interface ArenaLeaderRow {
  modelId: string;
  rating: number;
  games: number;
  wins: number;
  losses: number;
  ties: number;
}

/** Highest rating first; ties broken by more games, then by id for a stable order. */
export function leaderboardRows(board: ArenaLeaderboard, category: ArenaCategory): ArenaLeaderRow[] {
  const entries = Object.entries(board[category] ?? {});
  return entries
    .map(([modelId, r]) => ({ modelId, rating: r.rating, games: r.wins + r.losses + r.ties, wins: r.wins, losses: r.losses, ties: r.ties }))
    .sort((a, b) => b.rating - a.rating || b.games - a.games || a.modelId.localeCompare(b.modelId));
}

function isRating(v: unknown): v is ArenaRating {
  if (typeof v !== "object" || v === null) return false;
  const r = v as Record<string, unknown>;
  return (
    typeof r["rating"] === "number" &&
    Number.isFinite(r["rating"]) &&
    r["rating"] >= 100 &&
    r["rating"] <= 3000 &&
    typeof r["wins"] === "number" &&
    Number.isInteger(r["wins"]) &&
    r["wins"] >= 0 &&
    typeof r["losses"] === "number" &&
    Number.isInteger(r["losses"]) &&
    r["losses"] >= 0 &&
    typeof r["ties"] === "number" &&
    Number.isInteger(r["ties"]) &&
    r["ties"] >= 0
  );
}

/** Anything stored is re-checked field by field; a bad value is ignored rather than trusted. */
export function parseLeaderboard(raw: string | null): ArenaLeaderboard {
  if (!raw) return {};
  try {
    const value = JSON.parse(raw) as unknown;
    if (typeof value !== "object" || value === null) return {};
    const out: ArenaLeaderboard = {};
    for (const [category, ratings] of Object.entries(value as Record<string, unknown>)) {
      if (!isArenaCategory(category) || typeof ratings !== "object" || ratings === null) continue;
      const clean: Record<string, ArenaRating> = {};
      for (const [id, rating] of Object.entries(ratings as Record<string, unknown>)) {
        if (id.length > 64 || !isRating(rating)) continue;
        clean[id] = { rating: rating.rating, wins: rating.wins, losses: rating.losses, ties: rating.ties };
      }
      out[category] = clean;
    }
    return out;
  } catch {
    return {};
  }
}

export function loadLeaderboard(): ArenaLeaderboard {
  return parseLeaderboard(readPref(ARENA_ELO_KEY));
}

export function saveLeaderboard(board: ArenaLeaderboard): void {
  writePref(ARENA_ELO_KEY, JSON.stringify(board));
}

export function formatRating(rating: number): string {
  return Math.round(rating).toLocaleString();
}

export interface ArenaReplyExport {
  label: string;
  modelId: string;
  modelName: string;
  text: string;
  ttftMs: number | null;
  tokPerSec: number | null;
  outputTokens: number | null;
  chars: number;
  error: string | null;
  stopped: boolean;
}

export interface ArenaExport {
  format: "cerebro-arena";
  version: 1;
  exportedAt: string;
  category: ArenaCategory;
  categoryLabel: string;
  prompt: string;
  blind: boolean;
  vote: string;
  replies: ArenaReplyExport[];
  leaderboard: ArenaLeaderRow[];
}

export function exportPayload(input: {
  at: Date;
  category: ArenaCategory;
  prompt: string;
  blind: boolean;
  vote: string;
  replies: readonly ArenaReplyExport[];
  leaderboard: ArenaLeaderRow[];
}): ArenaExport {
  return {
    format: "cerebro-arena",
    version: 1,
    exportedAt: input.at.toISOString(),
    category: input.category,
    categoryLabel: categoryLabel(input.category),
    prompt: input.prompt,
    blind: input.blind,
    vote: input.vote,
    replies: input.replies.map((r) => ({ ...r })),
    leaderboard: input.leaderboard.map((r) => ({ ...r })),
  };
}

/** A safe file name: no model name or prompt goes into it. */
export function arenaFilename(at: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `cerebro-arena-${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())}-${pad(at.getHours())}${pad(at.getMinutes())}.json`;
}
