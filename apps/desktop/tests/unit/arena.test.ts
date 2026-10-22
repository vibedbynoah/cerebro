import { beforeEach, describe, expect, it } from "vitest";
import {
  ARENA_BASE_RATING,
  ARENA_CATEGORIES,
  arenaFilename,
  categoryLabel,
  expectedScore,
  exportPayload,
  formatRating,
  isArenaCategory,
  leaderboardRows,
  loadLeaderboard,
  nextRating,
  parseLeaderboard,
  recordVote,
  saveLeaderboard,
  type ArenaLeaderboard,
  type ArenaReplyExport,
} from "../../ui/src/lib/arena";

const reply = (modelId: string, text = "An answer."): ArenaReplyExport => ({
  label: "A",
  modelId,
  modelName: `${modelId} name`,
  text,
  ttftMs: 120,
  tokPerSec: 20.5,
  outputTokens: 40,
  chars: text.length,
  error: null,
  stopped: false,
});

beforeEach(() => localStorage.clear());

describe("categories", () => {
  it("offers four fixed categories with plain labels", () => {
    expect(ARENA_CATEGORIES.map((c) => c.id)).toEqual(["general", "coding", "writing", "facts"]);
    expect(categoryLabel("facts")).toBe("Facts and reasoning");
    for (const c of ARENA_CATEGORIES) expect(isArenaCategory(c.id)).toBe(true);
  });

  it("rejects anything that is not a known category", () => {
    for (const bad of ["", "Coding", "__proto__", null, 3, {}]) expect(isArenaCategory(bad)).toBe(false);
  });
});

describe("Elo maths", () => {
  it("gives an even game an even chance and sums to one", () => {
    expect(expectedScore(1000, 1000)).toBe(0.5);
    expect(expectedScore(1200, 1000) + expectedScore(1000, 1200)).toBeCloseTo(1, 12);
    expect(expectedScore(1200, 1000)).toBeGreaterThan(0.5);
  });

  it("adds or removes 16 points for an even win or loss, and nothing for a tie", () => {
    expect(nextRating(1000, 1000, 1)).toBe(1016);
    expect(nextRating(1000, 1000, 0)).toBe(984);
    expect(nextRating(1000, 1000, 0.5)).toBe(1000);
    // An upset is worth more than beating a much weaker opponent.
    expect(nextRating(1000, 1400, 1)).toBeGreaterThan(nextRating(1000, 600, 1));
    expect(nextRating(1000, 1400, 1)).toBe(1029);
  });

  it("formats a rating with the reader's grouping", () => {
    expect(formatRating(1016.4)).toBe((1016).toLocaleString());
    expect(formatRating(1000)).toBe((1000).toLocaleString());
  });
});

describe("recording votes", () => {
  it("starts both models at the base rating and gives the winner a win", () => {
    const board = recordVote({}, "general", "a", ["a", "b"]);
    const rows = leaderboardRows(board, "general");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ modelId: "a", rating: 1016, games: 1, wins: 1, losses: 0, ties: 0 });
    expect(rows[1]).toMatchObject({ modelId: "b", rating: 984, games: 1, wins: 0, losses: 1, ties: 0 });
  });

  it("gives every pair half a point for a tie", () => {
    const board = recordVote({}, "coding", null, ["a", "b"]);
    for (const row of leaderboardRows(board, "coding")) {
      expect(row).toMatchObject({ rating: ARENA_BASE_RATING, wins: 0, losses: 0, ties: 1 });
    }
  });

  it("lets a winner beat every other answer", () => {
    const board = recordVote({}, "writing", "a", ["a", "b", "c"]);
    const rows = leaderboardRows(board, "writing");
    expect(rows[0]!.modelId).toBe("a");
    expect(rows[0]!.wins).toBe(2);
    expect(rows[0]!.losses).toBe(0);
    expect(rows.filter((r) => r.modelId !== "a").every((r) => r.losses === 1 && r.wins === 0)).toBe(true);
    expect(rows[0]!.rating).toBeGreaterThan(1016);
  });

  it("does not mutate the board it was given", () => {
    const board: ArenaLeaderboard = { general: { a: { rating: 1000, wins: 0, losses: 0, ties: 0 } } };
    const frozen = JSON.stringify(board);
    const next = recordVote(board, "general", "a", ["a", "b"]);
    expect(JSON.stringify(board)).toBe(frozen);
    expect(next).not.toBe(board);
    expect(next.general!.b).toMatchObject({ wins: 0, losses: 1 });
  });

  it("keeps each category separate and ignores a vote that cannot be paired", () => {
    const first = recordVote({}, "general", "a", ["a", "b"]);
    expect(first.coding).toBeUndefined();
    expect(recordVote(first, "general", "a", ["a"])).toBe(first);
    expect(recordVote(first, "general", "a", [])).toBe(first);
  });

  it("does not count the same model twice", () => {
    const board = recordVote({}, "general", "a", ["a", "a", "b"]);
    expect(leaderboardRows(board, "general")).toHaveLength(2);
    expect(leaderboardRows(board, "general")[0]!.wins).toBe(1);
  });

  it("turns an unknown winner into a tie rather than inventing a rating", () => {
    const board = recordVote({}, "general", "missing", ["a", "b"]);
    for (const row of leaderboardRows(board, "general")) expect(row.ties).toBe(1);
  });
});

describe("leaderboard rows", () => {
  it("sorts by rating, then games, then id", () => {
    const board: ArenaLeaderboard = {
      general: {
        z: { rating: 1000, wins: 2, losses: 0, ties: 0 },
        a: { rating: 1000, wins: 1, losses: 0, ties: 0 },
        c: { rating: 1100, wins: 0, losses: 1, ties: 0 },
        b: { rating: 1000, wins: 2, losses: 0, ties: 0 },
      },
    };
    expect(leaderboardRows(board, "general").map((r) => r.modelId)).toEqual(["c", "b", "z", "a"]);
    expect(leaderboardRows(board, "coding")).toEqual([]);
  });
});

describe("reading and saving the board", () => {
  it("accepts a valid saved board", () => {
    const board = recordVote({}, "facts", "a", ["a", "b"]);
    expect(parseLeaderboard(JSON.stringify(board))).toEqual(board);
  });

  it("drops unknown categories, bad values and oversized ids", () => {
    const raw = JSON.stringify({
      general: {
        good: { rating: 1040, wins: 2, losses: 0, ties: 1 },
        low: { rating: 20, wins: 2, losses: 0, ties: 0 },
        high: { rating: 9000, wins: 2, losses: 0, ties: 0 },
        fractional: { rating: 1000, wins: 1.5, losses: 0, ties: 0 },
        missing: { rating: 1000, wins: 0, losses: 0 },
        string: { rating: "1000", wins: 0, losses: 0, ties: 0 },
        ["x".repeat(65)]: { rating: 1000, wins: 0, losses: 0, ties: 0 },
      },
      invented: { a: { rating: 1000, wins: 0, losses: 0, ties: 0 } },
    });
    expect(parseLeaderboard(raw)).toEqual({ general: { good: { rating: 1040, wins: 2, losses: 0, ties: 1 } } });
  });

  it("returns an empty board for junk, null or a non-object", () => {
    for (const raw of [null, "", "not json", "[]", "null", '"text"']) expect(parseLeaderboard(raw)).toEqual({});
  });

  it("round-trips through local storage", () => {
    const board = recordVote({}, "general", "a", ["a", "b"]);
    saveLeaderboard(board);
    expect(loadLeaderboard()).toEqual(board);
    localStorage.setItem("cerebro-arena-elo-v1", "{");
    expect(loadLeaderboard()).toEqual({});
  });
});

describe("exports", () => {
  it("records the prompt, vote, replies and board, without sharing arrays", () => {
    const replies = [reply("a"), reply("b")];
    const rows = leaderboardRows(recordVote({}, "coding", "a", ["a", "b"]), "coding");
    const at = new Date(2026, 9, 21, 14, 5);
    const payload = exportPayload({ at, category: "coding", prompt: "Write a loop", blind: true, vote: "a", replies, leaderboard: rows });
    expect(payload).toMatchObject({
      format: "cerebro-arena",
      version: 1,
      exportedAt: at.toISOString(),
      category: "coding",
      categoryLabel: "Coding",
      prompt: "Write a loop",
      blind: true,
      vote: "a",
    });
    expect(payload.replies).toEqual(replies);
    expect(payload.replies[0]).not.toBe(replies[0]);
    expect(payload.leaderboard[0]!.modelId).toBe("a");
    expect(payload.leaderboard[0]).not.toBe(rows[0]);
  });

  it("builds a safe file name from the date only", () => {
    const name = arenaFilename(new Date(2026, 0, 2, 3, 4));
    expect(name).toBe("cerebro-arena-2026-01-02-0304.json");
    expect(name).not.toMatch(/[^a-z0-9.-]/);
  });
});
