/**
 * The shareable report card: a self-contained HTML file and a PNG drawn on a canvas. Both contain
 * only what the run recorded: no example values and no claims beyond the measurements.
 */
import type { BenchRun } from "../service/bench-types";
import { BENCH_CATEGORY_LABELS, contextDetail, contextLabel, machineLine, placementLine, statusLabel } from "./bench";

export function escapeHtml(text: string): string {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function number(value: number, digits = 1): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function pointCells(run: BenchRun, context: number): string[] {
  const point = run.points.find((p) => p.context === context);
  if (!point || !point.ran) {
    const note = point?.note ?? "Not measured.";
    return ["Not run", escapeHtml(note), "—", "—", "—", "—"];
  }
  return [
    escapeHtml(contextLabel(point.context)),
    `${point.promptTokens.toLocaleString()} tokens`,
    `${number(point.prefillTps)} tok/s`,
    `${number(point.decodeTps)} tok/s`,
    `${Math.round(point.ttftMs).toLocaleString()} ms`,
    `${number(point.memoryUsedGiB, 2)} GiB`,
  ];
}

function categoryScores(run: BenchRun): Array<{ label: string; correct: number; asked: number }> {
  const groups = new Map<string, { correct: number; asked: number }>();
  for (const q of run.accuracy.questions) {
    const g = groups.get(q.category) ?? { correct: 0, asked: 0 };
    g.asked += 1;
    if (q.correct) g.correct += 1;
    groups.set(q.category, g);
  }
  return [...groups.entries()].map(([category, score]) => ({
    label: BENCH_CATEGORY_LABELS[category as keyof typeof BENCH_CATEGORY_LABELS] ?? category,
    ...score,
  }));
}

/** A complete, script-free HTML document with the run's report card. */
export function reportHtml(run: BenchRun): string {
  const rows = [0, 4096, 16384]
    .map((context) => {
      const cells = pointCells(run, context);
      const detail = contextDetail(context);
      return `<tr><th scope="row">${escapeHtml(detail)}<span class="sub">${cells[0]}</span></th><td>${cells[1]}</td><td>${cells[2]}</td><td>${cells[3]}</td><td>${cells[4]}</td><td>${cells[5]}</td></tr>`;
    })
    .join("\n");
  const questionRows = run.accuracy.questions
    .map(
      (q) =>
        `<tr><td>${escapeHtml(q.question)}</td><td>${escapeHtml(q.answer || "No answer")}</td><td>${escapeHtml(q.expected)}</td><td class="${q.correct ? "ok" : "no"}">${q.correct ? "Correct" : "Wrong"}</td></tr>`,
    )
    .join("\n");
  const categories = categoryScores(run)
    .map((c) => `<li><strong>${escapeHtml(c.label)}</strong> ${c.correct}/${c.asked}</li>`)
    .join("");
  const flag = run.provenance === "illustrative" ? `<p class="flag">Illustrative data — not a measurement.</p>` : "";
  const footer =
    run.provenance === "illustrative"
      ? "Example data from the web preview. It is not a measurement. Run the benchmark in the desktop app to measure this computer."
      : "Measured on this computer by Cerebro's benchmark lab. Speeds depend on the model, the quant, the hardware and everything else running at the time. Compare runs from the same machine.";
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cerebro benchmark · ${escapeHtml(run.model.name)}</title>
<style>
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; background: #f4f0e8; color: #24211d; font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px; background: #fffdf9; border: 1px solid #d8d0c3; border-radius: 18px; }
h1 { margin: 0 0 4px; font-size: 30px; letter-spacing: -0.02em; }
h2 { margin: 28px 0 10px; font-size: 19px; }
p { margin: 4px 0; }
.sub { display: block; color: #6b6358; font-size: 12px; font-weight: 400; }
.meta { color: #5a534a; }
.flag { margin-top: 10px; padding: 8px 12px; border: 1px solid #b8752d; border-radius: 8px; background: #fff4e2; color: #7a4a12; font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e6e0d6; text-align: left; vertical-align: top; }
thead th { color: #6b6358; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
tbody th { font-weight: 600; }
td { white-space: nowrap; }
.score { font-size: 22px; font-weight: 700; }
.categories { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 8px 0 0; padding: 0; list-style: none; color: #4d473f; }
.ok { color: #2e6b45; font-weight: 600; }
.no { color: #8b3a2f; font-weight: 600; }
footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid #e6e0d6; color: #7a7268; font-size: 12px; }
@media print { body { padding: 0; background: #fff; } main { border: 0; padding: 0; } }
</style>
</head>
<body>
<main>
<h1>Cerebro benchmark</h1>
<p class="meta">${escapeHtml(run.model.name)} · ${escapeHtml(run.model.quant)} · ${escapeHtml(run.model.params || "size not reported")}</p>
<p class="meta">${escapeHtml(new Date(run.at).toLocaleString())} · ${escapeHtml(statusLabel(run.status))} · ${run.provenance === "illustrative" ? "Illustrative data" : "Measured"}</p>
<p class="meta">${escapeHtml(machineLine(run))}</p>
<p class="meta">${escapeHtml(placementLine(run))}</p>
${flag}
<h2>Speed at three depths</h2>
<table>
<thead><tr><th scope="col">Depth</th><th scope="col">Prompt read</th><th scope="col">Reading speed</th><th scope="col">Writing speed</th><th scope="col">First token</th><th scope="col">Memory in use</th></tr></thead>
<tbody>
${rows}
</tbody>
</table>
<h2>Accuracy check</h2>
<p class="score">${run.accuracy.correct}/${run.accuracy.asked} correct</p>
<ul class="categories">${categories}</ul>
<p class="sub">Twenty fixed short questions, graded automatically against one exact answer each.</p>
<h2>Every question</h2>
<table>
<thead><tr><th scope="col">Question</th><th scope="col">Answer</th><th scope="col">Expected</th><th scope="col">Result</th></tr></thead>
<tbody>
${questionRows}
</tbody>
</table>
<footer>${escapeHtml(footer)}</footer>
</main>
</body>
</html>`;
}

export function reportFilename(run: BenchRun, ext: "png" | "html"): string {
  const date = run.at.slice(0, 10);
  const model = run.model.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "model";
  return `cerebro-benchmark-${model}-${date}.${ext}`;
}

const W = 1080;
const H = 1420;

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** Draws the PNG report card and returns its data: URL. */
export function drawReportCard(run: BenchRun): string {
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("The report card could not be drawn.");
  const ink = "#24211d";
  const muted = "#6b6358";
  const line = "#e3dcd1";
  const accent = "#8a4b24";
  const left = 64;
  const right = W - 64;

  ctx.fillStyle = "#f4f0e8";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#fffdf9";
  ctx.strokeStyle = "#d8d0c3";
  ctx.lineWidth = 2;
  ctx.fillRect(32, 32, W - 64, H - 64);
  ctx.strokeRect(32, 32, W - 64, H - 64);

  ctx.fillStyle = ink;
  ctx.font = "700 40px Segoe UI, system-ui, sans-serif";
  ctx.fillText("Cerebro benchmark", left, 112);
  ctx.font = "600 24px Segoe UI, system-ui, sans-serif";
  ctx.fillText(run.model.name.slice(0, 64), left, 156);
  ctx.font = "16px Segoe UI, system-ui, sans-serif";
  ctx.fillStyle = muted;
  ctx.fillText(`${run.model.quant} · ${run.model.params || "size not reported"} · ${statusLabel(run.status)} · ${run.provenance === "illustrative" ? "Illustrative data" : "Measured"}`, left, 186);
  ctx.fillText(new Date(run.at).toLocaleString(), left, 212);
  let y = 246;
  for (const l of wrap(ctx, machineLine(run), W - 128)) {
    ctx.fillText(l, left, y);
    y += 22;
  }
  for (const l of wrap(ctx, placementLine(run), W - 128)) {
    ctx.fillText(l, left, y);
    y += 22;
  }
  if (run.provenance === "illustrative") {
    ctx.fillStyle = "#7a4a12";
    ctx.fillRect(left, y + 4, 520, 36);
    ctx.fillStyle = "#fff4e2";
    ctx.font = "600 16px Segoe UI, system-ui, sans-serif";
    ctx.fillText("Illustrative data — not a measurement", left + 12, y + 28);
    y += 52;
  }

  // Speed table.
  y += 18;
  ctx.fillStyle = ink;
  ctx.font = "700 24px Segoe UI, system-ui, sans-serif";
  ctx.fillText("Speed at three depths", left, y);
  y += 22;
  const cols = [left, left + 250, left + 410, left + 570, left + 730, left + 880];
  ctx.font = "600 14px Segoe UI, system-ui, sans-serif";
  ctx.fillStyle = muted;
  ["DEPTH", "PROMPT READ", "READING", "WRITING", "FIRST TOKEN", "MEMORY"].forEach((h, i) => ctx.fillText(h, cols[i]!, y));
  y += 16;
  ctx.strokeStyle = line;
  ctx.beginPath();
  ctx.moveTo(left, y);
  ctx.lineTo(right, y);
  ctx.stroke();
  y += 30;
  for (const context of [0, 4096, 16384]) {
    const point = run.points.find((p) => p.context === context);
    ctx.fillStyle = ink;
    ctx.font = "600 17px Segoe UI, system-ui, sans-serif";
    ctx.fillText(contextLabel(context), cols[0]!, y);
    ctx.font = "15px Segoe UI, system-ui, sans-serif";
    if (!point || !point.ran) {
      ctx.fillStyle = muted;
      ctx.fillText("Not run", cols[1]!, y);
      const noteLines = wrap(ctx, point?.note ?? "Not measured.", 500);
      ctx.font = "13px Segoe UI, system-ui, sans-serif";
      noteLines.slice(0, 2).forEach((l, i) => ctx.fillText(l, cols[2]!, y + i * 17));
    } else {
      ctx.fillText(`${point.promptTokens.toLocaleString()} tokens`, cols[1]!, y);
      ctx.fillText(`${point.prefillTps.toFixed(1)} tok/s`, cols[2]!, y);
      ctx.fillText(`${point.decodeTps.toFixed(1)} tok/s`, cols[3]!, y);
      ctx.fillText(`${Math.round(point.ttftMs).toLocaleString()} ms`, cols[4]!, y);
      ctx.fillText(`${point.memoryUsedGiB.toFixed(2)} GiB`, cols[5]!, y);
    }
    y += 38;
    ctx.strokeStyle = line;
    ctx.beginPath();
    ctx.moveTo(left, y - 12);
    ctx.lineTo(right, y - 12);
    ctx.stroke();
  }

  // Accuracy.
  y += 10;
  ctx.fillStyle = ink;
  ctx.font = "700 24px Segoe UI, system-ui, sans-serif";
  ctx.fillText("Accuracy check", left, y);
  y += 36;
  ctx.font = "700 30px Segoe UI, system-ui, sans-serif";
  ctx.fillStyle = accent;
  ctx.fillText(`${run.accuracy.correct}/${run.accuracy.asked} correct`, left, y);
  ctx.fillStyle = muted;
  ctx.font = "15px Segoe UI, system-ui, sans-serif";
  ctx.fillText("Twenty fixed short questions, graded automatically.", left + 230, y - 4);
  y += 26;
  for (const c of categoryScores(run)) {
    ctx.fillStyle = ink;
    ctx.font = "15px Segoe UI, system-ui, sans-serif";
    ctx.fillText(`${c.label}: ${c.correct}/${c.asked}`, left, y);
    y += 22;
  }

  // A short list of the questions that were wrong, or the note that none were.
  y += 12;
  const wrong = run.accuracy.questions.filter((q) => !q.correct);
  ctx.fillStyle = ink;
  ctx.font = "700 20px Segoe UI, system-ui, sans-serif";
  ctx.fillText(wrong.length ? "Missed questions" : "Every question answered correctly", left, y);
  y += 28;
  ctx.font = "14px Segoe UI, system-ui, sans-serif";
  for (const q of wrong.slice(0, 6)) {
    ctx.fillStyle = muted;
    const lines = wrap(ctx, `${q.question}  Answer: ${q.answer || "no answer"} (expected ${q.expected})`, W - 128);
    for (const l of lines.slice(0, 2)) {
      ctx.fillText(l, left, y);
      y += 20;
    }
  }
  if (wrong.length > 6) {
    ctx.fillText(`…and ${wrong.length - 6} more in the HTML report.`, left, y);
  }
  ctx.fillStyle = muted;
  ctx.font = "13px Segoe UI, system-ui, sans-serif";
  const footer =
    run.provenance === "illustrative"
      ? "Example data from the web preview. It is not a measurement. Run the benchmark in the desktop app to measure this computer."
      : "Measured on this computer by Cerebro's benchmark lab. Compare runs from the same machine.";
  wrap(ctx, footer, W - 128)
    .slice(0, 2)
    .forEach((text, i) => ctx.fillText(text, left, H - 70 + i * 18));
  return canvas.toDataURL("image/png");
}
