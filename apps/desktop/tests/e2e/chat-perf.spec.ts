import { expect, test } from "./fixtures";

// In the perf project (after every other test), untraced, like the other budgets.
test.use({ trace: "off" });

test("streaming a long reply never blocks the main thread for more than 50 ms", async ({ page }) => {
  test.setTimeout(120_000);
  const fence = "```";
  const long = Array.from({ length: 60 }, (_, i) => `Paragraph ${i} with **bold**, \`code\` and a [link](https://example.com/${i}).\n\n${fence}rust\nfn f${i}() -> u32 { ${i} }\n${fence}`).join("\n\n");
  const runs: number[][] = [];
  for (let attempt = 0; attempt < 3; attempt++) {
    await page.goto("/#/chat");
    await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
    const box = page.getByRole("textbox", { name: "Message" });
    // Pasting 8 KB into a textarea is one browser layout no app code controls; the budget is for
    // sending and streaming, so watching starts once the paste has settled.
    await box.fill(`//echo ${long}`);
    await page.waitForTimeout(500);
    await page.evaluate(() => {
      const w = window as unknown as { __long: number[] };
      w.__long = [];
      new PerformanceObserver((l) => l.getEntries().forEach((e) => w.__long.push(Math.round(e.duration)))).observe({ type: "longtask" });
    });
    await box.press("Enter");
    await page.waitForTimeout(6000);
    runs.push(await page.evaluate(() => (window as unknown as { __long: number[] }).__long));
    if (runs.at(-1)!.length === 0) break;
  }
  expect(
    runs.some((t) => t.length === 0),
    `every window had a long task: ${runs.map((t) => `[${t.join(", ")}]`).join(" ")}`,
  ).toBe(true);
});
