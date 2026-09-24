import { test, expect } from "@playwright/test";

/** Design round 5: honest outcomes, claim wording and the specimen's own rhythm. */

test.describe("claims: provenance", () => {
  test("no provenance row claims a default; the sampler is stated as values", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const dds = await page.locator(".stat-tile__provenance dd").allTextContents();
    expect(dds.length).toBeGreaterThan(0);
    expect(dds.filter((t) => /default/i.test(t))).toEqual([]);
    const samplers = await page.$$eval(".stat-tile__provenance dt", (dts) =>
      dts.filter((dt) => dt.textContent?.trim() === "sampler").map((dt) => dt.nextElementSibling?.textContent?.trim() ?? ""),
    );
    expect(samplers.length).toBeGreaterThan(0);
    for (const s of samplers) expect(s).toMatch(/temp \d/);
  });
});

test.describe("result-card outcomes", () => {
  test("the measuring card: no static aria-busy, a live status, a progressbar and the flag", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const card = page.locator(".result-card--measuring");
    await expect(card).toHaveCount(1);
    const info = await card.evaluate((el) => {
      // aria-busy is only ever set for the frame a batch of updates is applied (DESIGN.md); a static
      // one can make a screen reader hold back the step list for good.
      const busy = el.querySelector("[aria-busy]");
      const status = el.querySelector('[role="status"]');
      const bar = el.querySelector('[role="progressbar"]');
      return {
        busy: !!busy,
        statusText: status?.textContent?.trim() ?? "",
        // A live region inside a busy ancestor may never be announced.
        statusInsideBusy: !!status?.closest('[aria-busy="true"]'),
        progress: bar ? { now: bar.getAttribute("aria-valuenow"), text: bar.getAttribute("aria-valuetext") } : null,
        current: el.querySelectorAll('[aria-current="step"]').length,
        flag: el.querySelector("figure .stat-tile__flag")?.textContent?.trim(),
      };
    });
    expect(info.busy).toBe(false);
    expect(info.statusText.length).toBeGreaterThan(0);
    expect(info.statusInsideBusy).toBe(false);
    expect(info.progress?.now).toBe("55");
    expect(info.progress?.text).toMatch(/Step 2 of 3/);
    expect(info.current).toBe(1);
    expect(info.flag).toBe("Illustrative data — not a measurement");
  });

  test("the progress fill shows its value at once in full motion and never animates under reduced motion", async ({ browser }) => {
    for (const reducedMotion of ["no-preference", "reduce"] as const) {
      const context = await browser.newContext({ reducedMotion });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        const r = await page.locator(".tune-progress .bar").evaluate((el) => ({
          share: ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el) / el.parentElement!.getBoundingClientRect().width,
          armed: el.hasAttribute("data-motion-armed"),
          duration: getComputedStyle(el).transitionDuration,
        }));
        expect(r.armed, reducedMotion).toBe(false);
        expect(r.share, reducedMotion).toBeGreaterThan(0.54);
        expect(r.share, reducedMotion).toBeLessThan(0.56);
        if (reducedMotion === "reduce") expect(r.duration.split(",").every((d) => parseFloat(d) === 0)).toBe(true);
      } finally {
        await context.close();
      }
    }
  });

  test("the defaults-won card: neutral chip (never the success colour), near-equal bars, the flag and a reason", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.locator(".result-card--defaults-won").scrollIntoViewIfNeeded();
    await page.waitForTimeout(1200);
    const info = await page.locator(".result-card--defaults-won").evaluate((el) => {
      const chip = el.querySelector<HTMLElement>(".result-card__status")!;
      const probe = document.createElement("span");
      probe.style.setProperty("color", "var(--color-success)");
      chip.append(probe);
      const success = getComputedStyle(probe).color;
      probe.remove();
      const bars = Array.from(el.querySelectorAll(".bar")).map((b) => ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(b));
      return {
        text: chip.textContent?.trim(),
        neutral: chip.classList.contains("result-card__status--neutral"),
        color: getComputedStyle(chip).color,
        success,
        bars,
        flag: el.querySelector("figure .stat-tile__flag")?.textContent?.trim(),
        note: el.querySelector(".result-card__note")?.textContent?.trim() ?? "",
      };
    });
    expect(info.text).toBe("Defaults were fastest");
    expect(info.neutral).toBe(true);
    expect(info.color).not.toBe(info.success);
    expect(info.bars).toHaveLength(2);
    expect(info.bars[0]!).toBeGreaterThan(0);
    expect(info.bars[1]! / info.bars[0]!).toBeGreaterThan(0.9);
    expect(info.bars[1]! / info.bars[0]!).toBeLessThanOrEqual(1);
    expect(info.flag).toBe("Illustrative data — not a measurement");
    expect(info.note.length).toBeGreaterThan(20);
  });
});

test.describe("specimen rhythm", () => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
  ]) {
    test(`z-stack cards step in by one constant amount at ${viewport.width}px`, async ({ browser }) => {
      const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        const lefts = await page.$$eval(".z-stack__card", (els) => els.map((el) => el.getBoundingClientRect().left));
        expect(lefts).toHaveLength(9);
        const steps = lefts.slice(1).map((l, i) => l - lefts[i]!);
        expect(steps[0]!, `steps: ${steps.join(", ")}`).toBeGreaterThanOrEqual(4);
        for (const s of steps) expect(Math.abs(s - steps[0]!), `steps: ${steps.join(", ")}`).toBeLessThanOrEqual(1);
      } finally {
        await context.close();
      }
    });
  }

  test("a wrapped field error keeps its icon on the first line", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const r = await page.locator(".field__error").evaluate((el) => {
        const icon = el.querySelector("svg")!.getBoundingClientRect();
        const range = document.createRange();
        const text = Array.from(el.childNodes).find((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim())!;
        range.selectNodeContents(text);
        const lines = Array.from(range.getClientRects()).filter((r) => r.width > 1);
        const first = lines[0]!;
        return { lines: new Set(lines.map((l) => Math.round(l.top))).size, iconMid: icon.top + icon.height / 2, lineMid: first.top + first.height / 2 };
      });
      expect(r.lines, "the message should wrap at this width for the check to mean anything").toBeGreaterThan(1);
      expect(Math.abs(r.iconMid - r.lineMid)).toBeLessThanOrEqual(2);
    } finally {
      await context.close();
    }
  });

  test("no token in a type-sample label breaks across lines on a phone", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const broken = await page.$$eval(".type-sample__label .tok", (els) => els.filter((el) => el.getClientRects().length > 1).map((el) => el.textContent));
      expect(await page.locator(".type-sample__label .tok").count()).toBeGreaterThan(10);
      expect(broken).toEqual([]);
    } finally {
      await context.close();
    }
  });

  test("every swatch group fills its rows evenly at 1440 and 390px", async ({ browser }) => {
    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 390, height: 844 },
    ]) {
      const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        const ragged = await page.$$eval(".grid--swatches", (grids) =>
          grids
            .map((g) => {
              const cols = getComputedStyle(g).gridTemplateColumns.split(" ").length;
              return { n: g.children.length, cols };
            })
            .filter((g) => g.n % g.cols !== 0),
        );
        expect(ragged, `${viewport.width}px`).toEqual([]);
      } finally {
        await context.close();
      }
    }
  });

  test("the install block shows all four real commands and stays a readable width", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const r = await page.locator("#install").evaluate((el) => ({
        text: el.querySelector("pre")?.textContent ?? "",
        width: el.getBoundingClientRect().width,
        rem: parseFloat(getComputedStyle(document.documentElement).fontSize),
      }));
      for (const cmd of ["cerebro models", "cerebro tune ", "cerebro serve ", "cerebro chat "]) expect(r.text).toContain(cmd);
      expect(r.width).toBeLessThanOrEqual(44 * r.rem + 1);
    } finally {
      await context.close();
    }
  });

  test("the hero CTA is confident, not loud: 48px tall with body-size text, glow kept", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const r = await page.locator(".btn--hero").evaluate((el) => {
      const probe = document.createElement("span");
      probe.style.setProperty("font-size", "var(--text-body)");
      document.body.append(probe);
      const body = getComputedStyle(probe).fontSize;
      probe.remove();
      const cs = getComputedStyle(el);
      return { height: el.getBoundingClientRect().height, font: cs.fontSize, body, shadow: cs.boxShadow };
    });
    expect(Math.round(r.height)).toBe(48);
    expect(r.font).toBe(r.body);
    expect(r.shadow).not.toBe("none");
  });

  test("result cards side by side: the shorter card's slack never opens a gap under its head, and action rows align", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const cards = await page.$$eval(".result-pair", (pairs) =>
        pairs.map((pair) =>
          Array.from(pair.querySelectorAll<HTMLElement>(":scope > .result-card")).map((card) => {
            // The head's content, not the header box: a grid item stretches with its row, so the
            // header's own bottom would move down with any slack and hide the gap.
            const head = card.querySelector(".result-card__head > :last-child")!.getBoundingClientRect();
            const fig = card.querySelector(".result-card__figure")!.getBoundingClientRect();
            const figEl = card.querySelector<HTMLElement>(".result-card__figure")!;
            const firstInFig = (figEl.firstElementChild as HTMLElement).getBoundingClientRect();
            const actions = card.querySelector(".result-card__actions")!.getBoundingClientRect();
            const cs = getComputedStyle(card);
            const figCs = getComputedStyle(figEl);
            return {
              headToFig: fig.top - head.bottom,
              gap: parseFloat(cs.rowGap),
              figPadTop: firstInFig.top - fig.top,
              expectedPad: parseFloat(figCs.paddingTop) + parseFloat(figCs.borderTopWidth),
              actionsBottomGap: card.getBoundingClientRect().bottom - actions.bottom,
            };
          }),
        ),
      );
      expect(cards.length).toBeGreaterThanOrEqual(2);
      for (const pair of cards) {
        expect(pair).toHaveLength(2);
        for (const c of pair) {
          expect(Math.abs(c.headToFig - c.gap), `head-to-figure ${c.headToFig}px vs gap ${c.gap}px`).toBeLessThanOrEqual(1);
          expect(Math.abs(c.figPadTop - c.expectedPad), "figure content starts at its padding, not spread down").toBeLessThanOrEqual(1);
        }
        expect(Math.abs(pair[0]!.actionsBottomGap - pair[1]!.actionsBottomGap), "action rows share a baseline").toBeLessThanOrEqual(1);
      }
    } finally {
      await context.close();
    }
  });

  test("the phone pipeline connector stays on the circle centres even when measured mid pop-in", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await page.locator(".pipeline-card").scrollIntoViewIfNeeded();
      // Hold every node at its pop-in start transform, then make the specimen re-measure.
      // (CSSOM writes: the specimen's CSP rightly refuses an injected <style>.)
      await page.evaluate(() => {
        for (const n of Array.from(document.querySelectorAll<HTMLElement>(".pipeline__node"))) {
          n.style.setProperty("transition", "none", "important");
          n.style.setProperty("transform", "scale(0.92)", "important");
        }
        window.dispatchEvent(new Event("resize"));
      });
      await page.waitForTimeout(100);
      const r = await page.evaluate(() => {
        const line = document.querySelector(".pipeline__connector--v")!.getBoundingClientRect();
        document.querySelectorAll<HTMLElement>(".pipeline__node").forEach((n) => n.style.setProperty("transform", "none", "important"));
        const circles = Array.from(document.querySelectorAll(".pipeline__n")).map((n) => n.getBoundingClientRect());
        return {
          lineX: line.left + line.width / 2,
          top: line.top,
          bottom: line.bottom,
          xs: circles.map((c) => c.left + c.width / 2),
          firstY: circles[0]!.top + circles[0]!.height / 2,
          lastY: circles[circles.length - 1]!.top + circles[circles.length - 1]!.height / 2,
        };
      });
      for (const x of r.xs) expect(Math.abs(r.lineX - x), `connector x ${r.lineX} vs circle ${x}`).toBeLessThanOrEqual(1);
      expect(Math.abs(r.top - r.firstY)).toBeLessThanOrEqual(1);
      expect(Math.abs(r.bottom - r.lastY)).toBeLessThanOrEqual(1);
    } finally {
      await context.close();
    }
  });

  test("a link styled as a button is exactly as tall as a <button> of the same variant", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const link = await page.locator(".hero__actions a.btn--tertiary").evaluate((el) => el.getBoundingClientRect().height);
    const button = await page.locator("#buttons button.btn--tertiary").first().evaluate((el) => el.getBoundingClientRect().height);
    expect(Math.abs(link - button), `link ${link}px vs button ${button}px`).toBeLessThanOrEqual(0.5);
  });
});
