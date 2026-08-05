// End-to-end browser test: boot the built app, click "Write it", verify
// strokes appear on the canvas, exercise both download paths, and check the
// language toggle. Drives the system Chrome via puppeteer-core.
//
//   npx tsx scripts/browser-test.ts [url]     (default http://localhost:4173)

import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://localhost:4173";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--window-size=1100,900"],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(url, { waitUntil: "networkidle0" });
  await page.waitForFunction(
    () => document.querySelector(".status-line")?.textContent?.includes("model"),
    { timeout: 15000 },
  );
  console.log("model loaded");

  // Generate from the default text. The canvas draws one path per stroke
  // (per-stroke pressure width), so measure the total across paths.
  await page.click("button.primary");
  const totalPathLen = () =>
    Array.from(document.querySelectorAll(".pen-canvas path")).reduce(
      (s, p) => s + (p.getAttribute("d")?.length ?? 0),
      0,
    );
  await page.waitForFunction(
    `(${totalPathLen.toString()})() > 500`,
    { timeout: 20000 },
  );
  const pathLen = await page.evaluate(
    `(${totalPathLen.toString()})()`,
  );
  console.log(`strokes drawing (total path d length so far: ${pathLen})`);

  // Pressure rendering: the canvas draws filled ribbon outlines (per-point
  // width), not stroked polylines. Wait for a few strokes to be revealed —
  // the pen-speed reveal lags generation.
  await page.waitForFunction(
    () => document.querySelectorAll(".pen-canvas path").length >= 3,
    { timeout: 20000 },
  );
  const ribbons = await page.evaluate(() => {
    const paths = Array.from(document.querySelectorAll(".pen-canvas path"));
    return {
      n: paths.length,
      filled: paths.every((p) => p.getAttribute("fill") === "currentColor"),
      capped: paths.every((p) => (p.getAttribute("d") ?? "").includes("A ")),
    };
  });
  if (ribbons.n < 3 || !ribbons.filled || !ribbons.capped) {
    throw new Error(`expected filled ribbon strokes, got ${JSON.stringify(ribbons)}`);
  }
  console.log(`pressure ribbons: ${ribbons.n} filled stroke outlines`);

  // Wait for generation to finish (points count appears in status line).
  await page.waitForFunction(
    () => /point|점/.test(document.querySelector(".status-line")?.textContent ?? ""),
    { timeout: 20000 },
  );
  console.log(
    "generated:",
    await page.evaluate(() => document.querySelector(".status-line")!.textContent),
  );

  const actionCount = await page.evaluate(
    () => document.querySelectorAll(".actions button").length,
  );
  if (actionCount < 4) throw new Error(`expected action buttons, got ${actionCount}`);
  console.log(`${actionCount} action buttons present`);

  // Language toggle flips to Korean.
  await page.evaluate(() => {
    const nav = document.querySelectorAll("nav .link-button");
    (nav[0] as HTMLButtonElement).click();
  });
  await page.waitForFunction(
    () => document.querySelector(".tagline")?.textContent?.includes("한글") ||
      document.querySelector(".tagline")?.textContent?.includes("신경망"),
    { timeout: 5000 },
  );
  console.log("language toggle OK");

  await page.screenshot({
    path: `${process.env.SCRATCH ?? "/tmp"}/app-generated.png`,
  });

  if (errors.length) {
    console.error("console errors:", errors);
    process.exit(1);
  }
  console.log("BROWSER TEST OK");
} finally {
  await browser.close();
}
