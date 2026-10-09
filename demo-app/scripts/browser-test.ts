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
  const homeTitle = await page.evaluate(
    () => document.querySelector(".home-title")?.textContent ?? "",
  );
  if (!homeTitle.includes("Hangul")) throw new Error(`home title missing: ${homeTitle}`);
  const postLinks = await page.evaluate(() =>
    Array.from(document.querySelectorAll(".read-more-links a")).map((a) =>
      a.getAttribute("href"),
    ),
  );
  if (!postLinks.includes("/blog/hangul-day")) {
    throw new Error(`home should link the post, got ${JSON.stringify(postLinks)}`);
  }
  console.log(`home rendered: "${homeTitle}", links ${postLinks.join(" ")}`);
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
    const btn = Array.from(document.querySelectorAll("nav button")).find((b) =>
      b.textContent?.includes("한국어"),
    );
    (btn as HTMLButtonElement).click();
  });
  await page.waitForFunction(
    () => document.querySelector("button.primary")?.textContent?.includes("쓰기"),
    { timeout: 5000 },
  );
  console.log("language toggle OK");

  await page.screenshot({
    path: `${process.env.SCRATCH ?? "/tmp"}/app-generated.png`,
  });

  // The post page: article around the embedded demo, with its own meta.
  await page.goto(`${url}/blog/hangul-day`, { waitUntil: "networkidle0" });
  const post = await page.evaluate(() => ({
    title: document.querySelector(".post-title")?.textContent ?? "",
    prose: document.querySelectorAll(".prose").length,
    demo: !!document.querySelector(".demo"),
    ogType: document
      .querySelector('meta[property="og:type"]')
      ?.getAttribute("content"),
    docTitle: document.title,
  }));
  if (!post.title.includes("Hangul")) throw new Error(`post title missing: ${post.title}`);
  if (post.prose < 2 || !post.demo) throw new Error(`post layout wrong: ${JSON.stringify(post)}`);
  if (post.ogType !== "article" || !post.docTitle.startsWith("A neural network")) {
    throw new Error(`post meta not prerendered: ${JSON.stringify(post)}`);
  }
  console.log(`post rendered: "${post.title}" (${post.prose} prose blocks, og:type=${post.ogType})`);

  if (errors.length) {
    console.error("console errors:", errors);
    process.exit(1);
  }
  console.log("BROWSER TEST OK");
} finally {
  await browser.close();
}
