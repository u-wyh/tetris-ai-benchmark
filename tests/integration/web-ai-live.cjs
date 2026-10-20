"use strict";

const { game } = require("../game-harness.cjs");

const mode = process.argv[2] || "ppo";
const count = Number(process.argv[3] || 20);
const host = process.argv[4] || "127.0.0.1";
const origin = `http://${host}:8080`;
const g = game(new Map(), {
  fetch: (url, options) => fetch(url, {
    ...options, headers: { ...options.headers, Origin: origin }
  }),
  AbortController, setTimeout, clearTimeout,
  location: { protocol: "http:", hostname: host, search: "" }
});
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const snapshot = () => g.run(`(() => {
  const o = getPublicObservation();
  return {board:o.board, currentPiece:o.currentPiece, next:o.next, hold:o.hold,
    score:o.score, level:o.level, lines:o.lines, gameOver:o.gameOver,
    placed:placedCount, legal:getLegalPlacements().map(p=>p.actionId)};
})()`);

(async () => {
  g.run(`setMode(${JSON.stringify(mode)}); resetGame({seed:12345})`);
  const states = [snapshot()], actions = [], sources = [];
  for (let i = 0; i < count; i++) {
    let ready = false;
    for (let wait = 0; wait < 100; wait++) {
      ready = g.run("aiPlan.length > 0 || aiRequestFailed || state === 'over'");
      if (ready) break;
      await sleep(20);
    }
    if (!ready || g.run("aiRequestFailed")) throw new Error(`AI decision stalled at ${i}: ${g.run("aiStatus.textContent")}`);
    if (g.run("state") === "over") break;
    actions.push(g.run("aiExpectedAction"));
    sources.push(g.run("aiStatus.textContent"));
    const previous = g.run("placedCount");
    for (let tick = 0; tick < 100 && g.run("placedCount") === previous; tick++) {
      g.run("updateTraditional(1000)");
    }
    if (g.run("placedCount") === previous) throw new Error(`Browser path stalled at ${i}`);
    states.push(snapshot());
    if (g.run("state") === "over") break;
  }
  process.stdout.write(JSON.stringify({ mode, actions, sources, states }));
})().catch(error => { process.stderr.write(String(error.stack || error)); process.exitCode = 1; });
