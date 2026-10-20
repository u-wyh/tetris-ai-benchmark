"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { performance } = require("node:perf_hooks");
const { game } = require("../game-harness.cjs");

const [mode, capText, seedText, host, runName] = process.argv.slice(2);
const cap = Number(capText), seed = Number(seedText);
if (!["ppo", "bc", "b12"].includes(mode) || !Number.isInteger(cap) || cap < 1 ||
    !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff || !host || !runName) {
  throw new Error("Usage: node web-ai-longrun.cjs MODE CAP SEED TAILSCALE_IP RUN_DIRECTORY");
}
const run = path.resolve(runName);
const origin = `http://${host}:8080`;
const verifier = "training.evaluation.web_longrun_verify";
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const requests = [];
let failedRequests = 0;

function python(command, extra = []) {
  return JSON.parse(execFileSync(".venv/bin/python", ["-m", verifier, command,
    "--run", run, "--seed", String(seed), ...extra], {encoding: "utf8"}));
}

function atomicJson(name, value) {
  fs.mkdirSync(path.dirname(name), {recursive: true});
  const temporary = `${name}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(value));
  fs.renameSync(temporary, name);
}

function serverUsage() {
  const pid = process.env.TETRIS_WEB_AI_PID;
  if (!pid) return null;
  try {
    const [cpu, rss] = execFileSync("ps", ["-p", pid, "-o", "%cpu=,rss="],
      {encoding: "utf8"}).trim().split(/\s+/).map(Number);
    return {cpu_percent: cpu, rss_kb: rss};
  } catch (_) {
    return null;
  }
}

function snapshot(g) {
  return g.run(`(() => { const o = getPublicObservation();
    return {board:o.board, currentPiece:o.currentPiece, next:o.next, hold:o.hold,
      score:o.score, level:o.level, lines:o.lines, gameOver:o.gameOver,
      placed:placedCount}; })()`);
}

async function measuredFetch(url, options) {
  const started = performance.now();
  const body = JSON.parse(options.body);
  try {
    const response = await fetch(url, {
      ...options, headers: {...options.headers, Origin: origin}
    });
    const payload = await response.json();
    const record = {legal_ids: body.legalActionIds, server_ms: payload.decisionMs,
      http_ms: performance.now() - started, source: payload.source,
      status: response.status};
    requests.push(record);
    if (!response.ok) failedRequests++;
    return {ok: response.ok, json: async () => payload};
  } catch (error) {
    failedRequests++;
    throw error;
  }
}

async function main() {
  fs.mkdirSync(run, {recursive: true});
  const progress = python("resume");
  const storage = new Map();
  if (progress.browser_save) storage.set("neonBlocksGameV1", progress.browser_save);
  const g = game(storage, {fetch: measuredFetch, AbortController, setTimeout, clearTimeout,
    performance: {now: () => performance.now()},
    location: {protocol: "http:", hostname: host, search: ""}});
  g.run(`globalThis.bfsElapsedMs = 0;
    const originalPlacements = getLegalPlacements;
    getLegalPlacements = (...args) => {
      const begin = performance.now(); const result = originalPlacements(...args);
      bfsElapsedMs += performance.now() - begin; return result;
    };`);
  if (!progress.browser_save) g.run(`setMode(${JSON.stringify(mode)}); resetGame({seed:${seed}})`);
  if (g.run("aiMode") !== mode || g.run("placedCount") !== progress.pieces) {
    throw new Error("Browser resume mode or piece count changed");
  }
  let index = progress.chunk, pieces = progress.pieces;
  const started = performance.now();
  while (pieces < cap && g.run("state") !== "over") {
    const entries = [];
    const chunkStarted = performance.now();
    const cpuStart = process.cpuUsage();
    const failStart = failedRequests;
    let lastBfs = g.run("bfsElapsedMs");
    while (entries.length < 500 && pieces < cap && g.run("state") !== "over") {
      let ready = false;
      for (let wait = 0; wait < 750; wait++) {
        ready = g.run("aiPlan.length > 0 || aiRequestFailed || state === 'over'");
        if (ready) break;
        await sleep(20);
      }
      if (!ready || g.run("aiRequestFailed")) throw new Error(
        `HTTP decision failed at ${pieces}: ${g.run("aiStatus.textContent")}`);
      if (g.run("state") === "over") break;
      const before = snapshot(g), action = g.run("aiExpectedAction");
      const request = requests.at(-1);
      if (!request || request.status !== 200 || !request.legal_ids.includes(action)) {
        throw new Error(`HTTP Action ID invalid at ${pieces}`);
      }
      const initial = g.run("placedCount");
      const pathStarted = performance.now();
      for (let tick = 0; tick < 150 && g.run("placedCount") === initial; tick++) {
        g.run("updateEffects(1000); updateTraditional(1000)");
      }
      if (g.run("placedCount") !== initial + 1) throw new Error(`Browser BFS path stalled at ${pieces}`);
      const after = snapshot(g);
      const bfsNow = g.run("bfsElapsedMs");
      entries.push({before, after, action, legal_ids: request.legal_ids,
        source: request.source, server_ms: request.server_ms, http_ms: request.http_ms,
        bfs_ms: bfsNow - lastBfs, path_ms: performance.now() - pathStarted});
      lastBfs = bfsNow;
      pieces++;
    }
    if (!entries.length) break;
    g.run("saveGame()");
    const next = index + 1;
    const cpu = process.cpuUsage(cpuStart);
    const pending = path.join(run, `chunk_${String(next).padStart(6, "0")}.pending.json`);
    atomicJson(pending, {chunk: next, start_piece: pieces - entries.length,
      end_piece: pieces, entries, browser_save: storage.get("neonBlocksGameV1"),
      game_over: g.run("state") === "over",
      metrics: {server_ms: entries.map(x => x.server_ms),
        http_ms: entries.map(x => x.http_ms), bfs_ms: entries.map(x => x.bfs_ms),
        path_ms: entries.map(x => x.path_ms), failed_requests: failedRequests - failStart,
        source_counts: {ppo: entries.filter(x => x.source === "ppo").length,
          bc: entries.filter(x => x.source === "bc").length},
        wall_ms: performance.now() - chunkStarted,
        node_cpu_ms: (cpu.user + cpu.system) / 1000,
        node_rss_bytes: process.memoryUsage().rss,
        server_usage: serverUsage()}});
    const committed = python("verify", ["--pending", pending]);
    fs.unlinkSync(pending);
    index = next;
    atomicJson(path.join(run, "status.json"), {stage: "running", mode, seed,
      cap, chunk: index, pieces, game_over: committed.game_over,
      score: committed.score, elapsed_ms: performance.now() - started});
  }
  const result = {mode, seed, cap, pieces, game_over: g.run("state") === "over",
    truncated: g.run("state") !== "over" && pieces === cap, chunks: index,
    failed_requests: failedRequests, resumed_from_piece: progress.pieces,
    wall_ms: performance.now() - started};
  atomicJson(path.join(run, "result.json"), result);
  atomicJson(path.join(run, "status.json"), {stage: "completed", ...result});
  const oldFailure = path.join(run, "failure.json");
  if (fs.existsSync(oldFailure)) fs.unlinkSync(oldFailure);
  process.stdout.write(JSON.stringify(result) + "\n");
}

main().catch(error => {
  atomicJson(path.join(run, "failure.json"), {at: new Date().toISOString(),
    error: String(error.stack || error)});
  console.error(error);
  process.exitCode = 1;
});
