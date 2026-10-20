"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { game } = require("./game-harness.cjs");

function prepare() {
  const g = game();
  g.run(`location.protocol = "http:"; location.hostname = "100.91.227.47";
    setTimeout = () => 1; clearTimeout = () => {};
    AbortController = class { constructor() { this.signal = {}; } abort() { this.signal.aborted = true; } };`);
  return g;
}

async function flush() {
  await new Promise(resolve => setImmediate(resolve));
}

test("neural mode sends only public state and executes a legal browser path", async () => {
  const g = prepare();
  g.run(`fetch = async (url, options) => {
    globalThis.lastRequest = {url, body: JSON.parse(options.body)};
    return {ok: true, json: async () => ({actionId: getLegalPlacements()[0].actionId,
      source: "ppo", decisionMs: 5.2})};
  }; setMode("ppo"); resetGame({seed: 12345});`);
  await flush();
  const result = g.run(`({request: globalThis.lastRequest, expected: aiExpectedAction,
    planned: aiPlan.length, public: getPublicObservation()})`);
  assert.equal(result.request.url, "http://100.91.227.47:8081/api/decide");
  assert.equal(result.request.body.mode, "ppo");
  assert.equal(result.request.body.legalActionIds.length, 34);
  assert.deepEqual(Object.keys(result.request.body.observation), Object.keys(result.public));
  assert.equal(result.request.body.observation.next.length, 3);
  assert.ok(!("bag" in result.request.body.observation));
  assert.ok(result.request.body.legalActionIds.includes(result.expected));
  assert.ok(result.planned > 0);
});

test("pause invalidates a pending model response", async () => {
  const g = prepare();
  g.run(`fetch = () => new Promise(resolve => { globalThis.finishRequest = resolve; });
    setMode("b12"); resetGame({seed: 12345}); togglePause();
    finishRequest({ok: true, json: async () => ({actionId: 21, source: "bc", decisionMs: 4})});`);
  await flush();
  const result = g.run(`({state, planning: aiPlanning, planned: aiPlan.length,
    ppo: hybridCounts.ppo, bc: hybridCounts.bc})`);
  assert.equal(result.state, "paused");
  assert.equal(result.planning, false);
  assert.equal(result.planned, 0);
  assert.equal(result.ppo + result.bc, 0);
});

test("B12 displays its selected expert and keeps adoption counts", async () => {
  const g = prepare();
  g.run(`fetch = async () => ({ok: true, json: async () => ({
    actionId: getLegalPlacements()[0].actionId, source: "bc", decisionMs: 6.25
  })}); setMode("b12"); resetGame({seed:12345});`);
  await flush();
  const result = g.run(`({counts:hybridCounts, text:hybridInfo.textContent,
    timing:aiDecisionInfo.textContent, planned:aiPlan.length})`);
  assert.equal(result.counts.bc, 1);
  assert.equal(result.counts.ppo, 0);
  assert.match(result.text, /BC 1/);
  assert.match(result.timing, /6.3 ms/);
  assert.ok(result.planned > 0);
});

test("failed inference pauses automatic requests until Retry AI is clicked", async () => {
  const g = prepare();
  g.run(`globalThis.requests = 0;
    fetch = async () => {
      requests++;
      if (requests === 1) throw new Error("service unavailable");
      return {ok: true, json: async () => ({
        actionId: getLegalPlacements()[0].actionId, source: "ppo", decisionMs: 4
      })};
    }; setMode("ppo"); resetGame({seed:12345});`);
  await flush();
  assert.equal(g.run("aiRequestFailed"), true);
  assert.equal(g.run("retryAIButton.hidden"), false);
  g.run("planTraditional(); planTraditional();");
  assert.equal(g.run("requests"), 1);
  g.run('document.querySelector("#retryAIButton").click()');
  await flush();
  assert.equal(g.run("requests"), 2);
  assert.equal(g.run("aiRequestFailed"), false);
  assert.ok(g.run("aiPlan.length") > 0);
});
