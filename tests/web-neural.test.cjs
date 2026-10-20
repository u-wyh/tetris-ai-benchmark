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

test("old responses cannot act after mode switch or game restart", async () => {
  const g = prepare();
  g.run(`globalThis.pending = [];
    fetch = () => new Promise(resolve => pending.push(resolve));
    setMode("ppo"); resetGame({seed:12345});
    setMode("bc"); resetGame({seed:67890});`);
  assert.equal(g.run("pending.length"), 3);
  g.run(`pending[0]({ok:true,json:async()=>({actionId:21,source:"ppo",decisionMs:3})});
    pending[1]({ok:true,json:async()=>({actionId:21,source:"bc",decisionMs:3})});`);
  await flush();
  assert.equal(g.run("aiPlan.length"), 0);
  assert.equal(g.run("placedCount"), 0);
  g.run(`pending[2]({ok:true,json:async()=>({
    actionId:getLegalPlacements()[0].actionId,source:"bc",decisionMs:3
  })});`);
  await flush();
  assert.ok(g.run("aiPlan.length") > 0);
  assert.equal(g.run("aiMode"), "bc");
});

test("refresh restores a neural game after legal scripts finish loading", async () => {
  const g = prepare();
  g.run(`fetch = async () => ({ok:true,json:async()=>({
    actionId:getLegalPlacements()[0].actionId,source:"ppo",decisionMs:4
  })}); setMode("ppo"); resetGame({seed:12345});`);
  await flush();
  g.run(`while (placedCount === 0) updateTraditional(1000); saveGame()`);
  let restored;
  restored = game(g.storage, {
    fetch: async () => ({ok:true,json:async()=>({
      actionId:restored.run("getLegalPlacements()[0].actionId"),source:"ppo",decisionMs:4
    })}),
    AbortController, setTimeout, clearTimeout,
    location: {protocol:"http:",hostname:"100.91.227.47",search:""}
  });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(restored.run("state"), "playing");
  assert.equal(restored.run("aiMode"), "ppo");
  assert.equal(restored.run("placedCount"), 1);
  assert.equal(restored.run("aiPlanning || aiPlan.length > 0"), true);
});

test("inference timeout stops the AI and offers Retry", async () => {
  const g = prepare();
  g.run(`globalThis.timeoutCallback = null;
    setTimeout = callback => { timeoutCallback = callback; return 1; };
    fetch = (_, options) => new Promise((_, reject) => {
      options.signal.onabort = () => reject({name:"AbortError"});
    });
    AbortController = class {
      constructor() { this.signal = {onabort:null}; }
      abort() { this.signal.onabort?.(); }
    };
    setMode("ppo"); resetGame({seed:12345}); timeoutCallback();`);
  await flush();
  assert.equal(g.run("aiRequestFailed"), true);
  assert.equal(g.run("retryAIButton.hidden"), false);
  assert.match(g.run("aiStatus.textContent"), /请求超时/);
});
