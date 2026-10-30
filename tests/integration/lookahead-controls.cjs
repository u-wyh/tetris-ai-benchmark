"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {chromium}=require(process.env.PLAYWRIGHT_NODE_PATH||path.resolve(__dirname,"../../runs/model_based_planning_web/browser-runtime/node_modules/playwright"));
const origin=process.argv[2]||"http://100.91.227.47:8080",out=process.argv[3];
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const endpoint="**/api/decide";
async function ready(p){await p.waitForFunction(()=>aiPlan.length>0||aiRequestFailed);assert.equal(await p.evaluate(()=>aiRequestFailed),false)}
async function move(p){await ready(p);return p.evaluate(()=>{const a=aiExpectedAction,n=placedCount;for(let i=0;i<150&&placedCount===n;i++){updateEffects(1000);updateTraditional(1000)};if(placedCount!==n+1)throw Error("executor");return a})}
async function manual(b){
 const c=await b.newContext();await c.addInitScript(()=>{window.requestAnimationFrame=()=>0;window.setInterval=()=>0;window.Worker=undefined});
 const p=await c.newPage();await p.goto(origin);return {c,p};
}
(async()=>{
 const b=await chromium.launch({headless:true}),result={checks:{},origin};
 try{
  let {c,p}=await manual(b);
  assert.equal(await p.locator("#modeSelect").inputValue(),"hold");
  for(const mode of ["human","v1","hold","beam","ppo","bc","b12","lookahead"])assert.equal(await p.locator(`#modeSelect option[value="${mode}"]`).count(),1);
  await p.locator("#seedInput").fill("3102841010");await p.locator("#modeSelect").selectOption("lookahead");await p.locator("#restartButton").click();await ready(p);
  await p.locator("#pauseButton").click();const paused=await p.evaluate(()=>JSON.stringify({board,placedCount,lookaheadCounts}));await sleep(200);
  assert.equal(await p.evaluate(()=>state),"paused");assert.equal(await p.evaluate(()=>JSON.stringify({board,placedCount,lookaheadCounts})),paused);
  await p.locator("#pauseButton").click();await move(p);await p.evaluate(()=>saveGame());await p.reload();await ready(p);
  assert.equal(await p.evaluate(()=>placedCount),1);assert.equal(await p.evaluate(()=>lookaheadCounts.retained+lookaheadCounts.overrides),1);
  assert.equal(await p.locator("#modeSelect").inputValue(),"lookahead");result.checks.pause_resume_refresh=true;
  for(const mode of ["v1","hold","beam","ppo","bc","b12"]){await p.evaluate(mode=>{setMode("human");resetGame({seed:3102841011});setMode(mode)},mode);await move(p)}
  result.checks.previous_modes=true;await c.close();

  ({c,p}=await manual(b));let release,servedResolve;const served=new Promise(r=>servedResolve=r),gate=new Promise(r=>release=r);
  await p.route(endpoint,async route=>{const response=await route.fetch();servedResolve();await gate;try{await route.fulfill({response})}catch{}});
  await p.evaluate(()=>{resetGame({seed:3102841012});setMode("lookahead")});await served;
  await p.locator("#modeSelect").selectOption("human");release();await sleep(150);
  assert.equal(await p.evaluate(()=>placedCount),0);assert.equal(await p.evaluate(()=>aiPlan.length),0);
  assert.equal(await p.evaluate(()=>lookaheadCounts.retained+lookaheadCounts.overrides),0);result.checks.old_request_isolation=true;await c.close();

  ({c,p}=await manual(b));let calls=0;
  await p.route(endpoint,async route=>{calls++;await route.fetch();await route.fulfill({status:503,contentType:"application/json",body:JSON.stringify({error:"injected response failure"})})});
  await p.evaluate(()=>{resetGame({seed:3102841013});setMode("lookahead")});await p.waitForFunction(()=>aiRequestFailed);
  await p.evaluate(()=>{planTraditional();planTraditional()});assert.equal(calls,1);assert.equal(await p.locator("#retryAIButton").isVisible(),true);
  await p.unroute(endpoint);await p.locator("#retryAIButton").click();await ready(p);await move(p);result.checks.failure_retry=true;await c.close();

  ({c,p}=await manual(b));let releaseTimeout;const timeoutGate=new Promise(r=>releaseTimeout=r);
  await p.route(endpoint,async route=>{const response=await route.fetch();await timeoutGate;try{await route.fulfill({response})}catch{}});
  const timeoutStart=Date.now();await p.evaluate(()=>{resetGame({seed:3102841014});setMode("lookahead")});
  await p.waitForFunction(()=>aiRequestFailed,{},{timeout:15000});
  assert.match(await p.locator("#aiStatus").textContent(),/请求超时/);result.timeout_ms=Date.now()-timeoutStart;
  releaseTimeout();await p.unroute(endpoint);await p.locator("#retryAIButton").click();await ready(p);result.checks.timeout_retry=true;await c.close();

  const b2=await chromium.launch({headless:true});
  try{
   const clients=await Promise.all([manual(b),manual(b2)]);
   const seq=await Promise.all(clients.map(async({p})=>{await p.evaluate(()=>{resetGame({seed:3102841020});setMode("lookahead")});const actions=[];for(let i=0;i<64;i++)actions.push(await move(p));return {actions,counts:await p.evaluate(()=>({...lookaheadCounts})),pieces:await p.evaluate(()=>placedCount)}}));
   assert.deepEqual(seq[0].actions,seq[1].actions);assert.deepEqual(seq[0].counts,seq[1].counts);
   assert.equal(seq[0].pieces,64);result.concurrent={browsers:2,pieces_each:64,deterministic:true,counts:seq[0].counts};
   for(const client of clients)await client.c.close();result.checks.concurrent_browsers=true;
  }finally{await b2.close()}

  const motion=await b.newContext(),m=await motion.newPage();await m.goto(origin);await m.bringToFront();
  result.animation=[];
  for(const speed of [.25,8]){
   await m.evaluate(speed=>{setMode("human");resetGame({seed:3102841015});demoSpeed.value=String(speed);setMode("lookahead")},speed);
   await ready(m);const first=await m.evaluate(()=>aiExpectedAction);await sleep(3000);
   result.animation.push(await m.evaluate(({speed,first})=>({speed,first,pieces:placedCount,decision:aiDecisionInfo.textContent,playback:aiPlaybackInfo.textContent}),{speed,first}));
  }
  assert.equal(result.animation[0].first,result.animation[1].first);assert.ok(result.animation[1].pieces>result.animation[0].pieces);
  result.checks.animation_separate_from_decision=true;await motion.close();
 }finally{await b.close()}
 fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out+".tmp",JSON.stringify(result,null,2));fs.renameSync(out+".tmp",out);console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1});
