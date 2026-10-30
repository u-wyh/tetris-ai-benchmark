"use strict";
// Real Chromium + real HTTP + the production browser executor; never changes AI rules.
const fs = require("node:fs"), path = require("node:path");
const {execFileSync} = require("node:child_process");
const {performance} = require("node:perf_hooks");
const {chromium} = require(process.env.PLAYWRIGHT_NODE_PATH || path.resolve(__dirname,
  "../../runs/model_based_planning_web/browser-runtime/node_modules/playwright"));
const [capText, seedText, origin, directory, pidText, stagingPort, frozen] = process.argv.slice(2);
const cap = Number(capText), seed = Number(seedText), run = path.resolve(directory);
if (!Number.isInteger(cap) || cap < 1 || !Number.isInteger(seed) || !origin) throw Error("CAP SEED ORIGIN RUN PID [STAGING_PORT] [frozen]");
function atomic(name, data) {
  fs.mkdirSync(path.dirname(name), {recursive:true});
  fs.writeFileSync(name + ".tmp", JSON.stringify(data)); fs.renameSync(name + ".tmp", name);
}
function py(module, args) {return JSON.parse(execFileSync(".venv/bin/python", ["-m",module,...args],{encoding:"utf8"}));}
function usage() {
  if (!pidText || pidText === "0") return null;
  const raw = fs.readFileSync(`/proc/${pidText}/stat`,"utf8");
  const fields = raw.slice(raw.lastIndexOf(")")+2).trim().split(/\s+/);
  const rss = fs.readFileSync(`/proc/${pidText}/status`,"utf8").match(/VmRSS:\s+(\d+)/);
  return {ticks:Number(fields[11])+Number(fields[12]),rss_kb:Number(rss?.[1]),at:performance.now()};
}
function stats(values) {
  const v=values.filter(Number.isFinite).sort((a,b)=>a-b);
  const q=p=>v[Math.min(v.length-1,Math.ceil(p*v.length)-1)];
  return {count:v.length,mean:v.reduce((a,b)=>a+b,0)/v.length,p95:q(.95),p99:q(.99)};
}
async function install(context) {
  await context.addInitScript(({port})=>{
    // Accelerate only rendering/scheduling in this test. Execute the real BFS input path.
    window.requestAnimationFrame=()=>0;window.Worker=undefined;window.setInterval=()=>0;
    const realFetch=window.fetch.bind(window);window.__http=[];
    window.fetch=async(url,options)=>{
      if (port) url=url.replace(":8081/",`:${port}/`);
      const start=performance.now(),body=JSON.parse(options.body);
      const index=placedCount;
      const response=await realFetch(url,options),payload=await response.clone().json();
      window.__http.push({piece:index,mode:body.mode,legal_ids:body.legalActionIds,
        keys:Object.keys(body.observation),next:body.observation.next,
        http_ms:performance.now()-start,response:payload,status:response.status});
      return response;
    };
  },{port:stagingPort||null});
}
async function helper(page) {
  await page.evaluate(`window.__snapshot=()=>({...getPublicObservation(),placed:placedCount});
    window.__runChunk=async(count)=>{
      const rows=[];
      for(let i=0;i<count&&state!=="over";i++){
        const start=performance.now();
        while(!aiPlan.length&&!aiRequestFailed&&state!=="over"){
          if(performance.now()-start>15000)throw Error("decision stalled");
          await new Promise(resolve=>setTimeout(resolve,3));
        }
        if(aiRequestFailed)throw Error(aiStatus.textContent);
        if(state==="over")break;
        const before=__snapshot(),action=aiExpectedAction;
        const req=__http.findLast(r=>r.piece===placedCount);
        if(!req||req.status!==200||req.next.length!==3||req.keys.some(k=>["bag","rng","queue"].includes(k)))throw Error("public HTTP invariant failed");
        const begin=performance.now();
        for(let tick=0;tick<150&&placedCount===before.placed;tick++){updateEffects(1000);updateTraditional(1000)}
        if(placedCount!==before.placed+1)throw Error("executor stalled");
        if(lookaheadCounts.retained+lookaheadCounts.overrides!==placedCount)throw Error("counter drift");
        rows.push({before,after:__snapshot(),action,legal_ids:req.legal_ids,response:req.response,
          http_ms:req.http_ms,path_ms:performance.now()-begin});
      }
      saveGame();return {entries:rows,browser_save:localStorage.getItem("neonBlocksGameV1"),
        game_over:state==="over",counts:{...lookaheadCounts},display:lookaheadInfo.textContent};
    };`);
}
(async()=>{
  fs.mkdirSync(run,{recursive:true});
  const identity={cap,seed,origin,stagingPort:stagingPort||null,
    planner_sha256:"0c2dbea26708fed866c5089eb102672b0cfccafeaa9f130b58086c847e8ffb08"};
  const idFile=path.join(run,"identity.json");
  if(fs.existsSync(idFile)&&JSON.stringify(JSON.parse(fs.readFileSync(idFile)))!==JSON.stringify(identity))throw Error("resume identity changed");
  atomic(idFile,identity);
  const progress=py("training.evaluation.web_longrun_verify",["resume","--run",run,"--seed",String(seed)]);
  const browser=await chromium.launch({headless:true});
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1100}});await install(context);
    const page=await context.newPage(),errors=[];page.on("pageerror",e=>errors.push(String(e)));
    await page.goto(origin,{waitUntil:"load"});
    if(progress.browser_save){
      await page.evaluate(save=>localStorage.setItem("neonBlocksGameV1",save),progress.browser_save);await page.reload();
    } else await page.evaluate(seed=>{setMode("lookahead");resetGame({seed})},seed);
    await helper(page);
    let pieces=progress.pieces,index=progress.chunk;const start=performance.now(),cpuStart=usage();
    while(pieces<cap){
      const chunk=await page.evaluate(n=>__runChunk(n),Math.min(frozen?300:500,cap-pieces));
      if(!chunk.entries.length)break;
      const next=index+1,pending=path.join(run,`chunk_${String(next).padStart(6,"0")}.json`);
      const data={chunk:next,start_piece:pieces,end_piece:pieces+chunk.entries.length,...chunk,
        metrics:{http_ms:chunk.entries.map(r=>r.http_ms),server_ms:chunk.entries.map(r=>r.response.decisionMs),
          compute_ms:chunk.entries.map(r=>r.response.computeMs),queue_ms:chunk.entries.map(r=>r.response.queueMs),
          path_ms:chunk.entries.map(r=>r.path_ms)}};
      atomic(pending,data);
      py("training.evaluation.lookahead_web_verify",["--run",run,"--seed",String(seed),"--pending",pending,...(frozen?["--frozen"]:[])]);
      index=next;pieces=data.end_piece;
      atomic(path.join(run,"status.json"),{stage:"running",pieces,cap,seed,score:chunk.entries.at(-1).after.score});
      console.log(JSON.stringify({pieces,cap,seed}));if(chunk.game_over)break;
    }
    const end=usage(),hertz=Number(execFileSync("getconf",["CLK_TCK"],{encoding:"utf8"}));
    const rows=fs.readdirSync(run).filter(n=>/^chunk_\d+\.json$/.test(n)).sort().flatMap(n=>JSON.parse(fs.readFileSync(path.join(run,n))).entries);
    const result={...identity,pieces,right_censored:await page.evaluate(()=>state!=="over")&&pieces===cap,
      game_over:await page.evaluate(()=>state==="over"),core_checked_steps:rows.length,frozen_checked_steps:frozen?rows.length:0,
      metrics:{http_ms:stats(rows.map(r=>r.http_ms)),server_ms:stats(rows.map(r=>r.response.decisionMs)),
        compute_ms:stats(rows.map(r=>r.response.computeMs)),queue_ms:stats(rows.map(r=>r.response.queueMs)),path_ms:stats(rows.map(r=>r.path_ms))},
      server_cpu_percent:cpuStart&&end?100*(end.ticks-cpuStart.ticks)/hertz/((end.at-cpuStart.at)/1000):null,
      server_rss_kb:end?.rss_kb,wall_ms:performance.now()-start,resumed_from:progress.pieces,browser:browser.version(),errors,
      final:await page.evaluate(()=>({state:__snapshot(),counts:lookaheadCounts,text:lookaheadInfo.textContent,
        timing:aiDecisionInfo.textContent,animation:aiPlaybackInfo.textContent}))};
    await page.evaluate(()=>{drawBoard();drawNext();drawHold()});
    await page.screenshot({path:path.join(run,"page.png"),fullPage:true});
    atomic(path.join(run,"result.json"),result);atomic(path.join(run,"status.json"),{stage:"completed",pieces,cap,seed});
    if(errors.length)throw Error(errors.join("\n"));
    console.log(JSON.stringify(result));
  }finally{await browser.close()}
})().catch(e=>{atomic(path.join(run,"failure.json"),{error:String(e.stack||e)});console.error(e);process.exitCode=1});
