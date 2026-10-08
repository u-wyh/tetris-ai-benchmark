const fs = require("node:fs");
const { game } = require("../game-harness.cjs");
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const g = game();
g.run(`resetGame({seed:${request.seed}})`);
const states = [];
const actions = [];
function snapshot() {
  return JSON.parse(g.run(`JSON.stringify((() => {
    const o = getPublicObservation();
    return {board:o.board,currentPiece:o.currentPiece,hold:o.hold,
      holdAvailable:o.holdAvailable,next:o.next,score:o.score,lines:o.lines,
      level:o.level,phase:o.phase,gameOver:o.gameOver,placedCount};
  })())`));
}
states.push(snapshot());
for (let i = 0; i < request.steps && !states.at(-1).gameOver; i++) {
  let action;
  if (request.mode === "v1") {
    const choice = g.run("traditionalChooseV1(getPublicObservation(), aiRngState)");
    action = choice.actionId;
    g.run(`aiRngState = ${choice.rngState}`);
  } else {
    action = g.run(`traditionalChooseV2(getPublicObservation(), "${request.mode}")`);
  }
  const placement = g.run(`getLegalPlacements().find(p => p.actionId === ${action})`);
  if (!placement) throw new Error(`Illegal action ${action}`);
  g.run(`for (const step of ${JSON.stringify(placement.path)}) {
    if (step === "Hold") holdPiece();
    else if (step === "Left") move(-1);
    else if (step === "Right") move(1);
    else if (step === "RotateCW") rotate();
    else if (step === "Down") softDrop(false);
    else if (step === "HardDrop") hardDrop();
  }`);
  actions.push(action);
  states.push(snapshot());
}
process.stdout.write(JSON.stringify({ actions, states }));
