const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../tetris-game/game.js"), "utf8");
const placementSource = fs.readFileSync(path.join(__dirname, "../tetris-game/legal-placements.js"), "utf8");

function game(storage = new Map()) {
  const context2d = () => ({
    save() {}, restore() {}, clearRect() {}, fillRect() {}, beginPath() {},
    moveTo() {}, lineTo() {}, stroke() {}, translate() {}, roundRect() {}, fill() {},
    createLinearGradient() { return { addColorStop() {} }; }
  });
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      width: id === "gameCanvas" ? 320 : 144,
      height: id === "gameCanvas" ? 640 : id === "nextCanvas" ? 218 : 92,
      textContent: "", lastChild: { textContent: "" },
      classList: { add() {}, remove() {}, toggle() {} },
      setAttribute() {}, addEventListener() {}, getContext: context2d
    });
    return elements.get(id);
  };
  const sandbox = vm.createContext({
    document: {
      hidden: false,
      querySelector: selector => element(selector.slice(1)),
      querySelectorAll: () => [],
      addEventListener() {}
    },
    window: { addEventListener() {} },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value)
    },
    performance: { now: () => 1000 },
    requestAnimationFrame() {}, setInterval() {},
    location: { search: "" }, URLSearchParams
  });
  vm.runInContext(source, sandbox);
  vm.runInContext(placementSource, sandbox);
  return { run: code => vm.runInContext(code, sandbox), storage };
}

module.exports = { game };
