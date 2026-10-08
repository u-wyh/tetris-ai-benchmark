"use strict";
importScripts("traditional-rules.js", "legal-placements.js", "traditional-ai.js");
onmessage = event => {
  const { id, observation, mode, rngState } = event.data;
  try {
    const result = mode === "v1"
      ? traditionalChooseV1(observation, rngState)
      : { actionId: traditionalChooseV2(observation, mode === "beam" ? "beam" : "hold"), rngState };
    postMessage({ id, ...result });
  } catch (error) {
    postMessage({ id, error: String(error) });
  }
};
