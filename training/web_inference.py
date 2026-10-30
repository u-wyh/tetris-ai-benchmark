"""Read-only public-state inference for the existing browser game."""

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sb3_contrib import MaskablePPO

from training.candidate_features import candidate_features
from training.env.tetris_env import TetrisEnv
from training.evaluation.failure_curriculum_eval import _DeterministicAdapter
from training.evaluation.hybrid_queue import HASHES, digest, load_models, verify_models
from training.hybrid_policy import HybridConfig, HybridController
from training.public_lookahead import PublicLookaheadController, PlanningConfig
from training.train_ppo import ROOT
from training.tetris_core.legal_placements import get_legal_placements

LOG = logging.getLogger(__name__)
PIECES = set("IJLOSTZ")
MODES = {"ppo", "bc", "b12", "lookahead"}
STUDENT = ROOT / "runs/bc_sequence_transfer/supervised/checkpoints/epoch_01/model.zip"
STUDENT_SHA256 = "e40bfad269566c052d880cb01857f5bd998e22a1730c6a71d75845387f605914"
PLANNER_SHA256 = "0c2dbea26708fed866c5089eb102672b0cfccafeaa9f130b58086c847e8ffb08"
PUBLIC_KEYS = {"board", "currentPiece", "hold", "holdAvailable", "next", "score",
               "level", "lines", "phase", "gameOver", "lock"}


def validate_public(public):
    if not isinstance(public, dict) or public.get("phase") != "playing":
        raise ValueError("Expected a public playing state")
    if set(public) - PUBLIC_KEYS:
        raise ValueError("Only public observation fields are accepted")
    if (any(type(public.get(k)) is not int or public[k] < 0 for k in ("score", "lines", "level"))
            or public["level"] != public["lines"] // 10 + 1):
        raise ValueError("Invalid public scoring state")
    board = public.get("board")
    if (not isinstance(board, list) or len(board) != 20 or
            any(not isinstance(row, list) or len(row) != 10 or
                any(cell is not None and cell not in PIECES for cell in row) for row in board)):
        raise ValueError("Invalid public board")
    current = public.get("currentPiece")
    if (not isinstance(current, dict) or current.get("type") not in PIECES or
            current.get("rotation") not in ("0", "R", "2", "L") or
            type(current.get("x")) is not int or type(current.get("y")) is not int or
            not isinstance(current.get("matrix"), list)):
        raise ValueError("Invalid public current piece")
    if (public.get("hold") is not None and public.get("hold") not in PIECES or
            type(public.get("holdAvailable")) is not bool or
            not isinstance(public.get("next"), list) or len(public["next"]) != 3 or
            any(piece not in PIECES for piece in public["next"])):
        raise ValueError("Invalid public Hold/Next")


class Inference:
    def __init__(self):
        verify_models()
        if digest(STUDENT) != STUDENT_SHA256 or digest(ROOT / "training/public_lookahead.py") != PLANNER_SHA256:
            raise RuntimeError("Frozen public planner/student SHA256 mismatch")
        models = load_models()
        self.models = {name: _DeterministicAdapter(model)
                       for name, model in models.items()}
        self.planner = PublicLookaheadController(MaskablePPO.load(STUDENT, device="cpu"), models["bc"])
        self.lock = threading.Lock()
        self.capacity = threading.BoundedSemaphore(4)

    def decide(self, mode, public, legal_ids=None):
        if mode not in MODES:
            raise ValueError("Unknown AI mode")
        validate_public(public)
        if mode == "lookahead" and legal_ids is None:
            raise ValueError("Public Lookahead requires the browser legal action list")
        started = time.monotonic()
        with self.lock:
            compute_started = time.monotonic()
            placements = get_legal_placements(public)
            features, mask = candidate_features(public, placements)
            ids = [p["actionId"] for p in placements]
            if legal_ids is not None and legal_ids != ids:
                raise ValueError("Browser and Python legal actions differ")
            observation = {"state": TetrisEnv._encode_observation(public),
                           "candidate": features}
            extra = {}
            if mode == "lookahead":
                action, detail = self.planner.decide(observation, mask, public)
                source, reason = "lookahead", detail["reason"]
                extra = {"studentActionId": detail["student_action"], "bcActionId": detail["bc_action"],
                         "override": detail["override"],
                         "thirdAction": action not in (detail["student_action"], detail["bc_action"]),
                         "boundedSearch": True}
            elif mode == "b12":
                controller = HybridController(self.models["ppo"], self.models["bc"],
                                              HybridConfig("B", 12, 8))
                action, detail = controller.decide(observation, mask, public, placements)
                source, reason = detail["source"], detail["reason"]
            else:
                action = int(self.models[mode].predict(
                    observation, deterministic=True, action_masks=mask)[0])
                source, reason = mode, "model"
            if not mask[action]:
                raise RuntimeError("Model chose an illegal Action ID")
            finished = time.monotonic()
        return {"actionId": action, "source": source, "reason": reason,
                "decisionMs": round((finished - started) * 1000, 3),
                "computeMs": round((finished - compute_started) * 1000, 3),
                "queueMs": round((compute_started - started) * 1000, 3),
                "legalCount": len(ids), **extra}


def make_handler(inference, allowed_origin):
    class Handler(BaseHTTPRequestHandler):
        def origin(self):
            value = self.headers.get("Origin")
            if value and value != allowed_origin:
                raise ValueError("Origin is not allowed")
            return value

        def send_json(self, status, payload, origin=None):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                LOG.debug("Client cancelled a completed decision")

        def do_OPTIONS(self):
            try:
                origin = self.origin()
                if self.path != "/api/decide" or origin is None:
                    raise ValueError("Unsupported preflight")
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "POST")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Max-Age", "300")
                self.end_headers()
            except ValueError as exc:
                self.send_json(403, {"error": str(exc)})

        def do_GET(self):
            try:
                origin = self.origin()
                if self.path != "/api/health":
                    self.send_json(404, {"error": "Not found"}, origin)
                else:
                    self.send_json(200, {"ready": True, "models": {**HASHES, "student": STUDENT_SHA256},
                                         "actionSpace": 1840, "modes": sorted(MODES),
                                         "planner": {"baseline": "a3bfc02", "sha256": PLANNER_SHA256,
                                                     "config": asdict(PlanningConfig())}}, origin)
            except ValueError as exc:
                self.send_json(403, {"error": str(exc)})

        def do_POST(self):
            try:
                origin = self.origin()
                if self.path != "/api/decide":
                    self.send_json(404, {"error": "Not found"}, origin)
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected JSON")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                legal = request.get("legalActionIds")
                if legal is not None and (not isinstance(legal, list) or
                                           any(type(x) is not int or not 0 <= x < 1840 for x in legal)):
                    raise ValueError("Invalid legal action list")
                if not inference.capacity.acquire(blocking=False):
                    self.send_json(503, {"error": "AI busy; retry shortly"}, origin)
                    return
                try:
                    result = inference.decide(request.get("mode"), request.get("observation"), legal)
                finally:
                    inference.capacity.release()
                self.send_json(200, result, origin)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                LOG.exception("Inference failed")
                self.send_json(500, {"error": "Inference failed"})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--web-origin", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    if args.host in ("0.0.0.0", "::"):
        parser.error("Bind to a trusted local or Tailscale address")
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(Inference(), args.web_origin))
    LOG.info("Frozen models %s; serving %s:%s for %s", HASHES, args.host,
             args.port, args.web_origin)
    server.serve_forever()


if __name__ == "__main__":
    main()
