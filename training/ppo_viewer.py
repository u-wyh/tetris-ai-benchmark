"""Local browser viewer backed by live MaskablePPO inference and TetrisEnv."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from training.env import TetrisEnv


PAGE = Path(__file__).with_name("ppo_viewer.html")


class Viewer:
    def __init__(self, model_path, seed):
        torch.set_num_threads(1)
        self.model = MaskablePPO.load(str(model_path), device="cpu")
        self.model_name = str(model_path)
        self.env = TetrisEnv(max_pieces=5000)
        self.reset(seed)

    def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed 必须是 0 到 4294967295 的整数")
        self.observation, _ = self.env.reset(seed=seed)
        self.seed = seed
        self.finished = False
        self.last_placement = None
        return self.state()

    def state(self):
        return {"seed": self.seed, "pieces": self.env.pieces,
                "finished": self.finished, "maxPieces": self.env.max_pieces,
                "model": self.model_name,
                "game": self.env.core.get_public_observation(),
                "lastPlacement": self.last_placement}

    def step(self):
        if self.finished:
            raise ValueError("本局已结束，请开始新游戏")
        before = self.env.core.get_public_observation()
        action, _ = self.model.predict(self.observation, deterministic=True,
                                       action_masks=self.env.action_masks())
        action = int(action)
        placement = next(p for p in self.env.core.get_legal_placements()
                         if p["actionId"] == action)
        piece_type = ((before["hold"] or before["next"][0]) if placement["hold"]
                      else before["currentPiece"]["type"])
        self.last_placement = {
            "actionId": action, "pieceType": piece_type,
            "hold": bool(placement["hold"]),
            "rotation": placement["rotation"],
            "x": placement["x"], "y": placement["y"],
            "occupiedCells": placement["occupiedCells"],
        }
        self.observation, _, terminated, truncated, _ = self.env.step(action)
        self.finished = bool(terminated or truncated)
        return self.state()


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def send_data(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self.send_data(viewer.state())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ("/api/reset", "/api/step"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1024:
                    raise ValueError("请求过大")
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/reset":
                    data = viewer.reset(body["seed"])
                else:
                    data = viewer.step()
                self.send_data(data)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_data({"error": str(error)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Watch trained PPO play in a browser")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/ppo_raw_10m_seed42"))
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if args.checkpoint == "best":
        model_path = args.run_dir / "best" / "model.zip"
    else:
        transaction = json.loads((args.run_dir / "transaction_state.json").read_text())
        model_path = (args.run_dir / "committed"
                      / f"task_{transaction['committed_task']:06d}" / "model.zip")
    if not model_path.is_file():
        parser.error(f"找不到模型：{model_path}")
    viewer = Viewer(model_path, args.seed)
    server = HTTPServer((args.host, args.port), make_handler(viewer))
    print(f"PPO viewer: http://{args.host}:{args.port}/", flush=True)
    print(f"Model: {model_path.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        viewer.env.close()


if __name__ == "__main__":
    main()
