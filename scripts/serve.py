"""
MineArt Diffusion — Minimal Web Server & Service Boundary

Serves the minimal research frontend on http://127.0.0.1:8000 and generates
Minecraft artwork directly using our custom-trained PyTorch DDPM diffusion model.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
from PIL import Image
import torch

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "output_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add root to sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_diffusion_32x32 import UNet32, Diffusion

# Global Model Cache
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DDPM_MODEL = None
DIFFUSION_SAMPLER = None
CHECKPOINT_PATH = OUTPUT_DIR / "ddpm_minecraft_32x32.pt"


def get_model():
    global DDPM_MODEL, DIFFUSION_SAMPLER
    if DDPM_MODEL is None and CHECKPOINT_PATH.exists():
        try:
            print(f"[Server] Loading custom DDPM model from {CHECKPOINT_PATH} on {DEVICE}...")
            model = UNet32(in_channels=3, out_channels=3, time_dim=64).to(DEVICE)
            weights = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(weights)
            model.eval()
            DDPM_MODEL = model
            DIFFUSION_SAMPLER = Diffusion(timesteps=250, schedule="cosine", device=DEVICE)
            print("[Server] Custom Structural DDPM Model successfully loaded and ready for sampling.")
        except Exception as e:
            print(f"[Server] Warning: Failed to load checkpoint ({e})")
    return DDPM_MODEL, DIFFUSION_SAMPLER


class MineArtRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # 1. Direct Static Output & Download Routes ending with .png
        if parsed.path.startswith("/outputs/") or parsed.path.startswith("/api/download"):
            file_name = os.path.basename(parsed.path)
            if not file_name or file_name in ["download", "outputs"]:
                query = urllib.parse.parse_qs(parsed.query)
                file_name = query.get("file", ["mineart_diffusion_32x32.png"])[0]

            safe_name = os.path.basename(file_name)
            file_path = OUTPUT_DIR / safe_name
            
            if not file_path.exists():
                pngs = sorted(OUTPUT_DIR.glob("*.png"), key=os.path.getmtime)
                file_path = pngs[-1] if pngs else None

            if file_path and file_path.exists():
                with open(file_path, "rb") as f:
                    content = f.read()
                
                # Format download attachment name with explicit .png extension
                attachment_name = "mineart_diffusion_32x32.png"
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # Default static file serving (HTML, CSS, JS)
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)
            
            prompt = "Minecraft Scene"
            if "application/json" in content_type:
                try:
                    payload = json.loads(body.decode("utf-8"))
                    prompt = payload.get("prompt", "")
                except Exception:
                    pass

            model, sampler = get_model()

            if model is not None and sampler is not None:
                # 1. Real Reverse Diffusion Sampling from trained PyTorch DDPM Model
                print(f"[Server] Running DDPM Reverse Diffusion (200 steps) for prompt: '{prompt}'...")
                t0 = time.time()
                with torch.no_grad():
                    sample = sampler.sample(model, n_samples=1, image_size=32)
                
                # Convert tensor to PIL Image [32x32]
                arr = (sample[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(arr)
                
                # Save to output_samples
                timestamp = int(time.time() * 1000)
                gen_file_name = f"mineart_sample_{timestamp}.png"
                gen_file = OUTPUT_DIR / gen_file_name
                img.save(gen_file, format="PNG")
                
                # Also save a canonical latest.png
                latest_file = OUTPUT_DIR / "latest.png"
                img.save(latest_file, format="PNG")

                # Also encode base64 Data URL for fallback
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                data_url = f"data:image/png;base64,{b64_img}"
                
                elapsed = time.time() - t0
                print(f"[Server] Generated 32x32 Minecraft sample in {elapsed:.2f}s -> {gen_file.name}")

                response_data = {
                    "status": "success",
                    "prompt": prompt,
                    "model": "Custom DDPM (UNet32)",
                    "resolution": "32x32",
                    "elapsed_seconds": round(elapsed, 2),
                    "image_url": f"/outputs/{gen_file_name}",
                    "data_url": data_url,
                    "download_url": f"/outputs/{gen_file_name}"
                }
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                err_msg = json.dumps({
                    "detail": "No trained DDPM checkpoint found at output_samples/ddpm_minecraft_32x32.pt. Please train the model first using 'python scripts/train_diffusion_32x32.py'."
                }).encode("utf-8")
                self.send_header("Content-Length", str(len(err_msg)))
                self.end_headers()
                self.wfile.write(err_msg)
                return
            
            encoded = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[Server] {self.address_string()} - {args[0]} {args[1]}\n")


def run_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    get_model()
    httpd = HTTPServer(server_address, MineArtRequestHandler)
    print("==================================================")
    print("      MineArt Diffusion — Minimal Web Server      ")
    print("==================================================")
    print(f" Web UI running at : http://127.0.0.1:{port}")
    print(" Model Mode        : Custom PyTorch DDPM (32x32)")
    print(" Press Ctrl+C to stop the server.")
    print("--------------------------------------------------")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve MineArt minimal web interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
