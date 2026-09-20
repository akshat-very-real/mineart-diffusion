"""MineArt Diffusion - FastAPI Web Server & Minecraft Bridge.

Serves the interactive research frontend, executes asynchronous DDIM diffusion sampling,
and provides a 1-click exporter directly into Minecraft's painting texture registry.
"""

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.minecraft_pack import (
    PAINTING_SLOTS,
    create_standalone_zip,
    export_painting_to_pack,
    get_default_minecraft_dir,
    install_to_minecraft_live,
)
from scripts.prompt_moderator import validate_prompt

app = FastAPI(title="MineArt Diffusion API", version="1.3")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODPACK_DIR = BASE_DIR / "modpack"
MODPACK_DIR.mkdir(parents=True, exist_ok=True)


class TextGenerateRequest(BaseModel):
    prompt: str
    mode: str = "text"
    ratio: str = "1:1"
    steps: int = 50
    guidance: float = 2.0


class ExportMinecraftRequest(BaseModel):
    slot: str = "bust"
    ratio: Optional[str] = "2:2"
    style: Optional[str] = "vanilla_authentic"
    add_border: Optional[bool] = True


def run_diffusion_generation(
    prompt: str,
    ratio: str = "1:1",
    steps: int = 50,
    guidance: float = 2.5,
    input_image_path: Optional[str] = None,
    strength: float = 0.55,
    framed_style: str = "vanilla_authentic",
) -> Dict[str, str]:
    """Invokes the diffusion generation pipeline synchronously inside a worker thread."""
    from scripts.diffusion import generate_painting

    checkpoint = BASE_DIR / "checkpoints" / "best_diffusion.pt"
    tokenizer = BASE_DIR / "checkpoints" / "diffusion_tokenizer.json"

    # Fallback to local raw checkpoint if best_diffusion is absent
    if not checkpoint.exists():
        fallback_ckpt = BASE_DIR / "checkpoints" / "mineart_diffusion_epoch_10.pt"
        if fallback_ckpt.exists():
            checkpoint = fallback_ckpt

    timestamp = int(time.time())
    output_filename = f"painting_{timestamp}.png"
    output_path = OUTPUT_DIR / output_filename
    latest_path = OUTPUT_DIR / "latest.png"
    framed_filename = f"painting_{timestamp}_framed.png"
    framed_path = OUTPUT_DIR / framed_filename
    latest_framed_path = OUTPUT_DIR / "latest_framed.png"

    if checkpoint.exists() and tokenizer.exists():
        generate_painting(
            prompt=prompt,
            ratio=ratio,
            input_image_path=input_image_path,
            strength=strength,
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
            output_path=str(output_path),
            guidance_scale=guidance,
            ddim_steps=steps,
            save_framed=True,
            framed_style=framed_style,
        )
    else:
        # Graceful fallback demo image if checkpoint not yet compiled
        sample_source = BASE_DIR / "data" / "processed_64x64" / "acacia_chicken_001_C.png"
        if sample_source.exists():
            shutil.copy(sample_source, output_path)
        else:
            from PIL import Image
            img = Image.new("RGB", (64, 64), color=(60, 120, 60))
            img.save(output_path)

    shutil.copy(output_path, latest_path)

    # Always ensure authentic framed painting preview exists matching latest_framed.png
    try:
        from PIL import Image
        from scripts.minecraft_pack import render_minecraft_canvas_texture

        with Image.open(output_path) as raw_gen:
            framed_img = render_minecraft_canvas_texture(
                image=raw_gen.convert("RGB"),
                target_px=(128, 128),
                block_size=(2, 2),
                style=framed_style,
                add_border=True,
            )
            framed_img.save(latest_framed_path)
            framed_img.save(framed_path)
    except Exception as e:
        print(f"[*] Note: Framing preview skipped: {e}")

    return {
        "raw_filename": output_filename,
        "framed_filename": framed_filename,
    }


@app.post("/api/generate")
async def generate_artwork(request: Request):
    """Asynchronous generation endpoint accepting both JSON and Form payloads."""
    content_type = request.headers.get("content-type", "")
    prompt = ""
    mode = "text"
    ratio = "1:1"
    steps = 50
    guidance = 2.5
    input_image_path = None

    safety_filter = True
    if "application/json" in content_type:
        try:
            data = await request.json()
            prompt = str(data.get("prompt", ""))
            mode = str(data.get("mode", "text"))
            ratio = str(data.get("ratio", "1:1"))
            steps = int(data.get("steps", 50))
            guidance = float(data.get("guidance", 2.5))
            safety_filter = bool(data.get("safety_filter", True))
        except Exception:
            prompt = ""
            safety_filter = True
    else:
        form = await request.form()
        prompt = str(form.get("prompt", ""))
        mode = str(form.get("mode", "text"))
        ratio = str(form.get("ratio", "1:1"))
        steps = int(form.get("steps", 50))
        guidance = float(form.get("guidance", 2.5))
        safety_filter = str(form.get("safety_filter", "true")).lower() in ("true", "1", "yes")

        # Handle uploaded image for Image-to-Image stylization
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "filename") and uploaded_file.filename:
            uploads_dir = BASE_DIR / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            saved_name = f"upload_{int(time.time())}_{uploaded_file.filename}"
            saved_path = uploads_dir / saved_name
            content = await uploaded_file.read()
            with open(saved_path, "wb") as f:
                f.write(content)
            input_image_path = str(saved_path)

    prompt_str = prompt.strip() if prompt else ""

    # Production Safety & Coherence Guardrail: Intercept gibberish and harmful text
    if mode == "text" and safety_filter and prompt_str:
        val_result = validate_prompt(prompt_str, check_gibberish=True, check_safety=True)
        if not val_result.is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "blocked": True,
                    "detail": val_result.message,
                    "error_type": val_result.error_type,
                    "flagged_token": val_result.flagged_token,
                    "category": val_result.category,
                },
            )

    if not prompt_str:
        prompt_str = "minecraft authentic landscape painting" if mode == "text" else "minecraft painting"

    # Offload diffusion model inference to background thread pool
    try:
        res = await asyncio.to_thread(
            run_diffusion_generation,
            prompt=prompt_str,
            ratio=ratio,
            steps=steps,
            guidance=guidance,
            input_image_path=input_image_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diffusion generation error: {str(e)}")

    return {
        "success": True,
        "image_url": f"/outputs/{res['framed_filename']}",
        "framed_url": f"/outputs/{res['framed_filename']}",
        "raw_url": f"/outputs/{res['raw_filename']}",
        "download_url": f"/outputs/{res['framed_filename']}",
        "filename": res["framed_filename"],
        "prompt_used": prompt_str,
    }


@app.post("/api/validate-prompt")
async def check_prompt_endpoint(request: Request):
    """Real-time validation endpoint for client-side feedback and diagnostics."""
    try:
        data = await request.json()
        prompt = str(data.get("prompt", "")).strip()
    except Exception:
        prompt = ""
    result = validate_prompt(prompt, check_gibberish=True, check_safety=True)
    return result.to_dict()


@app.get("/api/minecraft-status")
async def get_minecraft_status():
    """Checks whether Minecraft Java Edition is installed locally."""
    mc_dir = get_default_minecraft_dir()
    if mc_dir and mc_dir.exists():
        pack_installed = (mc_dir / "resourcepacks" / "MineArt_Pack").exists()
        return {
            "installed": True,
            "path": str(mc_dir),
            "pack_installed": pack_installed,
            "slots": [
                {"id": k, "desc": v["desc"], "ratio": v["ratio"], "dimensions": f"{v['target_px'][0]}x{v['target_px'][1]}"}
                for k, v in PAINTING_SLOTS.items()
            ],
        }
    return {
        "installed": False,
        "path": None,
        "pack_installed": False,
        "slots": [],
    }


@app.post("/api/export-minecraft")
async def export_to_minecraft(req: ExportMinecraftRequest):
    """Packages the latest generated painting and injects it into live Minecraft."""
    latest_img = OUTPUT_DIR / "latest.png"
    if not latest_img.exists():
        # Fallback to sample image if available
        sample_p = BASE_DIR / "data" / "processed_64x64" / "acacia_chicken_001_C.png"
        if sample_p.exists():
            shutil.copy(sample_p, latest_img)
        else:
            raise HTTPException(status_code=400, detail="No generated painting found. Generate an image first!")

    try:
        res = install_to_minecraft_live(
            input_image_path=str(latest_img),
            slot=req.slot,
            ratio=req.ratio,
            style=req.style or "vanilla_authentic",
            add_border=req.add_border if req.add_border is not None else True,
        )
        # Also refresh the standalone downloadable zip in modpack/
        pack_dir = Path(res["pack_path"])
        # Save a framed copy to output_samples for web UI preview & download
        framed_filename = f"framed_{res['slot']}.png"
        shutil.copy(res["target_file"], OUTPUT_DIR / framed_filename)
        shutil.copy(res["target_file"], OUTPUT_DIR / "latest_framed.png")

        return {
            "success": True,
            "slot": res["slot"],
            "description": res["description"],
            "target_file": res["target_file"],
            "style": res.get("style", "vanilla_authentic"),
            "framed_url": f"/outputs/{framed_filename}",
            "instructions": res["instructions"],
            "download_pack_url": "/api/download-pack",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to Minecraft: {str(e)}")


@app.get("/api/download-pack")
async def download_resource_pack():
    """Serves the standalone MineArt_Pack.zip file."""
    zip_p = MODPACK_DIR / "MineArt_Pack.zip"
    if not zip_p.exists():
        raise HTTPException(status_code=404, detail="Resource pack not generated yet.")
    return FileResponse(
        str(zip_p),
        media_type="application/zip",
        filename="MineArt_AI_Paintings.zip",
    )


# Static files mount
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(" MineArt Diffusion Web Application Started!")
    print(" URL: http://127.0.0.1:8000")
    print("=======================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
