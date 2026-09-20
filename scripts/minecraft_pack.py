"""Minecraft Painting Resource Pack Generator & Live Game Importer.

Bridges AI-generated diffusion artwork with Minecraft by automatically packaging
images into authentic Minecraft painting textures (1x1, 2x1, 2x2, 4x4 blocks)
and installing them directly into the player's `.minecraft/resourcepacks/` folder.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Standard painting slots and their block dimensions in Minecraft Java Edition
PAINTING_SLOTS: Dict[str, Dict] = {
    # 1x1 block paintings (Default target resolution 64x64)
    "alban": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Alban (1x1 blocks)"},
    "aztec": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Aztec (1x1 blocks)"},
    "kebab": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Kebab (1x1 blocks)"},
    "bomb": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Bomb (1x1 blocks)"},
    "plant": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Plant (1x1 blocks)"},
    "wasteland": {"width_blocks": 1, "height_blocks": 1, "target_px": (64, 64), "ratio": "1:1", "desc": "Wasteland (1x1 blocks)"},
    # 2x1 block paintings (128x64)
    "pool": {"width_blocks": 2, "height_blocks": 1, "target_px": (128, 64), "ratio": "2:1", "desc": "The Pool (2x1 blocks)"},
    "courbet": {"width_blocks": 2, "height_blocks": 1, "target_px": (128, 64), "ratio": "2:1", "desc": "Bonjour monsieur Courbet (2x1 blocks)"},
    "sunset": {"width_blocks": 2, "height_blocks": 1, "target_px": (128, 64), "ratio": "2:1", "desc": "Sunset (2x1 blocks)"},
    "sea": {"width_blocks": 2, "height_blocks": 1, "target_px": (128, 64), "ratio": "2:1", "desc": "Sea (2x1 blocks)"},
    # 1x2 block paintings (64x128)
    "wanderer": {"width_blocks": 1, "height_blocks": 2, "target_px": (64, 128), "ratio": "1:2", "desc": "Wanderer (1x2 blocks)"},
    "graham": {"width_blocks": 1, "height_blocks": 2, "target_px": (64, 128), "ratio": "1:2", "desc": "Graham (1x2 blocks)"},
    # 2x2 block paintings (128x128)
    "match": {"width_blocks": 2, "height_blocks": 2, "target_px": (128, 128), "ratio": "2:2", "desc": "Match (2x2 blocks)"},
    "bust": {"width_blocks": 2, "height_blocks": 2, "target_px": (128, 128), "ratio": "2:2", "desc": "Bust (2x2 blocks)"},
    "stage": {"width_blocks": 2, "height_blocks": 2, "target_px": (128, 128), "ratio": "2:2", "desc": "Stage (2x2 blocks)"},
    "skull_and_roses": {"width_blocks": 2, "height_blocks": 2, "target_px": (128, 128), "ratio": "2:2", "desc": "Skull and Roses (2x2 blocks)"},
    "wither": {"width_blocks": 2, "height_blocks": 2, "target_px": (128, 128), "ratio": "2:2", "desc": "Wither (2x2 blocks)"},
    # 4x2 block paintings (256x128)
    "fighters": {"width_blocks": 4, "height_blocks": 2, "target_px": (256, 128), "ratio": "4:2", "desc": "Fighters (4x2 blocks)"},
    # 4x3 block paintings (256x192)
    "skeleton": {"width_blocks": 4, "height_blocks": 3, "target_px": (256, 192), "ratio": "4:3", "desc": "Mortal Coil (4x3 blocks)"},
    "donkey_kong": {"width_blocks": 4, "height_blocks": 3, "target_px": (256, 192), "ratio": "4:3", "desc": "Kong (4x3 blocks)"},
    # 4x4 block paintings (256x256)
    "pointer": {"width_blocks": 4, "height_blocks": 4, "target_px": (256, 256), "ratio": "4:4", "desc": "Pointer (4x4 blocks)"},
    "pigscene": {"width_blocks": 4, "height_blocks": 4, "target_px": (256, 256), "ratio": "4:4", "desc": "Pigscene (4x4 blocks)"},
    "burningskull": {"width_blocks": 4, "height_blocks": 4, "target_px": (256, 256), "ratio": "4:4", "desc": "Skull on Fire (4x4 blocks)"},
}


def get_default_minecraft_dir() -> Optional[Path]:
    """Auto-detects the standard .minecraft directory on Windows, Linux, and macOS."""
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            mc = Path(appdata) / ".minecraft"
            if mc.exists():
                return mc
    elif sys.platform == "darwin":
        mc = Path.home() / "Library" / "Application Support" / "minecraft"
        if mc.exists():
            return mc
    else:
        mc = Path.home() / ".minecraft"
        if mc.exists():
            return mc
    return None


def create_pack_metadata(dest_dir: Path, pack_name: str = "MineArt AI Paintings") -> None:
    """Creates pack.mcmeta and a default pack.png icon."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Modern format 15 covers Minecraft 1.20 - 1.21 (compatible back to 1.16)
    mcmeta_content = {
        "pack": {
            "pack_format": 15,
            "supported_formats": [4, 40],
            "description": f"MineArt Diffusion: {pack_name} (AI Generated)",
        }
    }
    with open(dest_dir / "pack.mcmeta", "w", encoding="utf-8") as f:
        json.dump(mcmeta_content, f, indent=2)

    # Generate an authentic 64x64 MineArt pack icon if not present
    pack_png = dest_dir / "pack.png"
    if not pack_png.exists():
        icon = Image.new("RGB", (64, 64), color=(34, 139, 34))  # Forest Green
        # Simple pixel art easel border
        for x in range(64):
            for y in range(64):
                if x < 4 or x >= 60 or y < 4 or y >= 60:
                    icon.putpixel((x, y), (101, 67, 33))  # Oak wood border
                elif (x + y) % 8 == 0:
                    icon.putpixel((x, y), (50, 160, 50))
        icon.save(pack_png)


# Authentic Kristoffer Zetterstrand wooden frame palette extracted from vanilla Minecraft assets
VANILLA_WOOD_PALETTE = np.array([
    [74, 52, 18],
    [77, 45, 14],
    [77, 41, 21],
    [67, 32, 14],
    [75, 35, 18],
    [74, 32, 16],
    [76, 41, 13],
    [62, 25, 12],
    [76, 37, 19],
    [76, 39, 20],
    [78, 47, 15],
    [65, 29, 13],
    [78, 48, 16],
], dtype=np.uint8)


def render_minecraft_canvas_texture(
    image: Image.Image,
    target_px: Tuple[int, int],
    block_size: Tuple[int, int],
    style: str = "vanilla_authentic",
    add_border: bool = True,
) -> Image.Image:
    """Transforms raw generated diffusion artwork into authentic Minecraft oil canvas texture.
    
    Styles:
      - 'vanilla_authentic': 16px per block, pixel-quantized, Floyd-Steinberg dithered, linen weave.
      - 'crisp_hd': 32px per block, high-definition pixel art with fine canvas grain.
      - 'oil_studio': 64px per block, oil brush impasto and heavy linen weave.
      
    Features:
      - Authentic wooden frame border matching official Minecraft painting textures (Alban, Bust, Aztec).
      - Recessed canvas shadow bevel simulating depth inside the frame.
    """
    w_blocks, h_blocks = block_size
    target_w, target_h = target_px

    if style == "vanilla_authentic":
        canvas_w = max(16, w_blocks * 16)
        canvas_h = max(16, h_blocks * 16)
        border_px = 1 if add_border else 0
        dither_colors = 64
        contrast_boost = 1.25
        saturation_boost = 1.25
    elif style == "crisp_hd":
        canvas_w = max(32, w_blocks * 32)
        canvas_h = max(32, h_blocks * 32)
        border_px = 2 if add_border else 0
        dither_colors = 96
        contrast_boost = 1.20
        saturation_boost = 1.20
    else:  # oil_studio
        canvas_w = max(64, w_blocks * 64)
        canvas_h = max(64, h_blocks * 64)
        border_px = 3 if add_border else 0
        dither_colors = 160
        contrast_boost = 1.15
        saturation_boost = 1.15

    # 1. Determine inner canvas dimensions (framed area)
    inner_w = canvas_w - (2 * border_px)
    inner_h = canvas_h - (2 * border_px)

    # Downscale artwork to fit inside the frame with edge-preserving unsharp mask
    grid = image.resize((inner_w, inner_h), Image.Resampling.BILINEAR)
    grid = grid.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=1))

    # 2. Rich oil paint pigment color grading (removes flat downscaled camera look)
    enhancer = ImageEnhance.Contrast(grid)
    grid = enhancer.enhance(contrast_boost)
    enhancer = ImageEnhance.Color(grid)
    grid = enhancer.enhance(saturation_boost)

    # 3. Procedural linen canvas weave modulation
    arr_inner = np.array(grid, dtype=np.float32)
    H, W, _ = arr_inner.shape

    # Periodic thread weave
    y_idx = np.arange(H)[:, None]
    x_idx = np.arange(W)[None, :]
    weave = np.sin(y_idx * np.pi) * 3.0 + np.cos(x_idx * np.pi) * 3.0

    # Fine organic canvas fiber noise
    rng = np.random.RandomState(42)
    fiber_noise = rng.uniform(-2.5, 2.5, (H, W))
    arr_inner = np.clip(arr_inner + (weave + fiber_noise)[:, :, None], 0, 255)

    # 4. Recessed canvas bevel shadow (edge darkening where canvas sits inside the wooden frame)
    arr_inner[0, :] *= 0.82
    arr_inner[-1, :] *= 0.88
    arr_inner[:, 0] *= 0.82
    arr_inner[:, -1] *= 0.88

    textured_inner = Image.fromarray(np.uint8(np.clip(arr_inner, 0, 255)))

    # 5. Authentic oil palette quantization & Floyd-Steinberg dithering
    quantized_inner = textured_inner.quantize(
        colors=dither_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG,
    ).convert("RGB")

    # 6. Composite into full block grid with authentic wooden frame border
    full_arr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if border_px > 0:
        # Place quantized canvas into center
        full_arr[border_px:canvas_h-border_px, border_px:canvas_w-border_px] = np.array(quantized_inner)

        # Render authentic Minecraft wooden frame along outer perimeter
        frame_rng = np.random.RandomState(101)

        # Top border bar
        for r in range(border_px):
            for c in range(canvas_w):
                idx = frame_rng.randint(0, len(VANILLA_WOOD_PALETTE))
                full_arr[r, c] = VANILLA_WOOD_PALETTE[idx]

        # Bottom border bar (slightly shaded for 3D lighting)
        for r in range(canvas_h - border_px, canvas_h):
            for c in range(canvas_w):
                idx = frame_rng.randint(0, len(VANILLA_WOOD_PALETTE))
                base = VANILLA_WOOD_PALETTE[idx].astype(np.float32) * 0.92
                full_arr[r, c] = np.clip(base, 0, 255)

        # Left border stile
        for c in range(border_px):
            for r in range(canvas_h):
                idx = frame_rng.randint(0, len(VANILLA_WOOD_PALETTE))
                full_arr[r, c] = VANILLA_WOOD_PALETTE[idx]

        # Right border stile (slightly shaded)
        for c in range(canvas_w - border_px, canvas_w):
            for r in range(canvas_h):
                idx = frame_rng.randint(0, len(VANILLA_WOOD_PALETTE))
                base = VANILLA_WOOD_PALETTE[idx].astype(np.float32) * 0.92
                full_arr[r, c] = np.clip(base, 0, 255)

        # Top-left outer highlight, bottom-right outer shadow
        full_arr[0, 0] = [84, 60, 22]
        full_arr[-1, -1] = [52, 20, 10]
    else:
        full_arr = np.array(quantized_inner)

    # 7. Strict NEAREST-NEIGHBOR upscale to target texture resolution (ZERO in-game blur!)
    framed_img = Image.fromarray(full_arr)
    final_texture = framed_img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return final_texture


def export_painting_to_pack(
    input_image_path: str,
    pack_dir: Path,
    slot: str = "bust",
    ratio: Optional[str] = None,
    style: str = "vanilla_authentic",
    add_border: bool = True,
) -> Tuple[Path, str]:
    """Formats the generated image and places it in the painting texture registry."""
    if slot not in PAINTING_SLOTS:
        # Default mapping based on aspect ratio
        ratio_slot_map = {
            "1:1": "alban",
            "2:1": "pool",
            "1:2": "wanderer",
            "2:2": "bust",
            "4:2": "fighters",
            "4:3": "skeleton",
            "4:4": "pointer",
        }
        slot = ratio_slot_map.get(ratio, "bust")

    slot_info = PAINTING_SLOTS[slot]
    target_w, target_h = slot_info["target_px"]
    w_blocks = slot_info["width_blocks"]
    h_blocks = slot_info["height_blocks"]

    # Ensure texture directory exists
    textures_dir = pack_dir / "assets" / "minecraft" / "textures" / "painting"
    textures_dir.mkdir(parents=True, exist_ok=True)
    create_pack_metadata(pack_dir)

    # Load and format the image with authentic Minecraft canvas oil painting texture & wooden border
    with Image.open(input_image_path) as img:
        img_rgb = img.convert("RGB")
        formatted_img = render_minecraft_canvas_texture(
            image=img_rgb,
            target_px=(target_w, target_h),
            block_size=(w_blocks, h_blocks),
            style=style,
            add_border=add_border,
        )
        
        target_file = textures_dir / f"{slot}.png"
        formatted_img.save(target_file, "PNG")

    return target_file, slot_info["desc"]


def create_standalone_zip(pack_dir: Path, output_zip: Path) -> Path:
    """Compresses the pack folder into a single, distributable .zip resource pack."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(pack_dir):
            for file in files:
                full_p = Path(root) / file
                rel_p = full_p.relative_to(pack_dir)
                zf.write(full_p, rel_p)
    return output_zip


def install_to_minecraft_live(
    input_image_path: str,
    slot: str = "bust",
    ratio: Optional[str] = None,
    style: str = "vanilla_authentic",
    add_border: bool = True,
    custom_mc_dir: Optional[str] = None,
) -> Dict:
    """Directly installs the generated painting into the local .minecraft directory."""
    mc_dir = Path(custom_mc_dir) if custom_mc_dir else get_default_minecraft_dir()
    if not mc_dir or not mc_dir.exists():
        raise FileNotFoundError("Minecraft installation directory (.minecraft) not found on this system.")

    resourcepacks_dir = mc_dir / "resourcepacks"
    resourcepacks_dir.mkdir(parents=True, exist_ok=True)

    # Use a clean folder pack name: MineArt_Pack
    pack_dir = resourcepacks_dir / "MineArt_Pack"
    target_file, desc = export_painting_to_pack(
        input_image_path=input_image_path,
        pack_dir=pack_dir,
        slot=slot,
        ratio=ratio,
        style=style,
        add_border=add_border,
    )

    return {
        "success": True,
        "pack_path": str(pack_dir),
        "target_file": str(target_file),
        "slot": slot,
        "description": desc,
        "style": style,
        "add_border": add_border,
        "instructions": (
            "Painting successfully imported into Minecraft! "
            "In Minecraft, open Options -> Resource Packs -> activate 'MineArt AI Paintings'. "
            "If already activated, press F3 + T in-game to instantly reload textures!"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="MineArt Minecraft Resource Pack Generator")
    parser.add_argument("--image", type=str, required=True, help="Input generated image path")
    parser.add_argument("--slot", type=str, default="bust", choices=list(PAINTING_SLOTS.keys()), help="Target painting slot")
    parser.add_argument("--ratio", type=str, default=None, help="Aspect ratio (1:1, 2:1, 2:2, 4:4)")
    parser.add_argument("--style", type=str, default="vanilla_authentic", choices=["vanilla_authentic", "crisp_hd", "oil_studio"], help="Painting canvas texture style")
    parser.add_argument("--no-border", action="store_true", help="Omit the oak wooden frame border")
    parser.add_argument("--output-framed", type=str, default=None, help="Save standalone framed painting PNG to path")
    parser.add_argument("--install", action="store_true", help="Install directly into local .minecraft directory")
    parser.add_argument("--output-dir", type=str, default="modpack/MineArt_Pack", help="Output pack directory")
    parser.add_argument("--output-zip", type=str, default="modpack/MineArt_Pack.zip", help="Output standalone zip file")

    args = parser.parse_args()
    add_border = not args.no_border

    if args.install:
        print(f"[*] Installing image '{args.image}' directly into live Minecraft...")
        res = install_to_minecraft_live(args.image, slot=args.slot, ratio=args.ratio, style=args.style, add_border=add_border)
        print(f"[OK] Installed to: {res['target_file']}")
        print(f"[OK] {res['instructions']}")
        if args.output_framed:
            shutil.copy(res["target_file"], args.output_framed)
            print(f"[OK] Saved framed painting preview to: {args.output_framed}")
    else:
        pack_dir = Path(args.output_dir)
        print(f"[*] Exporting painting to pack directory: {pack_dir}")
        target_file, desc = export_painting_to_pack(args.image, pack_dir, slot=args.slot, ratio=args.ratio, style=args.style, add_border=add_border)
        print(f"[OK] Texture written to: {target_file} ({desc})")

        if args.output_framed:
            shutil.copy(target_file, args.output_framed)
            print(f"[OK] Saved framed painting preview to: {args.output_framed}")

        zip_p = create_standalone_zip(pack_dir, Path(args.output_zip))
        print(f"[OK] Created standalone zip: {zip_p}")


if __name__ == "__main__":
    main()
