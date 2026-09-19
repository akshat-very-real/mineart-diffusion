# MineArt Diffusion: Minecraft In-Game Painting Import Guide

This package delivers the **final end-to-end product**: taking AI-generated diffusion artwork and importing it as authentic, placeable painting blocks inside the Minecraft game world.

---

## 1. Quick Demonstration Workflow (1-Minute Live Demo)

### Method A: Live Web UI One-Click Export (Recommended)
1. Start the MineArt Web App on your machine:
   ```bash
   python app.py
   ```
2. Open your browser at: **`http://127.0.0.1:8000`**
3. Type a prompt (e.g. *"minecraft sunset plains mountain landscape"*) and click **Generate Image**.
4. Select your desired in-game painting size (e.g., **Bust: 2x2 Blocks** or **Pointer: 4x4 Blocks**).
5. Click **`🎮 Send to Minecraft Wall`**.
6. Switch to your Minecraft game window:
   * If Minecraft is running, press **`F3 + T`** to reload textures on the fly!
   * Place a Painting item on any wall—**your AI painting appears live on the wall in Minecraft!**

---

### Method B: Standalone Evaluator Installer (For Other Machines / Panelists)
If testing on a separate computer where only Minecraft is installed:
1. Double-click **`install_to_minecraft.bat`**.
2. It automatically places `MineArt_AI_Paintings.zip` into `%APPDATA%\.minecraft\resourcepacks\`.
3. Launch Minecraft $\rightarrow$ **Options $\rightarrow$ Resource Packs $\rightarrow$ Select "MineArt AI Paintings"**.
4. Place a painting in your world!

---

## 2. Technical Architecture & Painting Texture Slots

Minecraft Java Edition handles paintings through the texture registry:
`assets/minecraft/textures/painting/<variant>.png`

Our pipeline automatically handles canvas scaling and block aspect ratios:

| Canvas Ratio | In-Game Blocks | Resolution | Default Slot Replaced | Best For |
| :--- | :---: | :---: | :--- | :--- |
| **1:1** | 1×1 Block | $64 \times 64$ px | `alban.png` / `kebab.png` | Small interior wall art |
| **2:1** | 2×1 Blocks | $128 \times 64$ px | `pool.png` / `courbet.png` | Wide panoramic horizons |
| **2:2** | 2×2 Blocks | $128 \times 128$ px | `bust.png` / `skull_and_roses.png` | Standard living room canvas |
| **4:4** | 4×4 Blocks | $256 \times 256$ px | `pointer.png` / `burningskull.png` | Monumental centerpiece murals |

---

## 3. Fabric Mod Compatibility
* **Vanilla Minecraft (Java):** 100% Native compatibility via Resource Pack (`pack.mcmeta` format 15-34, versions 1.20 through 1.21+).
* **Fabric Loader:** Fully compatible with Fabric Loader & Fabric API without any mod conflicts or Java version crashes.
* **Shaders:** Fully compatible with Iris / Sodium / VulkanMod shaders.
