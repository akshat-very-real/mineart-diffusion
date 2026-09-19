@echo off
title MineArt - Install Painting Resource Pack to Minecraft
echo ==============================================================
echo    MineArt Diffusion: Minecraft In-Game Painting Installer
echo ==============================================================
echo.

set MC_PACK_DIR=%APPDATA%\.minecraft\resourcepacks
if not exist "%MC_PACK_DIR%" (
    echo [!] Minecraft directory not found at standard path: %MC_PACK_DIR%
    echo [*] Creating resourcepacks folder...
    mkdir "%MC_PACK_DIR%"
)

echo [*] Installing MineArt AI Paintings to Minecraft...
copy /Y "%~dp0MineArt_Pack.zip" "%MC_PACK_DIR%\MineArt_AI_Paintings.zip" >nul

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================
    echo [SUCCESS] MineArt Paintings successfully installed to Minecraft!
    echo ==============================================================
    echo.
    echo How to view in Minecraft:
    echo  1. Launch Minecraft (Vanilla or Fabric).
    echo  2. Go to: Options -^> Resource Packs.
    echo  3. Move "MineArt AI Paintings" to the Selected side.
    echo  4. In your world, place a Painting on any wall!
    echo.
    echo Tip: If Minecraft is already open, press F3 + T to reload!
    echo ==============================================================
) else (
    echo.
    echo [ERROR] Failed to copy resource pack. Please manually copy:
    echo "%~dp0MineArt_Pack.zip" to "%MC_PACK_DIR%"
)

echo.
pause
