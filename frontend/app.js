/**
 * MineArt Diffusion - Frontend
 */

let currentMode = "text";
let selectedFile = null;
let currentDownloadUrl = "/outputs/latest.png";

function setMode(mode) {
  currentMode = mode;
  const isText = mode === "text";
  
  document.getElementById("mode-text-btn").classList.toggle("active", isText);
  document.getElementById("mode-image-btn").classList.toggle("active", !isText);
  
  document.getElementById("text-input-section").classList.toggle("hidden", !isText);
  document.getElementById("image-input-section").classList.toggle("hidden", isText);
  
  clearStatus();
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("upload-preview-img").src = e.target.result;
    document.getElementById("upload-empty-state").classList.add("hidden");
    document.getElementById("upload-preview-state").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  clearStatus();
}

function clearUpload(event) {
  if (event) event.stopPropagation();
  selectedFile = null;
  document.getElementById("image-file-input").value = "";
  document.getElementById("upload-preview-img").src = "";
  document.getElementById("upload-empty-state").classList.remove("hidden");
  document.getElementById("upload-preview-state").classList.add("hidden");
  clearStatus();
}

function showStatus(msg, type = "info") {
  const el = document.getElementById("status-message");
  el.textContent = msg;
  el.className = `status-msg ${type}`;
  el.classList.remove("hidden");
}

function clearStatus() {
  const el = document.getElementById("status-message");
  el.textContent = "";
  el.classList.add("hidden");
}

function showResultView(state) {
  document.getElementById("result-empty-state")?.classList.toggle("hidden", state !== "empty");
  document.getElementById("result-loading-state")?.classList.toggle("hidden", state !== "loading");
  document.getElementById("result-image-state")?.classList.toggle("hidden", state !== "image");
  document.getElementById("result-blocked-state")?.classList.toggle("hidden", state !== "blocked");
}

function handlePromptInput() {
  const promptEl = document.getElementById("prompt-input");
  const counterEl = document.getElementById("prompt-word-counter");
  if (!promptEl || !counterEl) return;

  const text = promptEl.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  counterEl.textContent = `${words} / 16 words`;
  if (words > 16) {
    counterEl.style.color = "#dc2626";
    counterEl.style.fontWeight = "600";
  } else {
    counterEl.style.color = "var(--text-dim)";
    counterEl.style.fontWeight = "400";
  }
}

function toggleSafetyFilter() {
  const checkbox = document.getElementById("safety-filter-toggle");
  const badge = document.getElementById("guardrail-badge");
  if (!checkbox || !badge) return;

  if (checkbox.checked) {
    badge.textContent = "Active";
    badge.className = "guardrail-badge";
  } else {
    badge.textContent = "Disabled";
    badge.className = "guardrail-badge disabled";
  }
}

function applySuggestedPrompt(suggestedText) {
  const promptEl = document.getElementById("prompt-input");
  if (promptEl) {
    promptEl.value = suggestedText;
    handlePromptInput();
    clearStatus();
    showResultView("empty");
    promptEl.focus();
  }
}

async function handleGenerate() {
  clearStatus();
  
  if (currentMode === "text") {
    const prompt = document.getElementById("prompt-input").value.trim();
    if (!prompt) {
      showStatus("Please enter a text prompt to generate.", "error");
      return;
    }
  } else {
    if (!selectedFile) {
      showStatus("Please select an input image file first.", "error");
      return;
    }
  }

  const btn = document.getElementById("generate-btn");
  btn.disabled = true;
  showResultView("loading");

  try {
    let response;
    const safetyFilterEnabled = document.getElementById("safety-filter-toggle")?.checked ?? true;

    if (currentMode === "text") {
      const prompt = document.getElementById("prompt-input").value.trim();
      response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          mode: "text",
          safety_filter: safetyFilterEnabled,
        })
      });
    } else {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("mode", "image");
      formData.append("safety_filter", safetyFilterEnabled ? "true" : "false");
      response = await fetch("/api/generate", {
        method: "POST",
        body: formData
      });
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      if (errorData.blocked) {
        showResultView("blocked");
        const titleEl = document.getElementById("blocked-title");
        const reasonEl = document.getElementById("blocked-reason");

        if (titleEl) {
          if (errorData.error_type === "GIBBERISH") {
            titleEl.textContent = "⚠️ Generation Blocked: Unrecognizable / Gibberish Input";
          } else if (errorData.error_type === "HARMFUL_CONTENT") {
            titleEl.textContent = "⛔ Generation Blocked: Harmful Content";
          } else {
            titleEl.textContent = "🛡️ Generation Blocked";
          }
        }

        if (reasonEl) {
          reasonEl.textContent = errorData.detail || "Input was rejected by the safety guardrail.";
        }

        showStatus(errorData.detail || "Prompt blocked by guardrail filter.", "error");
        return;
      }
      throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    currentDownloadUrl = data.download_url || "/outputs/latest.png";
    
    // Display result image with cache-busting timestamp
    const imgEl = document.getElementById("generated-image");
    imgEl.src = data.image_url + "?t=" + Date.now();
    
    const downloadEl = document.getElementById("download-btn");
    if (downloadEl) {
      downloadEl.href = currentDownloadUrl;
      downloadEl.setAttribute("download", "mineart_painting.png");
    }
    
    showResultView("image");

  } catch (err) {
    console.error("Generation Error:", err);
    showResultView("empty");
    showStatus(err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function handleExportToMinecraft() {
  const exportBtn = document.getElementById("export-mc-btn");
  const slotSelect = document.getElementById("painting-slot-select");
  const selectedSlot = slotSelect ? slotSelect.value : "bust";
  const styleSelect = document.getElementById("painting-style-select");
  const selectedStyle = styleSelect ? styleSelect.value : "vanilla_authentic";
  const borderToggle = document.getElementById("painting-border-toggle");
  const addBorder = borderToggle ? borderToggle.checked : true;

  exportBtn.disabled = true;
  exportBtn.textContent = "Injecting Canvas Texture...";
  toast.className = "mc-toast";
  toast.textContent = "Applying wooden frame, oil canvas texture, Floyd-Steinberg dithering & injecting into Minecraft...";
  toast.classList.remove("hidden");

  try {
    const res = await fetch("/api/export-minecraft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slot: selectedSlot,
        ratio: "2:2",
        style: selectedStyle,
        add_border: addBorder
      })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Export failed.");
    }

    toast.className = "mc-toast success";
    toast.innerHTML = `<strong>✅ Successfully Imported to Minecraft!</strong><br>${data.instructions}`;

    // Switch preview to show the authentic framed canvas
    if (data.framed_url) {
      const imgEl = document.getElementById("generated-image");
      imgEl.src = data.framed_url + "?t=" + Date.now();
      const downloadEl = document.getElementById("download-btn");
      if (downloadEl) {
        downloadEl.href = data.framed_url;
        downloadEl.setAttribute("download", `mineart_framed_${data.slot || "painting"}.png`);
      }
    }
  } catch (err) {
    console.error("Minecraft Export Error:", err);
    toast.className = "mc-toast error";
    toast.textContent = `Error exporting to Minecraft: ${err.message}`;
  } finally {
    exportBtn.disabled = false;
    exportBtn.textContent = "🎮 Send to Minecraft Wall";
  }
}

// Auto-check Minecraft installation on page load
async function checkMinecraftStatus() {
  try {
    const res = await fetch("/api/minecraft-status");
    if (res.ok) {
      const data = await res.json();
      const statusText = document.getElementById("mc-status-text");
      if (statusText) {
        if (data.installed) {
          statusText.textContent = "Minecraft Linked (.minecraft)";
        } else {
          statusText.textContent = "Standalone Pack Mode";
        }
      }
    }
  } catch (e) {
    console.log("Status check skipped:", e);
  }
}

document.addEventListener("DOMContentLoaded", checkMinecraftStatus);

