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
  document.getElementById("result-empty-state").classList.toggle("hidden", state !== "empty");
  document.getElementById("result-loading-state").classList.toggle("hidden", state !== "loading");
  document.getElementById("result-image-state").classList.toggle("hidden", state !== "image");
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

    if (currentMode === "text") {
      const prompt = document.getElementById("prompt-input").value.trim();
      response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt, mode: "text" })
      });
    } else {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("mode", "image");
      response = await fetch("/api/generate", {
        method: "POST",
        body: formData
      });
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
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
      downloadEl.setAttribute("download", "mineart_diffusion_32x32.png");
    }
    
    showResultView("image");

  } catch (err) {
    console.error("Generation Error:", err);
    showResultView("empty");
    showStatus(`Generation request: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}
