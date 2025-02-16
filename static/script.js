const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const submitBtn = document.getElementById("submitBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const wasteType = document.getElementById("wasteType");
const confidence = document.getElementById("confidence");
const processingTime = document.getElementById("processingTime");

// Prevent default drag behaviors
["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Handle drag & drop
dropZone.addEventListener("dragenter", () =>
  dropZone.classList.add("drag-over")
);
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("drag-over")
);
dropZone.addEventListener("drop", (e) => {
  dropZone.classList.remove("drag-over");
  handleFiles(e.dataTransfer.files);
});

// Handle file selection
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

function handleFiles(files) {
  if (files.length > 0) {
    const file = files[0];
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.add("show");
        submitBtn.classList.add("show");
        results.style.display = "none";
      };
      reader.readAsDataURL(file);
    } else {
      alert("Please upload an image file");
    }
  }
}

// Handle form submission
submitBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  loading.style.display = "block";
  submitBtn.disabled = true;
  results.style.display = "none";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.error) throw new Error(data.error);

    wasteType.textContent = data.class;
    confidence.textContent = data.confidence;
    processingTime.textContent = data.processing_time;

    results.style.display = "block";
  } catch (error) {
    alert("Error: " + error.message);
  } finally {
    loading.style.display = "none";
    submitBtn.disabled = false;
  }
});

// WebSocket connection for real-time IoT updates
const ws = new WebSocket("ws://your-server-address:port/ws");

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);

  // If this is an IoT-triggered classification
  if (data.source === "iot") {
    // Update UI with IoT image
    previewImage.src = `data:image/jpeg;base64,${data.image}`;
    previewContainer.classList.add("show");

    // Update results
    wasteType.textContent = data.class;
    confidence.textContent = data.confidence;
    processingTime.textContent = data.processing_time;

    results.style.display = "block";

    // Add IoT indicator
    const iotIndicator = document.createElement("div");
    iotIndicator.className = "result-card";
    iotIndicator.innerHTML = `
            <div class="result-label">Source</div>
            <div class="result-value">IoT Device</div>
        `;
    results.appendChild(iotIndicator);
  }
};

// Add error handling for WebSocket
ws.onerror = function (error) {
  console.error("WebSocket error:", error);
};

ws.onclose = function () {
  console.log("WebSocket connection closed");
  // Implement reconnection logic if needed
};
