const state = {
  variant: "speech_only",
  results: null,
  plotVariant: "speech_only",
  recordingFile: null,
  recordingPreviewUrl: null,
  recordingActive: false,
};

const variantNames = {
  speech_only: "Speech-only",
  text_only: "Text-only",
  multimodal_fusion: "Multimodal fusion",
};

const form = document.querySelector("#predictForm");
const audioInput = document.querySelector("#audioInput");
const transcriptInput = document.querySelector("#transcriptInput");
const fileName = document.querySelector("#fileName");
const message = document.querySelector("#formMessage");
const predictionOutput = document.querySelector("#predictionOutput");
const predictionLabel = document.querySelector("#predictionLabel");
const confidenceList = document.querySelector("#confidenceList");
const statusStrip = document.querySelector("#modelStatus");
const uploadZone = document.querySelector("#uploadZone");
const recordCard = document.querySelector("#recordCard");
const recordStart = document.querySelector("#recordStart");
const recordStop = document.querySelector("#recordStop");
const recordReset = document.querySelector("#recordReset");
const recordTimer = document.querySelector("#recordTimer");
const recordStatus = document.querySelector("#recordStatus");
const recordPreview = document.querySelector("#recordPreview");

let recordingStream = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordingTimerId = null;
let recordingStartedAt = 0;

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function setMessage(text, isError = true) {
  message.textContent = text;
  message.style.color = isError ? "#b91c1c" : "#166534";
}

function setUploadState(active) {
  if (!uploadZone) return;
  uploadZone.classList.toggle("is-dragover", active);
}

function formatTimer(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function stopTimer() {
  if (recordingTimerId) {
    window.clearInterval(recordingTimerId);
    recordingTimerId = null;
  }
}

function updateRecordingTimer() {
  recordTimer.textContent = formatTimer(Date.now() - recordingStartedAt);
}

function setRecordedFile(file) {
  state.recordingFile = file;
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  audioInput.files = dataTransfer.files;
  fileName.textContent = file.name;
  recordReset.disabled = false;
}

function clearRecordedFile() {
  state.recordingFile = null;
  audioInput.value = "";
  fileName.textContent = "Select a TESS-style WAV file.";
  recordReset.disabled = true;
}

function clearRecordedPreview() {
  if (state.recordingPreviewUrl) {
    URL.revokeObjectURL(state.recordingPreviewUrl);
    state.recordingPreviewUrl = null;
  }
  recordPreview.hidden = true;
  recordPreview.removeAttribute("src");
}

function setRecordingUi(active) {
  state.recordingActive = active;
  recordCard.classList.toggle("recording", active);
  recordStart.disabled = active;
  recordStop.disabled = !active;
  recordReset.disabled = active || !state.recordingFile;
  recordStatus.textContent = active
    ? "Recording in progress..."
    : state.recordingFile
      ? "Recording saved and ready for prediction."
      : "Use your microphone to record a live speech sample.";
}

function setVariant(variant) {
  state.variant = variant;
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.variant === variant);
  });
  const needsAudio = variant !== "text_only";
  const needsText = variant !== "speech_only";
  document.querySelector(".audio-field").classList.toggle("hidden", !needsAudio);
  document.querySelector(".text-field").classList.toggle("hidden", !needsText);
  predictionOutput.hidden = true;
  setMessage("");
}

function renderAccuracy() {
  const rows = state.results.accuracy || [];
  const table = document.querySelector("#accuracyTable");
  table.innerHTML = rows
    .map((row) => `<tr><td>${variantNames[row.variant]}</td><td>${percent(row.accuracy)}</td></tr>`)
    .join("");

  const byVariant = Object.fromEntries(rows.map((row) => [row.variant, row.accuracy]));
  document.querySelector("#speechAccuracy").textContent = percent(byVariant.speech_only ?? 0);
  document.querySelector("#textAccuracy").textContent = percent(byVariant.text_only ?? 0);
  document.querySelector("#fusionAccuracy").textContent = percent(byVariant.multimodal_fusion ?? 0);
}

function renderReport(variant) {
  const rows = state.results.reports[variant] || [];
  const visibleRows = rows.filter((row) => row["Unnamed: 0"] && !["accuracy", "macro avg", "weighted avg"].includes(row["Unnamed: 0"]));
  document.querySelector("#reportTable").innerHTML = visibleRows
    .map(
      (row) => `
      <tr>
        <td>${row["Unnamed: 0"].replaceAll("_", " ")}</td>
        <td>${Number(row.precision).toFixed(3)}</td>
        <td>${Number(row.recall).toFixed(3)}</td>
        <td>${Number(row["f1-score"]).toFixed(3)}</td>
      </tr>
    `,
    )
    .join("");
}

function renderPlots(variant) {
  state.plotVariant = variant;
  document.querySelectorAll(".plot-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.plot === variant);
  });
  const plots = state.results.plots[variant];
  document.querySelector("#confusionPlot").src = `${plots.confusion}?v=${Date.now()}`;
  document.querySelector("#clusterPlot").src = `${plots.clusters}?v=${Date.now()}`;
}

function renderPrediction(data) {
  predictionLabel.textContent = data.label;
  predictionLabel.style.color = data.tone;
  confidenceList.innerHTML = data.probabilities
    .map(
      (row) => `
      <div class="confidence-row">
        <span>${row.label}</span>
        <div class="bar"><span style="width: ${Math.max(row.probability * 100, 2)}%; background: ${row.tone}"></span></div>
        <span>${percent(row.probability)}</span>
      </div>
    `,
    )
    .join("");
  predictionOutput.hidden = false;
}

async function loadResults() {
  const [healthResponse, resultResponse] = await Promise.all([fetch("/api/health"), fetch("/api/results")]);
  const health = await healthResponse.json();
  state.results = await resultResponse.json();
  statusStrip.lastElementChild.textContent = health.ready ? "All trained model artifacts are ready." : `Missing artifacts: ${health.missing.join(", ")}`;
  if (!health.ready) {
    statusStrip.querySelector(".pulse").style.background = "#ef4444";
  }
  renderAccuracy();
  renderReport("speech_only");
  renderPlots("speech_only");
}

function pickRecorderOptions() {
  const mimeTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  if (!window.MediaRecorder) return {};
  const mimeType = mimeTypes.find((type) => MediaRecorder.isTypeSupported(type));
  return mimeType ? { mimeType } : {};
}

async function blobToWavBlob(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("This browser cannot decode recorded audio.");
  }
  const audioContext = new AudioContextCtor();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    if (!audioBuffer.length) {
      throw new Error("Recording is empty.");
    }
    const wavBuffer = encodeWavBuffer(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
  } finally {
    await audioContext.close().catch(() => {});
  }
}

function encodeWavBuffer(audioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const sampleCount = audioBuffer.length;
  const monoSamples = new Float32Array(sampleCount);

  for (let channel = 0; channel < channelCount; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let index = 0; index < sampleCount; index += 1) {
      monoSamples[index] += channelData[index] / channelCount;
    }
  }

  const bytesPerSample = 2;
  const dataSize = sampleCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let index = 0; index < monoSamples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, monoSamples[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += bytesPerSample;
  }

  return buffer;
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setMessage("Recording is not supported in this browser.");
    return;
  }

  try {
    clearRecordedPreview();
    clearRecordedFile();
    recordTimer.textContent = "00:00";
    recordStatus.textContent = "Requesting microphone access...";
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorderOptions = pickRecorderOptions();
    mediaRecorder = new MediaRecorder(recordingStream, recorderOptions);
    recordingChunks = [];

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        recordingChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      stopTimer();
      setRecordingUi(false);
      const combinedBlob = new Blob(recordingChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
      recordingChunks = [];

      try {
        if (combinedBlob.size < 1000) {
          throw new Error("Recording is empty.");
        }
        const wavBlob = await blobToWavBlob(combinedBlob);
        const recordedFile = new File([wavBlob], `voice-recording-${Date.now()}.wav`, { type: "audio/wav" });
        setRecordedFile(recordedFile);
        clearRecordedPreview();
        state.recordingPreviewUrl = URL.createObjectURL(wavBlob);
        recordPreview.src = state.recordingPreviewUrl;
        recordPreview.hidden = false;
        recordStatus.textContent = "Recording ready for prediction.";
        setMessage("Recording saved. You can run prediction now.", false);
      } catch (error) {
        clearRecordedFile();
        clearRecordedPreview();
        recordStatus.textContent = "Recording failed. Try again.";
        setMessage(error.message || "Recording failed.");
      } finally {
        recordingStream?.getTracks().forEach((track) => track.stop());
        recordingStream = null;
        mediaRecorder = null;
      }
    });

    mediaRecorder.start();
    recordingStartedAt = Date.now();
    updateRecordingTimer();
    stopTimer();
    recordingTimerId = window.setInterval(updateRecordingTimer, 250);
    setRecordingUi(true);
    setMessage("Recording started. Speak naturally for 3 to 10 seconds.", false);
  } catch (error) {
    recordingStream?.getTracks().forEach((track) => track.stop());
    recordingStream = null;
    mediaRecorder = null;
    stopTimer();
    setRecordingUi(false);
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      setMessage("Microphone permission was denied.");
    } else {
      setMessage(error.message || "Unable to start recording.");
    }
    recordStatus.textContent = "Microphone access is required to record voice samples.";
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }
  recordStatus.textContent = "Finishing recording...";
  mediaRecorder.stop();
}

function resetRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  recordingStream?.getTracks().forEach((track) => track.stop());
  recordingStream = null;
  mediaRecorder = null;
  recordingChunks = [];
  stopTimer();
  recordTimer.textContent = "00:00";
  clearRecordedPreview();
  clearRecordedFile();
  setRecordingUi(false);
  recordStatus.textContent = "Use your microphone to record a live speech sample.";
}

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => setVariant(button.dataset.variant));
});

document.querySelectorAll(".plot-tab").forEach((button) => {
  button.addEventListener("click", () => renderPlots(button.dataset.plot));
});

document.querySelector("#reportSelect").addEventListener("change", (event) => {
  renderReport(event.target.value);
});

audioInput.addEventListener("change", () => {
  if (audioInput.files.length) {
    clearRecordedPreview();
    state.recordingFile = audioInput.files[0];
    fileName.textContent = audioInput.files[0].name;
    recordStatus.textContent = "Uploaded audio ready for prediction.";
    recordReset.disabled = false;
  } else {
    clearRecordedFile();
    recordStatus.textContent = "Use your microphone to record a live speech sample.";
  }
});

if (uploadZone) {
  ["dragenter", "dragover"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      setUploadState(true);
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      setUploadState(false);
    });
  });

  uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    audioInput.files = dataTransfer.files;
    audioInput.dispatchEvent(new Event("change"));
  });
}

recordStart?.addEventListener("click", startRecording);
recordStop?.addEventListener("click", stopRecording);
recordReset?.addEventListener("click", resetRecording);

if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
  recordStart.disabled = true;
  recordStatus.textContent = "Recording is not supported in this browser.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("");

  const needsAudio = state.variant !== "text_only";
  const needsText = state.variant !== "speech_only";
  if (needsAudio && !audioInput.files.length) {
    setMessage("Upload or record a WAV audio file before running this model.");
    return;
  }
  if (needsText && !transcriptInput.value.trim()) {
    setMessage("Enter the transcript word before running this model.");
    return;
  }

  const button = form.querySelector(".primary-action");
  const label = button.querySelector(".button-label");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  label.textContent = "Predicting...";

  const payload = new FormData();
  payload.append("variant", state.variant);
  payload.append("transcript", transcriptInput.value.trim());
  if (audioInput.files.length) {
    payload.append("audio", audioInput.files[0]);
  }

  try {
    const response = await fetch("/api/predict", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    renderPrediction(data);
    setMessage(`${variantNames[data.variant]} prediction completed.`, false);
  } catch (error) {
    predictionOutput.hidden = true;
    setMessage(error.message);
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    label.textContent = "Run prediction";
  }
});

setVariant("speech_only");
setRecordingUi(false);
loadResults().catch((error) => {
  statusStrip.lastElementChild.textContent = "Unable to load model results.";
  statusStrip.querySelector(".pulse").style.background = "#ef4444";
  setMessage(error.message);
});
