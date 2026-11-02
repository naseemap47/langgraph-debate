document.addEventListener("DOMContentLoaded", () => {
  const cdpStatusEl = document.getElementById("cdp-status");
  const refreshCdpBtn = document.getElementById("refresh-cdp-btn");
  const startBtn = document.getElementById("start-debate-btn");
  const topicInput = document.getElementById("topic-input");
  const judgeSelect = document.getElementById("judge-select");
  const synthesisSection = document.getElementById("synthesis-section");
  const synthesisContent = document.getElementById("synthesis-content");
  const copyBtn = document.getElementById("copy-report-btn");

  let rawSynthesisMarkdown = "";

  async function checkCdp() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.connected) {
        cdpStatusEl.className = "status-pill status-online";
        cdpStatusEl.textContent = `● Chrome CDP Online (${data.raw_tab_count} tabs)`;
      } else {
        cdpStatusEl.className = "status-pill status-offline";
        cdpStatusEl.textContent = "● Chrome CDP Offline";
      }
    } catch {
      cdpStatusEl.className = "status-pill status-offline";
      cdpStatusEl.textContent = "● API Server Offline";
    }
  }

  refreshCdpBtn.addEventListener("click", checkCdp);
  checkCdp();

  function updateStep(stage) {
    document.querySelectorAll(".step").forEach(el => el.classList.remove("active", "completed"));
    if (stage === "init") {
      document.getElementById("step-1").classList.add("active");
    } else if (stage === "critique") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("active");
    } else if (stage === "synthesis") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("completed");
      document.getElementById("step-3").classList.add("active");
    } else if (stage === "completed") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("completed");
      document.getElementById("step-3").classList.add("completed");
    }
  }

  startBtn.addEventListener("click", async () => {
    const topic = topicInput.value.trim();
    if (!topic) {
      alert("Please enter a debate topic.");
      return;
    }

    const selectedModels = Array.from(document.querySelectorAll(".chips input:checked")).map(cb => cb.value);
    if (selectedModels.length < 2) {
      alert("Please select at least 2 models.");
      return;
    }

    startBtn.disabled = true;
    startBtn.textContent = "Debating...";
    synthesisSection.style.display = "none";

    selectedModels.forEach(m => {
      const body = document.getElementById(`content-${m}`);
      const status = document.getElementById(`status-${m}`);
      if (body) body.innerHTML = "<em>Waiting for turn...</em>";
      if (status) status.textContent = "Queued";
    });

    try {
      const res = await fetch("/api/debate/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic,
          models: selectedModels,
          judge: judgeSelect.value,
        }),
      });
      const data = await res.json();
      const debateId = data.debate_id;

      const eventSource = new EventSource(`/api/debate/${debateId}/events`);
      eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        handleDebateEvent(payload);
        if (payload.event === "synthesis_done" || payload.event === "fatal_error") {
          eventSource.close();
          startBtn.disabled = false;
          startBtn.textContent = "Start Debate";
        }
      };
    } catch (err) {
      alert(`Error starting debate: ${err.message}`);
      startBtn.disabled = false;
      startBtn.textContent = "Start Debate";
    }
  });

  function handleDebateEvent(data) {
    if (data.event === "stage_change") {
      updateStep(data.stage);
    } else if (data.event === "model_start") {
      const status = document.getElementById(`status-${data.model}`);
      if (status) status.textContent = "Thinking...";
    } else if (data.event === "model_done") {
      const status = document.getElementById(`status-${data.model}`);
      const body = document.getElementById(`content-${data.model}`);
      if (status) status.textContent = "Submitted";
      if (body) body.innerHTML = marked.parse(data.content);
    } else if (data.event === "model_error") {
      const status = document.getElementById(`status-${data.model}`);
      const body = document.getElementById(`content-${data.model}`);
      if (status) status.textContent = "Failed";
      if (body) body.innerHTML = `<span style="color:#ef4444">Error: ${data.content}</span>`;
    } else if (data.event === "synthesis_done") {
      updateStep("completed");
      rawSynthesisMarkdown = data.content;
      synthesisContent.innerHTML = marked.parse(data.content);
      synthesisSection.style.display = "block";
      synthesisSection.scrollIntoView({ behavior: "smooth" });
    }
  }

  copyBtn.addEventListener("click", () => {
    if (rawSynthesisMarkdown) {
      navigator.clipboard.writeText(rawSynthesisMarkdown);
      copyBtn.textContent = "Copied!";
      setTimeout(() => copyBtn.textContent = "Copy Report", 2000);
    }
  });
});
