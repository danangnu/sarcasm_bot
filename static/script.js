document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const themeBtn = document.getElementById("theme-btn");
    const charCount = document.getElementById("char-count");
    const apiStatus = document.getElementById("api-status");
    const geminiStatus = document.getElementById("gemini-status");
    const sampleButtons = document.querySelectorAll(".sample-btn");
    const MAX_CHARS = 8000;

    const formatPercent = (value) => `${Math.round(Number(value) * 100)}%`;
    const timestamp = () => new Intl.DateTimeFormat([], {
        hour: "2-digit",
        minute: "2-digit"
    }).format(new Date());

    function updateCounter() {
        const length = userInput.value.length;
        charCount.textContent = `${length} / ${MAX_CHARS}`;
        charCount.classList.toggle("near-limit", length > MAX_CHARS * 0.9);
    }

    function setBusy(isBusy) {
        userInput.disabled = isBusy;
        sendBtn.disabled = isBusy;
        analyzeBtn.disabled = isBusy;
        clearBtn.disabled = isBusy;
        sendBtn.textContent = isBusy ? "Working..." : "Send to bot";
    }

    function addMessageActions(msgDiv, text) {
        const meta = document.createElement("div");
        meta.className = "message-meta";
        const time = document.createElement("span");
        time.textContent = timestamp();
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "copy-btn";
        copy.textContent = "Copy";
        copy.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(text);
                copy.textContent = "Copied";
            } catch {
                copy.textContent = "Copy failed";
            }
            setTimeout(() => { copy.textContent = "Copy"; }, 1200);
        });
        meta.append(time, copy);
        msgDiv.appendChild(meta);
    }

    function createResultLine(label, score, isSarcastic, confidence) {
        const line = document.createElement("div");
        line.className = "result-line";

        const labelEl = document.createElement("span");
        labelEl.className = "result-label";
        labelEl.textContent = label;

        const valueEl = document.createElement("span");
        valueEl.className = "result-value";
        valueEl.textContent = `${formatPercent(score)} · ${isSarcastic ? "Sarcastic" : "Not sarcastic"} · ${confidence || "—"} confidence`;

        line.append(labelEl, valueEl);
        return line;
    }

    function sourceBadge(source) {
        const normalized = (source || "Analyze only").toLowerCase();
        const badge = document.createElement("span");
        badge.className = "source-badge";
        if (normalized.includes("gemini")) badge.classList.add("source-gemini");
        else if (normalized.includes("fallback")) badge.classList.add("source-fallback");
        else badge.classList.add("source-analysis");
        badge.textContent = source || "Analyze only";
        return badge;
    }

    function createScoreCard(scores) {
        const scoreCard = document.createElement("div");
        scoreCard.className = "score-card";

        if (scores.user_sarcasm_score !== undefined) {
            scoreCard.appendChild(createResultLine(
                "Your text",
                scores.user_sarcasm_score,
                scores.user_is_sarcastic,
                scores.user_confidence
            ));
        }

        if (scores.user_explanation) {
            const explanation = document.createElement("p");
            explanation.className = "explanation";
            explanation.textContent = scores.user_explanation;
            scoreCard.appendChild(explanation);
        }

        if (scores.reply_sarcasm_score !== undefined) {
            scoreCard.appendChild(createResultLine(
                "Bot response",
                scores.reply_sarcasm_score,
                scores.reply_is_sarcastic,
                scores.reply_confidence
            ));
        }

        const sourceRow = document.createElement("div");
        sourceRow.className = "source-row";
        const sourceLabel = document.createElement("span");
        sourceLabel.className = "result-label";
        sourceLabel.textContent = "Response source";
        sourceRow.append(sourceLabel, sourceBadge(scores.response_source));
        scoreCard.appendChild(sourceRow);

        const fallbackMessage = scores.fallback_message || scores.fallback_reason;
        if (fallbackMessage) {
            const fallback = document.createElement("p");
            fallback.className = "fallback-note";
            fallback.textContent = fallbackMessage;
            scoreCard.appendChild(fallback);
        }

        return scoreCard;
    }

    function appendMessage(text, sender, scores = null, variant = "") {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;
        const bubble = document.createElement("div");
        bubble.className = `bubble ${variant}`.trim();
        bubble.textContent = text;
        msgDiv.appendChild(bubble);

        if (scores) msgDiv.appendChild(createScoreCard(scores));

        addMessageActions(msgDiv, text);
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showTypingIndicator(label = "Thinking") {
        const item = document.createElement("div");
        item.className = "message bot-message";
        item.id = "active-typing-indicator";
        item.innerHTML = `<div class="bubble typing-indicator"><span>${label}</span><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
        chatBox.appendChild(item);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeTypingIndicator() {
        document.getElementById("active-typing-indicator")?.remove();
    }

    function normalizeApiError(data, status) {
        if (Array.isArray(data.detail)) {
            return data.detail.map((item) => item.msg).filter(Boolean).join(" ") || "The request was not valid.";
        }
        if (typeof data.detail === "string") return data.detail;
        if (status === 422) return "The message is empty or too long.";
        if (status >= 500) return "The server could not complete the request. Check the backend log.";
        return `Request failed (${status}).`;
    }

    async function postJson(url, body) {
        let response;
        try {
            response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
        } catch {
            throw new Error("Cannot reach the FastAPI server.");
        }
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(normalizeApiError(data, response.status));
        return data;
    }

    async function loadHealth() {
        try {
            const response = await fetch("/health");
            const data = await response.json();
            const ready = data.model_ready && data.tokenizer_ready;
            apiStatus.textContent = ready ? "Ready" : "Needs setup";
            apiStatus.className = ready ? "status-ok" : "status-warn";
            geminiStatus.textContent = data.gemini_configured ? `Configured · ${data.gemini_model}` : "Local fallback";
            geminiStatus.className = data.gemini_configured ? "status-ok" : "status-warn";
        } catch {
            apiStatus.textContent = "Offline";
            apiStatus.className = "status-error";
            geminiStatus.textContent = "Unknown";
            geminiStatus.className = "status-error";
        }
    }

    function updateGeminiRuntimeStatus(data) {
        if (data.response_source === "Gemini") {
            geminiStatus.textContent = "Active · Gemini";
            geminiStatus.className = "status-ok";
            return;
        }
        if (data.fallback_code === "quota_error") {
            geminiStatus.textContent = "Quota unavailable · fallback active";
        } else if (data.fallback_code === "authentication_error") {
            geminiStatus.textContent = "API key problem · fallback active";
        } else {
            geminiStatus.textContent = "Fallback active";
        }
        geminiStatus.className = "status-warn";
    }

    function getText() {
        const text = userInput.value.trim();
        if (!text) {
            userInput.focus();
            return null;
        }
        return text;
    }

    async function analyzeOnly() {
        const text = getText();
        if (!text) return;
        appendMessage(text, "user");
        setBusy(true);
        showTypingIndicator("Scoring");
        try {
            const data = await postJson("/analyze", { message: text });
            removeTypingIndicator();
            appendMessage(`Analysis complete: ${data.label}.`, "bot", {
                user_sarcasm_score: data.score,
                user_is_sarcastic: data.is_sarcastic,
                user_confidence: data.confidence,
                user_explanation: data.explanation,
                response_source: "Analyze only"
            });
            userInput.value = "";
            updateCounter();
        } catch (error) {
            removeTypingIndicator();
            appendMessage(error.message, "bot", null, "error-bubble");
        } finally {
            setBusy(false);
            userInput.focus();
        }
    }

    async function sendMessage() {
        const text = getText();
        if (!text) return;
        appendMessage(text, "user");
        userInput.value = "";
        updateCounter();
        setBusy(true);
        showTypingIndicator("Generating reply");
        try {
            const data = await postJson("/chat", { message: text });
            removeTypingIndicator();
            appendMessage(data.reply, "bot", data);
            updateGeminiRuntimeStatus(data);
        } catch (error) {
            removeTypingIndicator();
            appendMessage(error.message, "bot", null, "error-bubble");
        } finally {
            setBusy(false);
            userInput.focus();
        }
    }

    function resetChat() {
        chatBox.innerHTML = "";
        appendMessage(
            "Paste a tweet, article paragraph, headline, or message. I will estimate the sarcasm and can generate a reply.",
            "bot"
        );
    }

    sendBtn.addEventListener("click", sendMessage);
    analyzeBtn.addEventListener("click", analyzeOnly);
    clearBtn.addEventListener("click", resetChat);
    themeBtn.addEventListener("click", () => {
        document.body.classList.toggle("light-theme");
        themeBtn.textContent = document.body.classList.contains("light-theme") ? "☾" : "☀";
    });
    userInput.addEventListener("input", updateCounter);
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    sampleButtons.forEach((button) => button.addEventListener("click", () => {
        userInput.value = button.dataset.sample;
        updateCounter();
        userInput.focus();
    }));

    resetChat();
    updateCounter();
    loadHealth();
    userInput.focus();
});
