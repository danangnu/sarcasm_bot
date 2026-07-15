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
    const timestamp = () => new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" }).format(new Date());

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
            await navigator.clipboard.writeText(text);
            copy.textContent = "Copied";
            setTimeout(() => { copy.textContent = "Copy"; }, 1200);
        });
        meta.append(time, copy);
        msgDiv.appendChild(meta);
    }

    function appendMessage(text, sender, scores = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text;
        msgDiv.appendChild(bubble);

        if (scores) {
            const scoreCard = document.createElement("div");
            scoreCard.className = "score-card";
            const rows = [];
            if (scores.user_sarcasm_score !== undefined) {
                rows.push(`<span><b>Your result:</b> ${formatPercent(scores.user_sarcasm_score)} · ${scores.user_is_sarcastic ? "Sarcastic" : "Not sarcastic"} · ${scores.user_confidence || "—"} confidence</span>`);
            }
            if (scores.user_explanation) rows.push(`<span class="explanation">${scores.user_explanation}</span>`);
            if (scores.reply_sarcasm_score !== undefined) {
                rows.push(`<span><b>Bot reply:</b> ${formatPercent(scores.reply_sarcasm_score)} · ${scores.reply_is_sarcastic ? "Sarcastic" : "Not sarcastic"} · ${scores.reply_confidence || "—"} confidence</span>`);
            }
            if (scores.response_source) {
                rows.push(`<span><b>Source:</b> ${scores.response_source}${scores.generation_attempts ? ` · ${scores.generation_attempts} attempt(s)` : ""}</span>`);
            }
            if (scores.fallback_reason) rows.push(`<span class="fallback-note"><b>Fallback reason:</b> ${scores.fallback_reason}</span>`);
            scoreCard.innerHTML = rows.join("");
            msgDiv.appendChild(scoreCard);
        }

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
        if (!response.ok) throw new Error(data.detail || `Request failed (${response.status}).`);
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
            appendMessage(`Analysis: ${data.label} (${formatPercent(data.score)}).`, "bot", {
                user_sarcasm_score: data.score,
                user_is_sarcastic: data.is_sarcastic,
                user_confidence: data.confidence,
                user_explanation: data.explanation
            });
            userInput.value = "";
            updateCounter();
        } catch (error) {
            removeTypingIndicator();
            appendMessage(`Analyze error: ${error.message}`, "bot");
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
        } catch (error) {
            removeTypingIndicator();
            appendMessage(`Chat error: ${error.message}`, "bot");
        } finally {
            setBusy(false);
            userInput.focus();
        }
    }

    function resetChat() {
        chatBox.innerHTML = "";
        appendMessage("Paste a tweet, article paragraph, headline, or message. I will estimate the sarcasm and can generate a reply.", "bot");
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
