document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");
    const themeBtn = document.getElementById("theme-btn");
    const apiStatus = document.getElementById("api-status");
    const geminiStatus = document.getElementById("gemini-status");
    const feedbackCount = document.getElementById("feedback-count");
    const retrainBtn = document.getElementById("retrain-btn");
    const retrainStatus = document.getElementById("retrain-status");
    const charCount = document.getElementById("char-count");
    const sampleButtons = document.querySelectorAll(".sample-btn");
    let busy = false;

    const formatPercent = (value) => `${Math.round(Number(value || 0) * 100)}%`;
    const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    function updateCounter() { charCount.textContent = `${userInput.value.length} / 8000`; }
    function setBusy(value) {
        busy = value;
        analyzeBtn.disabled = value;
        sendBtn.disabled = value;
        userInput.disabled = value;
    }

    function addMessageActions(container, text) {
        const meta = document.createElement("div");
        meta.className = "message-meta";
        const time = document.createElement("span");
        time.textContent = now();
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "copy-btn";
        copy.textContent = "Copy";  
        copy.addEventListener("click", async () => {
            try { await navigator.clipboard.writeText(text); copy.textContent = "Copied"; }
            catch { copy.textContent = "Copy failed"; }
            setTimeout(() => copy.textContent = "Copy", 1200);
        });
        meta.append(time, copy);
        container.appendChild(meta);
    }

    function resultLabel(scores, prefix) {
        return scores[`${prefix}_label`] || (scores[`${prefix}_is_sarcastic`] === true ? "Sarcastic" : scores[`${prefix}_is_sarcastic`] === false ? "Not sarcastic" : "Ambiguous");
    }

    function createResultLine(title, score, label, confidence) {
        const line = document.createElement("div");
        line.className = "result-line";
        const name = document.createElement("span");
        name.className = "result-label";
        name.textContent = title;
        const value = document.createElement("span");
        value.className = "result-value";
        value.textContent = `${formatPercent(score)} · ${label} · ${confidence || "—"} classification confidence`;
        line.append(name, value);
        return line;
    }

    function sourceBadge(source) {
        const badge = document.createElement("span");
        const normalized = (source || "Analyze only").toLowerCase();
        badge.className = "source-badge";
        if (normalized.includes("gemini")) badge.classList.add("source-gemini");
        else if (normalized.includes("fallback")) badge.classList.add("source-fallback");
        else badge.classList.add("source-analysis");
        badge.textContent = source || "Analyze only";
        return badge;
    }

    function createFeedbackPanel(context) {
        const panel = document.createElement("div");
        panel.className = "feedback-panel";

        const prompt = document.createElement("span");
        prompt.textContent = "Is this prediction incorrect?";

        const incorrect = document.createElement("button");
        incorrect.type = "button";
        incorrect.className = "feedback-incorrect";
        incorrect.textContent = "Correct this result";

        const correction = document.createElement("div");
        correction.className = "correction-panel";
        correction.hidden = true;
        correction.innerHTML = `
            <label>Correct label
                <select>
                    <option value="not_sarcastic">Not sarcastic</option>
                    <option value="sarcastic">Sarcastic</option>
                    <option value="insufficient_context">Insufficient context</option>
                </select>
            </label>
            <button type="button" class="save-correction">Save correction</button>
            <span class="feedback-message" aria-live="polite"></span>`;

        incorrect.addEventListener("click", () => {
            const isOpen = !correction.hidden;
            correction.hidden = isOpen;
            incorrect.textContent = isOpen ? "Correct this result" : "Cancel correction";
        });

        correction.querySelector(".save-correction").addEventListener("click", async () => {
            const message = correction.querySelector(".feedback-message");
            try {
                const data = await postJson("/feedback", {
                    text: context.text,
                    predicted_label: context.label,
                    predicted_score: context.score,
                    correct_label: correction.querySelector("select").value,
                    source: context.source || "analyze"
                });

                prompt.textContent = "Correction saved for the next retraining cycle.";
                message.textContent = data.message || "Saved.";
                feedbackCount.textContent = data.feedback_count;
                incorrect.disabled = true;
                correction.querySelector("button").disabled = true;
                correction.querySelector("select").disabled = true;
            } catch (error) {
                message.textContent = error.message;
            }
        });

        panel.append(prompt, incorrect, correction);
        return panel;
    }

    function createScoreCard(scores, feedbackContext = null) {
        const card = document.createElement("div");
        card.className = "score-card";
        if (scores.user_sarcasm_score !== undefined) {
            card.appendChild(createResultLine("Your text", scores.user_sarcasm_score, resultLabel(scores, "user"), scores.user_confidence));
        }
        if (scores.user_explanation) {
            const explanation = document.createElement("p");
            explanation.className = "explanation";
            explanation.textContent = scores.user_explanation;
            card.appendChild(explanation);
        }
        // Keep feedback directly below the input classification so it remains
        // visible even when the result card is taller than the chat viewport.
        if (feedbackContext) card.appendChild(createFeedbackPanel(feedbackContext));
        if (scores.reply_sarcasm_score !== undefined) {
            card.appendChild(createResultLine("Generated reply", scores.reply_sarcasm_score, resultLabel(scores, "reply"), scores.reply_confidence));
        }
        const row = document.createElement("div");
        row.className = "source-row";
        const label = document.createElement("span");
        label.className = "result-label";
        label.textContent = "Response source";
        row.append(label, sourceBadge(scores.response_source));
        card.appendChild(row);
        const fallback = scores.fallback_message || scores.fallback_reason;
        if (fallback) {
            const note = document.createElement("p");
            note.className = "fallback-note";
            note.textContent = fallback;
            card.appendChild(note);
        }
        return card;
    }

    function appendMessage(text, sender, scores = null, variant = "", feedbackContext = null) {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;
        const bubble = document.createElement("div");
        bubble.className = `bubble ${variant}`.trim();
        bubble.textContent = text;
        wrapper.appendChild(bubble);
        if (scores) wrapper.appendChild(createScoreCard(scores, feedbackContext));
        addMessageActions(wrapper, text);
        chatBox.appendChild(wrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
        if (feedbackContext) {
            requestAnimationFrame(() => {
                const panel = wrapper.querySelector(".feedback-panel");
                panel?.scrollIntoView({ behavior: "smooth", block: "center" });
            });
        }
    }

    function showTyping(label) {
        const item = document.createElement("div");
        item.id = "active-typing-indicator";
        item.className = "message bot-message";
        item.innerHTML = `<div class="bubble typing-indicator"><span>${label}</span><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
        chatBox.appendChild(item);
    }
    function removeTyping() { document.getElementById("active-typing-indicator")?.remove(); }

    function normalizeApiError(data, status) {
        if (Array.isArray(data.detail)) return data.detail.map(x => x.msg).filter(Boolean).join(" ");
        if (typeof data.detail === "string") return data.detail;
        return status >= 500 ? "The server could not complete the request." : `Request failed (${status}).`;
    }
    async function postJson(url, body = {}) {
        let response;
        try { response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
        catch { throw new Error("Cannot reach the FastAPI server."); }
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
            feedbackCount.textContent = data.feedback_count || 0;
            retrainBtn.disabled = !data.retraining_available;
            retrainStatus.textContent = data.retraining_available ? (data.retraining_status?.message || "Ready for controlled retraining.") : "Retraining is offline for this release; corrections are still saved.";
        } catch {
            apiStatus.textContent = "Offline";
            apiStatus.className = "status-error";
        }
    }

    function updateGemini(data) {
        if (data.response_source === "Gemini") {
            geminiStatus.textContent = "Active · Gemini";
            geminiStatus.className = "status-ok";
        } else {
            const labels = {
                quota_error: "Quota unavailable · fallback active",
                authentication_error: "API key problem · fallback active",
                model_unavailable: "Model unavailable · fallback active",
                service_unavailable: "Service unavailable · fallback active"
            };
            geminiStatus.textContent = labels[data.fallback_code] || "Fallback active";
            geminiStatus.className = "status-warn";
        }
    }

    function getText() {
        const text = userInput.value.trim();
        if (!text) { userInput.focus(); return null; }
        return text;
    }

    async function analyzeOnly() {
        const text = getText(); if (!text || busy) return;
        appendMessage(text, "user"); setBusy(true); showTyping("Analyzing");
        try {
            const data = await postJson("/analyze", { message: text });
            removeTyping();
            appendMessage(`Analysis complete: ${data.label}.`, "bot", {
                user_sarcasm_score: data.score,
                user_is_sarcastic: data.is_sarcastic,
                user_label: data.label,
                user_confidence: data.confidence,
                user_explanation: data.explanation,
                response_source: "Analyze only"
            }, "", { text, label: data.label, score: data.score, source: "analyze" });
            userInput.value = ""; updateCounter();
        } catch (error) { removeTyping(); appendMessage(error.message, "bot", null, "error-bubble"); }
        finally { setBusy(false); userInput.focus(); }
    }

    async function generateReply() {
        const text = getText(); if (!text || busy) return;
        appendMessage(text, "user"); userInput.value = ""; updateCounter(); setBusy(true); showTyping("Generating reply");
        try {
            const data = await postJson("/chat", { message: text });
            removeTyping();
            appendMessage(data.reply, "bot", data, "", { text, label: data.user_label, score: data.user_sarcasm_score, source: "chat" });
            updateGemini(data);
        } catch (error) { removeTyping(); appendMessage(error.message, "bot", null, "error-bubble"); }
        finally { setBusy(false); userInput.focus(); }
    }

    async function retrain() {
        retrainBtn.disabled = true;
        retrainStatus.textContent = "Starting candidate retraining...";
        try {
            const data = await postJson("/admin/retrain");
            retrainStatus.textContent = data.message;
        } catch (error) {
            retrainStatus.textContent = error.message;
            retrainBtn.disabled = false;
        }
    }

    function resetChat() {
        chatBox.innerHTML = "";
        appendMessage("Paste a short headline or message. I will estimate whether it is sarcastic, ambiguous, or too short to judge. Generate uses Gemini when available.", "bot");
    }

    sendBtn.addEventListener("click", generateReply);
    analyzeBtn.addEventListener("click", analyzeOnly);
    retrainBtn.addEventListener("click", retrain);
    clearBtn.addEventListener("click", resetChat);
    themeBtn.addEventListener("click", () => {
        document.body.classList.toggle("light-theme");
        themeBtn.textContent = document.body.classList.contains("light-theme") ? "☾" : "☀";
    });
    userInput.addEventListener("input", updateCounter);
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); generateReply(); }
    });
    sampleButtons.forEach(button => button.addEventListener("click", () => {
        userInput.value = button.dataset.sample; updateCounter(); userInput.focus();
    }));

    resetChat(); updateCounter(); loadHealth(); userInput.focus();
});
