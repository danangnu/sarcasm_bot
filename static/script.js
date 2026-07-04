document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const apiStatus = document.getElementById("api-status");
    const geminiStatus = document.getElementById("gemini-status");
    const sampleButtons = document.querySelectorAll(".sample-btn");

    const formatPercent = (value) => `${Math.round(Number(value) * 100)}%`;

    function setBusy(isBusy) {
        userInput.disabled = isBusy;
        sendBtn.disabled = isBusy;
        analyzeBtn.disabled = isBusy;
    }

    function appendMessage(text, sender, scores = null) {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble");
        bubbleDiv.textContent = text;
        msgDiv.appendChild(bubbleDiv);

        if (scores) {
            const scoreDiv = document.createElement("div");
            scoreDiv.classList.add("score-card");

            const rows = [];
            if (scores.user_sarcasm_score !== undefined) {
                rows.push(`Your Sarcasm: ${formatPercent(scores.user_sarcasm_score)} (${scores.user_is_sarcastic ? "Sarcastic" : "Not sarcastic"})`);
            }
            if (scores.reply_sarcasm_score !== undefined) {
                rows.push(`Bot Sarcasm: ${formatPercent(scores.reply_sarcasm_score)} (${scores.reply_is_sarcastic ? "Sarcastic" : "Not sarcastic"})`);
            }
            if (scores.used_gemini !== undefined) {
                rows.push(`Response source: ${scores.used_gemini ? "Gemini" : "Local fallback"}`);
            }

            scoreDiv.innerHTML = rows.map((row) => `<span>${row}</span>`).join("");
            msgDiv.appendChild(scoreDiv);
        }

        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showTypingIndicator(label = "Thinking") {
        const typingMsg = document.createElement("div");
        typingMsg.classList.add("message", "bot-message");
        typingMsg.id = "active-typing-indicator";

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble", "typing-indicator");
        bubbleDiv.innerHTML = `<span>${label}</span><span class="dot"></span><span class="dot"></span><span class="dot"></span>`;

        typingMsg.appendChild(bubbleDiv);
        chatBox.appendChild(typingMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeTypingIndicator() {
        const activeIndicator = document.getElementById("active-typing-indicator");
        if (activeIndicator) {
            activeIndicator.remove();
        }
    }

    async function postJson(url, body) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || "Request failed.");
        }
        return data;
    }

    async function loadHealth() {
        try {
            const response = await fetch("/health");
            const data = await response.json();

            apiStatus.textContent = data.model_ready && data.tokenizer_ready ? "Ready" : "Needs setup";
            apiStatus.className = data.model_ready && data.tokenizer_ready ? "status-ok" : "status-warn";
            geminiStatus.textContent = data.gemini_configured ? "Configured" : "Fallback mode";
            geminiStatus.className = data.gemini_configured ? "status-ok" : "status-warn";
        } catch (error) {
            apiStatus.textContent = "Offline";
            apiStatus.className = "status-error";
            geminiStatus.textContent = "Unknown";
            geminiStatus.className = "status-error";
        }
    }

    async function analyzeOnly() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, "user");
        setBusy(true);
        showTypingIndicator("Scoring");

        try {
            const data = await postJson("/analyze", { message: text });
            removeTypingIndicator();
            appendMessage(`Analysis result: ${data.label}. Sarcasm confidence is ${formatPercent(data.score)}.`, "bot", {
                user_sarcasm_score: data.score,
                user_is_sarcastic: data.is_sarcastic
            });
            userInput.value = "";
        } catch (error) {
            removeTypingIndicator();
            appendMessage(`Analyze error: ${error.message}`, "bot");
        } finally {
            setBusy(false);
            userInput.focus();
        }
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, "user");
        userInput.value = "";
        setBusy(true);
        showTypingIndicator();

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

    sendBtn.addEventListener("click", sendMessage);
    analyzeBtn.addEventListener("click", analyzeOnly);

    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    sampleButtons.forEach((button) => {
        button.addEventListener("click", () => {
            userInput.value = button.dataset.sample;
            userInput.focus();
        });
    });

    appendMessage("Paste a tweet, article paragraph, headline, or message. I will score the sarcasm and reply.", "bot");
    loadHealth();
    userInput.focus();
});
