(function () {
  "use strict";

  const API_URL = "/api/v1/predict";

  const root = document.documentElement;
  const form = document.getElementById("predict-form");
  const textarea = document.getElementById("message");
  const charCount = document.getElementById("char-count");
  const shortcutHint = document.getElementById("shortcut-hint");
  const submitBtn = document.getElementById("submit-btn");
  const submitLabel = submitBtn.querySelector(".btn-label");
  const errorBox = document.getElementById("error");
  const result = document.getElementById("result");
  const resultLabel = document.getElementById("result-label");
  const resultScore = document.getElementById("result-score");
  const progress = document.getElementById("progress");
  const progressBar = document.getElementById("progress-bar");
  const resultDescription = document.getElementById("result-description");
  const resetBtn = document.getElementById("reset-btn");
  const themeToggle = document.getElementById("theme-toggle");
  const feedbackOpenBtn = document.getElementById("feedback-open-btn");
  const feedbackForm = document.getElementById("feedback-form");
  const feedbackSaveBtn = document.getElementById("feedback-save-btn");
  const feedbackCancelBtn = document.getElementById("feedback-cancel-btn");
  const feedbackError = document.getElementById("feedback-error");
  const feedbackSuccess = document.getElementById("feedback-success");
  const FEEDBACK_URL = "/feedback";

  const maxLength = Number(textarea.getAttribute("maxlength"));
  const percentFormat = new Intl.NumberFormat("tr-TR", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  const numberFormat = new Intl.NumberFormat("tr-TR");

  let loading = false;

  // ---- Theme ----

  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  function activeTheme() {
    return root.dataset.theme || (systemDark.matches ? "dark" : "light");
  }

  function renderTheme() {
    const theme = activeTheme();
    root.dataset.activeTheme = theme;
    themeToggle.setAttribute(
      "aria-label",
      theme === "dark" ? "Açık temaya geç" : "Koyu temaya geç"
    );
  }

  themeToggle.addEventListener("click", function () {
    const next = activeTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      // Storage unavailable; the theme still applies for this page view.
    }
    renderTheme();
  });

  systemDark.addEventListener("change", renderTheme);
  renderTheme();

  // ---- Input ----

  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) {
    shortcutHint.innerHTML = "Göndermek için <kbd>⌘</kbd> + <kbd>Enter</kbd>";
  }

  function updateInputState() {
    const length = textarea.value.length;
    charCount.textContent = numberFormat.format(length) + " / " + numberFormat.format(maxLength);
    charCount.classList.toggle("near-limit", length >= maxLength * 0.9);
    submitBtn.disabled = loading || textarea.value.trim() === "";
  }

  textarea.addEventListener("input", function () {
    hideError();
    updateInputState();
  });

  textarea.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (!submitBtn.disabled) form.requestSubmit();
    }
  });

  // ---- Errors ----

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function errorMessageFor(status) {
    if (status === 400) return "Mesaj geçersiz. Lütfen mesajı kontrol edip tekrar deneyin.";
    if (status === 503) return "Model şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.";
    return "Analiz sırasında bir hata oluştu. Lütfen tekrar deneyin.";
  }

  // ---- Loading ----

  function setLoading(value) {
    loading = value;
    submitBtn.classList.toggle("loading", value);
    submitBtn.setAttribute("aria-busy", String(value));
    submitLabel.textContent = value ? "Analiz ediliyor..." : "Analiz Et";
    textarea.readOnly = value;
    updateInputState();
  }

  // ---- Result ----

  function showResult(label, spamScore, message) {
    const isSpam = label === "spam";
    analyzedMessage = message;
    predictedLabel = label;
    resetFeedback();
    const percent = Math.round(spamScore * 1000) / 10;

    result.classList.toggle("is-spam", isSpam);
    result.classList.toggle("is-ham", !isSpam);
    resultLabel.textContent = isSpam ? "SPAM TESPİT EDİLDİ" : "NORMAL MESAJ";
    resultScore.textContent = percentFormat.format(spamScore);
    resultDescription.textContent = isSpam
      ? "Bu mesaj model tarafından spam olarak sınıflandırıldı."
      : "Bu mesaj model tarafından normal olarak sınıflandırıldı.";
    progress.setAttribute("aria-valuenow", String(percent));
    progress.setAttribute("aria-valuetext", percentFormat.format(spamScore));

    progressBar.style.width = "0";
    result.classList.remove("visible");
    result.hidden = false;

    // Next frame so the reveal and progress transitions run.
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        result.classList.add("visible");
        progressBar.style.width = percent + "%";
        result.focus({ preventScroll: true });
        result.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    });
  }

  function hideResult() {
    result.hidden = true;
    result.classList.remove("visible", "is-spam", "is-ham");
    progressBar.style.width = "0";
    analyzedMessage = null;
    resetFeedback();
  }

  // ---- Feedback ----

  // The exact message that produced the current result (not the live textarea).
  let analyzedMessage = null;
  let predictedLabel = null;
  let feedbackSaving = false;
  let feedbackRequestId = 0;

  function selectedFeedbackLabel() {
    const checked = feedbackForm.querySelector('input[name="feedback-label"]:checked');
    return checked ? checked.value : null;
  }

  function setFeedbackSaving(value) {
    feedbackSaving = value;
    feedbackSaveBtn.classList.toggle("loading", value);
    feedbackSaveBtn.setAttribute("aria-busy", String(value));
    feedbackSaveBtn.disabled = value || !selectedFeedbackLabel();
    feedbackCancelBtn.disabled = value;
    feedbackForm.querySelectorAll('input[name="feedback-label"]').forEach(function (input) {
      input.disabled = value;
    });
  }

  function closeFeedbackForm() {
    feedbackForm.hidden = true;
    feedbackError.hidden = true;
    feedbackError.textContent = "";
    feedbackOpenBtn.hidden = false;
    feedbackOpenBtn.setAttribute("aria-expanded", "false");
  }

  function resetFeedback() {
    feedbackRequestId++; // ignore any response still in flight
    setFeedbackSaving(false);
    closeFeedbackForm();
    feedbackForm.reset();
    feedbackSaveBtn.disabled = true;
    feedbackSuccess.hidden = true;
    feedbackSuccess.textContent = "";
  }

  feedbackOpenBtn.addEventListener("click", function () {
    feedbackSuccess.hidden = true;
    feedbackForm.reset();
    // Pre-select the opposite of the model's answer, since this is a correction.
    const suggested = predictedLabel === "spam" ? "ham" : "spam";
    const input = feedbackForm.querySelector('input[value="' + suggested + '"]');
    input.checked = true;
    feedbackSaveBtn.disabled = false;
    feedbackForm.hidden = false;
    feedbackOpenBtn.hidden = true;
    feedbackOpenBtn.setAttribute("aria-expanded", "true");
    input.focus();
  });

  feedbackForm.addEventListener("change", function () {
    feedbackSaveBtn.disabled = feedbackSaving || !selectedFeedbackLabel();
  });

  feedbackCancelBtn.addEventListener("click", function () {
    closeFeedbackForm();
    feedbackOpenBtn.focus();
  });

  feedbackForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const label = selectedFeedbackLabel();
    if (!label || !analyzedMessage || feedbackSaving) return;

    const requestId = ++feedbackRequestId;
    feedbackError.hidden = true;
    setFeedbackSaving(true);

    let errorMessage = null;
    let saved = null;
    try {
      const response = await fetch(FEEDBACK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: analyzedMessage, label: label }),
      });
      const data = response.ok ? await response.json() : null;
      if (data && data.status === "saved" && data.label === label) {
        saved = data;
      } else {
        errorMessage = response.status === 400
          ? "Düzeltme geçersiz. Lütfen bir seçim yapıp tekrar deneyin."
          : "Düzeltme kaydedilemedi. Lütfen tekrar deneyin.";
      }
    } catch (e) {
      errorMessage = "Sunucuya bağlanılamadı. Düzeltme kaydedilmedi.";
    }

    if (requestId !== feedbackRequestId) return; // result changed meanwhile
    setFeedbackSaving(false);

    if (!saved) {
      feedbackError.textContent = errorMessage;
      feedbackError.hidden = false;
      return;
    }

    // Shown only after the server confirmed the save.
    closeFeedbackForm();
    const labelText = label === "spam" ? "Spam" : "Normal (Ham)";
    feedbackSuccess.textContent = saved.updated
      ? "Düzeltmeniz güncellendi: " + labelText + ". Teşekkürler!"
      : "Düzeltmeniz kaydedildi: " + labelText + ". Teşekkürler!";
    feedbackSuccess.hidden = false;
    feedbackOpenBtn.focus();
  });

  resetBtn.addEventListener("click", function () {
    hideResult();
    hideError();
    textarea.value = "";
    updateInputState();
    textarea.focus();
  });

  // ---- Submit ----

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    const message = textarea.value.trim();
    if (!message || loading) return;

    hideError();
    hideResult();
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message }),
      });

      if (!response.ok) {
        showError(errorMessageFor(response.status));
        return;
      }

      const data = await response.json();
      if ((data.label !== "spam" && data.label !== "ham") || typeof data.spam_score !== "number") {
        showError("Sunucudan beklenmeyen bir yanıt alındı.");
        return;
      }

      showResult(data.label, data.spam_score, message);
    } catch (e) {
      showError("Sunucuya bağlanılamadı. Bağlantınızı kontrol edip tekrar deneyin.");
    } finally {
      setLoading(false);
    }
  });

  updateInputState();
})();
