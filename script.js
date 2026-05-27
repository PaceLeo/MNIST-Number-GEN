let currentDigit = 0;
let currentStyle = "clean";
let historyCount = 0;

// ─── 后端地址，方便统一修改 ───────────────────────────────────────────
const BACKEND = "http://101.76.221.147:8000";

function selectDigit(element, digit) {
    currentDigit = digit;
    document.querySelectorAll(".digit-btn").forEach(btn => btn.classList.remove("active"));
    element.classList.add("active");
    document.getElementById("digitInput").value = digit;
}

function selectStyle(element, style) {
    currentStyle = style;
    document.querySelectorAll(".style-option").forEach(item => item.classList.remove("active"));
    element.classList.add("active");
    document.getElementById("styleSelect").value = style;
}

// ─── 读取当前所有参数，拼成 API URL ───────────────────────────────────
function buildApiUrl() {
    const digit     = document.getElementById("digitInput").value;
    const thickness = document.getElementById("thicknessInput").value;
    const slant     = document.getElementById("slantInput").value;
    const num       = 24;
    const nrow      = 4;
    const clean     = 1;

    return `${BACKEND}/generate?digit=${digit}&num=${num}&nrow=${nrow}&thickness=${thickness}&slant=${slant}&clean=${clean}`;
}

// ─── 主生成函数 ────────────────────────────────────────────────────────
async function generateDigit() {
    document.body.classList.add("show-result-page");

    const image       = document.getElementById("resultImage");
    const loading     = document.getElementById("loading");
    const placeholder = document.getElementById("placeholder");
    const imageBox    = document.getElementById("imageBox");

    image.style.display = "none";
    placeholder.style.display = "none";
    loading.classList.remove("hidden");
    imageBox.classList.add("generating");

    const url = buildApiUrl();
    console.log("[MNIST GEN] 请求地址:", url);

    try {
        const response = await fetch(url, {
            method: "GET",
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const contentType = response.headers.get("content-type") || "";
        if (!contentType.startsWith("image/")) {
            const text = await response.text();
            throw new Error(`后端返回非图片内容 (${contentType}):\n${text.slice(0, 200)}`);
        }

        const blob     = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        loading.classList.add("hidden");
        imageBox.classList.remove("generating");

        showResult(imageUrl);

    } catch (error) {
        console.error("[MNIST GEN] 请求失败:", error);

        loading.classList.add("hidden");
        imageBox.classList.remove("generating");

        placeholder.style.display = "block";
        placeholder.innerHTML = `
            <div class="placeholder-icon">⚠</div>
            <p>后端连接失败</p>
            <small>${error.message}</small>
        `;
    }
}

// ─── 展示图片结果 ──────────────────────────────────────────────────────
function showResult(imageUrl) {
    const image = document.getElementById("resultImage");

    image.src           = imageUrl;
    image.style.display = "block";

    image.classList.remove("show-result");
    void image.offsetWidth;
    image.classList.add("show-result");

    addHistory(currentDigit, currentStyle);
}

// ─── 历史记录 ──────────────────────────────────────────────────────────
function addHistory(digit, style) {
    const historyList = document.getElementById("historyList");
    if (!historyList) return;

    if (historyCount === 0) historyList.innerHTML = "";
    historyCount++;

    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
        <div>
            <strong>Digit ${digit}</strong>
            <span>${style}</span>
        </div>
        <small>${new Date().toLocaleTimeString()}</small>
    `;
    historyList.prepend(item);

    const items = historyList.querySelectorAll(".history-item");
    if (items.length > 5) items[items.length - 1].remove();
}

// ─── 重置 ──────────────────────────────────────────────────────────────
function resetResult() {
    const image       = document.getElementById("resultImage");
    const placeholder = document.getElementById("placeholder");
    const loading     = document.getElementById("loading");
    const imageBox    = document.getElementById("imageBox");

    image.style.display = "none";
    image.src           = "";
    placeholder.style.display = "block";
    placeholder.innerHTML = `
        <div class="placeholder-icon">✦</div>
        <p>等待生成</p>
        <small>Choose a digit and click Generate</small>
    `;
    loading.classList.add("hidden");
    imageBox.classList.remove("generating");
}

// ─── 滑块数值实时显示 ──────────────────────────────────────────────────
function updateRangeValue(type) {
    if (type === "thickness") {
        document.getElementById("thicknessValue").innerText =
            document.getElementById("thicknessInput").value;
    }
    if (type === "slant") {
        document.getElementById("slantValue").innerText =
            document.getElementById("slantInput").value;
    }
}

// ─── 移动端返回 ────────────────────────────────────────────────────────
function backToInput() {
    document.body.classList.remove("show-result-page");
}
