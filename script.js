let currentDigit = 0;
let currentStyle = "clean";
let historyCount = 0;

function selectDigit(element, digit) {
    currentDigit = digit;

    document.querySelectorAll(".digit-btn").forEach(btn => {
        btn.classList.remove("active");
    });

    element.classList.add("active");

    document.getElementById("digitInput").value = digit;
    document.getElementById("metaDigit").innerText = digit;
}

function selectStyle(element, style) {
    currentStyle = style;

    document.querySelectorAll(".style-option").forEach(item => {
        item.classList.remove("active");
    });

    element.classList.add("active");

    document.getElementById("styleSelect").value = style;
    document.getElementById("metaStyle").innerText = style;
}

async function generateDigit() {
    document.body.classList.add("show-result-page");
    const digit = document.getElementById("digitInput").value;
    const thickness = document.getElementById("thicknessInput").value;
    const slant = document.getElementById("slantInput").value;
    const num = document.getElementById("numInput").value;
    const nrow = document.getElementById("nrowInput").value;
    const clean = document.getElementById("cleanInput").value;

    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");
    const loading = document.getElementById("loading");
    const placeholder = document.getElementById("placeholder");
    const imageBox = document.getElementById("imageBox");

    status.innerText = "AI 正在根据粗细与倾斜参数生成图像...";
    image.style.display = "none";
    placeholder.style.display = "none";
    loading.classList.remove("hidden");
    imageBox.classList.add("generating");

    const backendUrl =
    `http://101.76.221.147:8000/generate?digit=${digit}&num=${num}&nrow=${nrow}&thickness=${thickness}&slant=${slant}&clean=${clean}`;
    try {
        const response = await fetch(backendUrl);

        if (!response.ok) {
            throw new Error("后端返回错误");
        }

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        loading.classList.add("hidden");
        imageBox.classList.remove("generating");

        showResult(imageUrl, digit, `thickness=${thickness}, slant=${slant}`, false);

    } catch (error) {
        setTimeout(() => {
            loading.classList.add("hidden");
            imageBox.classList.remove("generating");

            const demoImage =
                "https://dummyimage.com/280x280/ffffff/000000&text=" + digit;

            showResult(demoImage, digit, `thickness=${thickness}, slant=${slant}`, true);
        }, 800);
    }
}

function showResult(imageUrl, digit, style, demoMode) {
    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");

    image.src = imageUrl;
    image.style.display = "block";

    image.classList.remove("show-result");
    void image.offsetWidth;
    image.classList.add("show-result");

    if (demoMode) {
        status.innerText = `前端演示模式：数字 ${digit}，风格 ${style}`;
    } else {
        status.innerText = `生成成功：数字 ${digit}，风格 ${style}`;
    }

    addHistory(digit, style);
}

function addHistory(digit, style) {
    const historyList = document.getElementById("historyList");

    if (historyCount === 0) {
        historyList.innerHTML = "";
    }

    historyCount++;

    const item = document.createElement("div");
    item.className = "history-item";

    const time = new Date().toLocaleTimeString();

    item.innerHTML = `
        <div>
            <strong>Digit ${digit}</strong>
            <span>${style}</span>
        </div>
        <small>${time}</small>
    `;

    historyList.prepend(item);

    const items = historyList.querySelectorAll(".history-item");

    if (items.length > 5) {
        items[items.length - 1].remove();
    }
}

function resetResult() {
    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");
    const placeholder = document.getElementById("placeholder");
    const loading = document.getElementById("loading");
    const imageBox = document.getElementById("imageBox");

    image.style.display = "none";
    image.src = "";
    placeholder.style.display = "block";
    loading.classList.add("hidden");
    imageBox.classList.remove("generating");

    status.innerText = "系统已重置，请重新选择数字与风格。";
}

function updateRangeValue(type) {
    if (type === "thickness") {
        const value = document.getElementById("thicknessInput").value;
        document.getElementById("thicknessValue").innerText = value;
    }

    if (type === "slant") {
        const value = document.getElementById("slantInput").value;
        document.getElementById("slantValue").innerText = value;
    }
}

function backToInput() {
    document.body.classList.remove("show-result-page");
}
