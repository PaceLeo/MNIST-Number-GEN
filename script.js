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
    const digit = document.getElementById("digitInput").value;
    const style = document.getElementById("styleSelect").value;

    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");
    const loading = document.getElementById("loading");
    const placeholder = document.getElementById("placeholder");
    const imageBox = document.getElementById("imageBox");

    status.innerText = "AI 正在分析输入条件...";
    image.style.display = "none";
    placeholder.style.display = "none";
    loading.classList.remove("hidden");
    imageBox.classList.add("generating");

    try {
        const response = await fetch("http://127.0.0.1:5000/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                digit: Number(digit),
                style: style
            })
        });

        const data = await response.json();

        loading.classList.add("hidden");
        imageBox.classList.remove("generating");

        if (data.success) {
            showResult(data.image_url, digit, style, false);
        } else {
            status.innerText = "生成失败：" + data.message;
            placeholder.style.display = "block";
        }

    } catch (error) {
        setTimeout(() => {
            loading.classList.add("hidden");
            imageBox.classList.remove("generating");

            const demoImage =
                "https://dummyimage.com/280x280/ffffff/000000&text=" + digit;

            showResult(demoImage, digit, style, true);
        }, 1000);
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

