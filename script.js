function selectStyle(element,style){

document
.querySelectorAll(".style-option")
.forEach(item=>{

item.classList.remove(
"active"
)

})

element.classList.add(
"active"
)

document
.getElementById(
"styleSelect"
).value=style

}

async function generateDigit() {
    const digit = document.getElementById("digitInput").value;
    const style = document.getElementById("styleSelect").value;
    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");
    const loading = document.getElementById("loading");
    const placeholder = document.getElementById("placeholder");

    if (digit === "" || digit < 0 || digit > 9) {
        status.innerText = "请输入 0 到 9 之间的数字";
        image.style.display = "none";
        placeholder.style.display = "block";
        return;
    }

    status.innerText = "请求已发送到后端接口";
    image.style.display = "none";
    placeholder.style.display = "none";
    loading.classList.remove("hidden");

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

        if (data.success) {
            image.src = data.image_url;
            image.style.display = "block";
            status.innerText = "生成成功：数字 " + digit + "，风格：" + style;
        } else {
            status.innerText = "生成失败：" + data.message;
            placeholder.style.display = "block";
        }

    } catch (error) {
        loading.classList.add("hidden");

        // 后端没接好之前，先用假图演示前端效果
        image.src = "https://dummyimage.com/260x260/ffffff/000000&text=" + digit;
        image.style.display = "block";
        status.innerText = "当前为前端演示模式：后端接口尚未连接";
    }
}