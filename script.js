async function generateDigit() {

    const digit = document.getElementById("digitInput").value;
    const status = document.getElementById("status");
    const image = document.getElementById("resultImage");

    if (digit === "" || digit < 0 || digit > 9){

        status.innerText="请输入0-9数字";
        return;
    }

    status.innerText="AI 正在生成中...";

    // 模拟后端等待
    setTimeout(()=>{

        fetch("http://127.0.0.1:5000/generate")

        image.style.display="block";

        status.innerText="生成成功";

    },1500);

}