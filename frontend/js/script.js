async function loadDetections() {
    const response = await fetch("http://127.0.0.1:5000/detections");
    const data = await response.json();

    const tableBody = document.getElementById("history-body");
    tableBody.innerHTML = "";

    data.reverse().forEach(item => {
        const row = `
            <tr>
                <td>${item.object}</td>
                <td>${item.confidence}</td>
                <td>${item.time}</td>
                <td>${item.object === "cell phone" ? "ALERT" : "SAFE"}</td>
            </tr>
        `;
        tableBody.innerHTML += row;
    });
}

setInterval(loadDetections, 2000);
loadDetections();