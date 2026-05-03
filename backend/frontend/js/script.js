let lastAlertTime = 0;

const dangerousObjects = ["knife", "scissors", "gun", "cell phone"];

async function loadDetections() {

    const response = await fetch("http://127.0.0.1:5000/detections");
    const data = await response.json();

    const tableBody = document.getElementById("history-body");

    // Prevent error if page doesn't have history table
    if (!tableBody) return;

    tableBody.innerHTML = "";

    data.reverse().forEach(item => {

        const isDangerous = dangerousObjects.includes(item.object);

        // Play alert sound with cooldown
        if (isDangerous) {
            const now = Date.now();

            if (now - lastAlertTime > 5000) {
                const sound = document.getElementById("alertSound");

                if (sound) {
                    sound.currentTime = 0;
                    sound.play();
                }

                lastAlertTime = now;
            }
        }

        const row = `
        <tr>
            <td>${item.object}</td>
            <td>${item.confidence}</td>
            <td>${item.time}</td>
            <td class="${isDangerous ? 'status-alert' : 'status-safe'}">
                ${isDangerous ? 'ALERT' : 'SAFE'}
            </td>
        </tr>
        `;

        tableBody.innerHTML += row;

    });
}

// run once
loadDetections();

// update every 2 seconds
setInterval(loadDetections, 2000);