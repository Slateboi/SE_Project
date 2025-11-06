const video = document.getElementById('camera');
const startBtn = document.getElementById('startBtn');
const output = document.getElementById('output');

startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    output.textContent = "Camera live — translate your signs in real-time";
  } catch (err) {
    console.error("Camera error:", err);
    output.textContent = "Error: Could not access camera.";
  }
});
