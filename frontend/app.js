const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Fill background black
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    lastX = e.offsetX;
    lastY = e.offsetY;
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;

    ctx.strokeStyle = "white";
    ctx.lineWidth = 10  ;
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    lastX = e.offsetX;
    lastY = e.offsetY;
});

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Predict function
async function predict() {

    // Create temp canvas (28x28)
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    const tctx = tempCanvas.getContext("2d");

    // Resize image
    // Centering logic (simple version)
    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, 28, 28);

    // Draw scaled image centered
    tctx.drawImage(canvas, 0, 0, 28, 28);

    // Get pixel data
    const imageData = tctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    // Convert to grayscale
    let input = [];

    for (let i = 0; i < data.length; i += 4) {
        let value = data[i] / 255.0;

        input.push(value);
    }

    // Create tensor
    const tensor = new ort.Tensor(
        "float32",
        new Float32Array(input),
        [1, 1, 28, 28]
    );

    // Load model
    const session = await ort.InferenceSession.create("model/lenet.onnx");

    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    // Run inference
    const results = await session.run(feeds);

    // Extract outputs FIRST
    const output = results["output"].data;
    const conv1 = results["conv1"].data;
    const conv2 = results["conv2"].data;

    // THEN visualize
    visualizeConvLayer(conv1, 6, 24, "conv1");
    visualizeConvLayer(conv2, 16, 8, "conv2");

    // Get prediction
    const max = Math.max(...output);
    const prediction = output.indexOf(max);

    document.getElementById("result").innerText =
        "Prediction: " + prediction;
}
// Visualize Conv1
function visualizeConvLayer(data, channels, size, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    for (let c = 0; c < channels; c++) {
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        canvas.style.margin = "5px";

        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(size, size);

        for (let i = 0; i < size * size; i++) {
            let value = data[c * size * size + i];

            // Normalize
            value = (value - Math.min(...data)) / (Math.max(...data) - Math.min(...data));

            let pixel = value * 255;

            imageData.data[i * 4] = pixel;
            imageData.data[i * 4 + 1] = pixel;
            imageData.data[i * 4 + 2] = pixel;
            imageData.data[i * 4 + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);
        container.appendChild(canvas);
    }
}