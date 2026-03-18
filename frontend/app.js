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
    ctx.lineWidth = 10;
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

// ✅ FIX 3: Safe min/max for large typed arrays (avoids call stack overflow)
function safeMin(arr) {
    let m = Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
    return m;
}

function safeMax(arr) {
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
}

// Predict function
async function predict() {

    // Create temp canvas (28x28)
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    const tctx = tempCanvas.getContext("2d");

    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, 28, 28);
    tctx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    let input = [];
    for (let i = 0; i < data.length; i += 4) {
        input.push(data[i] / 255.0);
    }

    const tensor = new ort.Tensor(
        "float32",
        new Float32Array(input),
        [1, 1, 28, 28]
    );

    const session = await ort.InferenceSession.create("model/lenet.onnx");

    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const results = await session.run(feeds);

    const output = results["output"].data;
    const conv1 = results["conv1"].data;
    const conv2 = results["conv2"].data;

    // Visualize 2D feature maps
    visualizeConvLayer(conv1, 6, 24, "conv1");
    visualizeConvLayer(conv2, 16, 8, "conv2");

    // ✅ FIX 4: Copy children array before removal to avoid mutation-during-iteration bug
    const toRemove = scene.children.filter(obj => obj.type !== "PointLight" && obj.type !== "AmbientLight");
    toRemove.forEach(obj => scene.remove(obj));

    // 3D visualization
    createLayer(conv1, 6, 24, -60);
    createLayer(conv2, 16, 8, 20);

    // ✅ FIX 5: Float32Array doesn't have .indexOf — convert first
    const outputArr = Array.from(output);
    const max = Math.max(...outputArr);
    const prediction = outputArr.indexOf(max);

    document.getElementById("result").innerText = "Prediction: " + prediction;
}

// Visualize Conv feature maps (2D)
function visualizeConvLayer(data, channels, size, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    // ✅ FIX 3: Use safe min/max
    const dmin = safeMin(data);
    const dmax = safeMax(data);

    for (let c = 0; c < channels; c++) {
        const cvs = document.createElement("canvas");
        cvs.width = size * 3;   // upscale for visibility
        cvs.height = size * 3;
        cvs.style.margin = "5px";
        cvs.style.imageRendering = "pixelated";

        const cctx = cvs.getContext("2d");
        const imgData = cctx.createImageData(size, size);

        for (let i = 0; i < size * size; i++) {
            let value = (data[c * size * size + i] - dmin) / (dmax - dmin + 1e-6);
            let pixel = Math.round(value * 255);
            imgData.data[i * 4]     = pixel;
            imgData.data[i * 4 + 1] = pixel;
            imgData.data[i * 4 + 2] = pixel;
            imgData.data[i * 4 + 3] = 255;
        }

        // Draw to temp canvas then scale up
        const tmp = document.createElement("canvas");
        tmp.width = size;
        tmp.height = size;
        tmp.getContext("2d").putImageData(imgData, 0, 0);

        cctx.drawImage(tmp, 0, 0, size * 3, size * 3);
        container.appendChild(cvs);
    }
}

let scene, camera, renderer;
let controls;

function init3D() {
    const container = document.getElementById("three-container");

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        2000
    );
    camera.position.set(0, 80, 160);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // ✅ FIX 1: OrbitControls — use THREE.OrbitControls (loaded via CDN script tag)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // ✅ FIX 2: Add AmbientLight so MeshStandardMaterial is visible from all angles
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1.5);
    pointLight.position.set(100, 100, 100);
    scene.add(pointLight);

    const pointLight2 = new THREE.PointLight(0x8888ff, 0.8);
    pointLight2.position.set(-100, -50, -100);
    scene.add(pointLight2);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

init3D();

function createLayer(data, channels, size, xOffset) {
    const group = new THREE.Group();

    // ✅ FIX 3: Safe min/max
    const min = safeMin(data);
    const max = safeMax(data);

    // ✅ FIX 6: Better channel spacing so they don't overlap
    const channelSpacing = size + 4;

    for (let c = 0; c < channels; c++) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                let idx = c * size * size + i * size + j;
                let value = (data[idx] - min) / (max - min + 1e-6);
                value = Math.pow(Math.max(0, value), 0.5); // gamma correction

                // Skip near-zero activations for performance
                if (value < 0.05) continue;

                const geometry = new THREE.BoxGeometry(1.8, 1.8, 1.8);
                const color = new THREE.Color().setHSL(0.6 - value * 0.5, 1.0, 0.2 + value * 0.5);
                const material = new THREE.MeshStandardMaterial({
                    color,
                    roughness: 0.4,
                    metalness: 0.1,
                    transparent: true,
                    opacity: 0.5 + value * 0.5
                });

                const cube = new THREE.Mesh(geometry, material);
                cube.position.set(
                    xOffset,
                    i - size / 2,
                    // ✅ FIX 6: proper spacing between channels
                    j - size / 2 + c * channelSpacing - (channels / 2) * channelSpacing
                );

                group.add(cube);
            }
        }
    }

    scene.add(group);
}