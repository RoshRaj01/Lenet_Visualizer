// ═══════════════════════════════════════════════════════════
//  LeNet Visualizer — app.js
//  Faithful TensorSpace-style 3D neural network visualization
// ═══════════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────────
//  DRAWING CANVAS
// ───────────────────────────────────────────────────────────
const canvas = document.getElementById("canvas");
const ctx    = canvas.getContext("2d");
ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 284, 284);

let drawing = false, lx = 0, ly = 0;
const getPos = (e, el) => {
  const r = el.getBoundingClientRect();
  return {
    x: (e.clientX !== undefined ? e.clientX : e.touches[0].clientX) - r.left,
    y: (e.clientY !== undefined ? e.clientY : e.touches[0].clientY) - r.top
  };
};
const startDraw = e => { drawing = true; const p = getPos(e, canvas); lx = p.x; ly = p.y; };
const stopDraw  = () => drawing = false;
const moveDraw  = e => {
  if (!drawing) return;
  const p = getPos(e, canvas);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(p.x, p.y); ctx.stroke();
  lx = p.x; ly = p.y;
};
canvas.addEventListener("mousedown",  startDraw);
canvas.addEventListener("mouseup",    stopDraw);
canvas.addEventListener("mouseleave", stopDraw);
canvas.addEventListener("mousemove",  moveDraw);
canvas.addEventListener("touchstart", e => { e.preventDefault(); startDraw(e); }, { passive: false });
canvas.addEventListener("touchend",   e => { e.preventDefault(); stopDraw();   }, { passive: false });
canvas.addEventListener("touchmove",  e => { e.preventDefault(); moveDraw(e);  }, { passive: false });

function clearCanvas() {
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 284, 284);
  resetLayers();
  document.getElementById("res-text").innerHTML = "LeNet thinks you write a \u2014";
  document.querySelectorAll(".dc").forEach(c => c.classList.remove("on"));
  document.querySelectorAll(".cfill").forEach(b => b.style.width = "0%");
  document.querySelectorAll(".cval").forEach(v => v.textContent = "0%");
  setStatus("idle", "Ready \u2014 draw something");
  document.getElementById("hint").style.opacity = "1";
  autoRotate = true;
}

// ───────────────────────────────────────────────────────────
//  UTILITIES
// ───────────────────────────────────────────────────────────
function safeMin(a) { let m = Infinity;  for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
function safeMax(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }

function softmax(arr) {
  const mx = Math.max(...arr);
  const ex = arr.map(x => Math.exp(x - mx));
  const s  = ex.reduce((a, b) => a + b, 0);
  return ex.map(x => x / s);
}

function setStatus(state, text) {
  document.getElementById("sdot").className = state;
  document.getElementById("stxt").textContent = text;
}

function setLoad(p, msg) {
  document.getElementById("lbar").style.width = p + "%";
  document.getElementById("lmsg").textContent = msg;
}

// ───────────────────────────────────────────────────────────
//  CONFIDENCE BARS
// ───────────────────────────────────────────────────────────
(function buildCBars() {
  const c = document.getElementById("cbars");
  for (let i = 0; i < 10; i++) {
    const r = document.createElement("div");
    r.className = "crow"; r.id = "cr" + i;
    r.innerHTML = '<div class="clabel">' + i + '</div>' +
      '<div class="ctrack"><div class="cfill" id="cf' + i + '"></div></div>' +
      '<div class="cval" id="cv' + i + '">0%</div>';
    c.appendChild(r);
  }
})();

function updateCBars(probs, top) {
  for (let i = 0; i < 10; i++) {
    const p = (probs[i] * 100).toFixed(1);
    document.getElementById("cf" + i).style.width = p + "%";
    document.getElementById("cv" + i).textContent = p + "%";
    document.getElementById("cr" + i).classList.toggle("top", i === top);
  }
}

// ───────────────────────────────────────────────────────────
//  THREE.JS SCENE
// ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let autoRotate = true;
let rotY = 0;
let frameCount = 0, lastFpsT = performance.now();
let liveMeshes = [];
let skeletonGroups = [];
let camTo = null;

// Layer definitions matching TensorSpace arrangement
// pw/ph = plane width/height in world units
// spread = vertical fan spacing between channels
// zSlant = depth offset per channel (creates diagonal fan look)
const LD = [
  { id: "input", label: "INPUT",  x: -230, ch:  1, sz: 28, pw: 9.5, ph: 9.5, spread: 0,   zSlant: 0,   tilt: Math.PI * 0.18 },
  { id: "conv1", label: "CONV1",  x: -140, ch:  6, sz: 24, pw: 9,   ph: 9,   spread: 6.5, zSlant: 3.5, tilt: Math.PI * 0.18 },
  { id: "pool1", label: "POOL1",  x:  -70, ch:  6, sz: 12, pw: 5.5, ph: 5.5, spread: 6.5, zSlant: 3.5, tilt: Math.PI * 0.18 },
  { id: "conv2", label: "CONV2",  x:   10, ch: 16, sz:  8, pw: 6,   ph: 6,   spread: 4.5, zSlant: 2.5, tilt: Math.PI * 0.18 },
  { id: "pool2", label: "POOL2",  x:   75, ch: 16, sz:  4, pw: 3.5, ph: 3.5, spread: 4.5, zSlant: 2.5, tilt: Math.PI * 0.18 },
  { id: "fc1",   label: "FC1",    x:  135, ch:  1, sz:  1, pw: 3,   ph: 20,  spread: 0,   zSlant: 0,   tilt: 0              },
  { id: "out",   label: "OUTPUT", x:  185, ch:  1, sz:  1, pw: 5,   ph: 28,  spread: 0,   zSlant: 0,   tilt: 0              },
];

const TILT = Math.PI * 0.18; // ~32deg Y tilt (used as fallback)

function init3D() {
  const vp = document.getElementById("viewport");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x040d18);
  scene.fog = new THREE.FogExp2(0x040d18, 0.0018);

  camera = new THREE.PerspectiveCamera(52, vp.clientWidth / vp.clientHeight, 0.5, 3000);
  camera.position.set(-10, 65, 270);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(vp.clientWidth, vp.clientHeight);
  vp.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 50;
  controls.maxDistance = 700;
  controls.target.set(-20, 0, 0);
  controls.addEventListener("start", () => { autoRotate = false; });

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  var pl1 = new THREE.PointLight(0x00c8ff, 2.0, 900); pl1.position.set(-60, 180, 160);  scene.add(pl1);
  var pl2 = new THREE.PointLight(0x00ff9d, 1.2, 900); pl2.position.set(120, -100, -140); scene.add(pl2);
  var pl3 = new THREE.PointLight(0xffffff, 0.5, 600); pl3.position.set(0, -200, 60);    scene.add(pl3);

  // Grid floor
  var grid = new THREE.GridHelper(700, 50, 0x081624, 0x081624);
  grid.position.y = -45;
  grid.material.transparent = true;
  grid.material.opacity = 0.6;
  scene.add(grid);

  buildSkeleton();
  addLayerLabels();

  window.addEventListener("resize", function() {
    camera.aspect = vp.clientWidth / vp.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(vp.clientWidth, vp.clientHeight);
  });

  animate();
}

// ── Skeleton placeholder planes ──
function buildSkeleton() {
  skeletonGroups.forEach(g => scene.remove(g));
  skeletonGroups = [];

  LD.forEach(function(def) {
    var g = new THREE.Group();
    var count = Math.min(def.ch, 8);
    for (var i = 0; i < count; i++) {
      var geo = new THREE.PlaneGeometry(def.pw, def.ph);
      var mat = new THREE.MeshStandardMaterial({
        color: 0x0a1e30, transparent: true, opacity: 0.65,
        side: THREE.DoubleSide, roughness: 0.8,
      });
      var m = new THREE.Mesh(geo, mat);
      var off = (i - count / 2 + 0.5);
      m.position.set(def.x, off * def.spread, off * def.zSlant);
      m.rotation.y = def.tilt;
      g.add(m);

      // Wire border
      var eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
      var em = new THREE.LineBasicMaterial({ color: 0x0e3050, transparent: true, opacity: 0.8 });
      var el = new THREE.LineSegments(eg, em);
      el.position.set(def.x, off * def.spread, off * def.zSlant);
      el.rotation.y = def.tilt;
      g.add(el);
    }
    scene.add(g);
    skeletonGroups.push(g);
  });

  // Connection dots / lines between layers
  for (var i = 0; i < LD.length - 1; i++) {
    var pts = [new THREE.Vector3(LD[i].x, 0, 0), new THREE.Vector3(LD[i+1].x, 0, 0)];
    var geo = new THREE.BufferGeometry().setFromPoints(pts);
    var mat = new THREE.LineBasicMaterial({ color: 0x0d2a45, transparent: true, opacity: 0.5 });
    scene.add(new THREE.Line(geo, mat));
  }
}

// ── Sprite text labels under each layer ──
function addLayerLabels() {
  LD.forEach(function(def) {
    var cv = document.createElement("canvas");
    cv.width = 256; cv.height = 40;
    var c = cv.getContext("2d");
    c.font = "bold 18px 'Courier New'";
    c.fillStyle = "#2e4d6a";
    c.textAlign = "center";
    c.fillText(def.label, 128, 26);
    var tex = new THREE.CanvasTexture(cv);
    var sp  = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.85 }));
    sp.scale.set(20, 3.2, 1);
    sp.position.set(def.x, -32, 0);
    scene.add(sp);
  });
}

// ───────────────────────────────────────────────────────────
//  ANIMATION LOOP
// ───────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (autoRotate) {
    rotY += 0.003;
    camera.position.x = -10 + Math.sin(rotY) * 40;
    camera.position.z = 270 + Math.cos(rotY) * 30;
    camera.lookAt(-20, 0, 0);
  }

  if (camTo) {
    camera.position.lerp(camTo.pos, 0.04);
    controls.target.lerp(camTo.tgt, 0.04);
    if (camera.position.distanceTo(camTo.pos) < 0.5) camTo = null;
  }

  renderer.render(scene, camera);

  frameCount++;
  var now = performance.now();
  if (now - lastFpsT >= 1000) {
    document.getElementById("fps").textContent = frameCount + " FPS";
    frameCount = 0; lastFpsT = now;
  }
}

// ───────────────────────────────────────────────────────────
//  TEXTURE GENERATORS
// ───────────────────────────────────────────────────────────
// Activation map — teal → gold colormap (matches TensorSpace)
function activation2tex(data, offset, size) {
  var cv = document.createElement("canvas");
  cv.width = size; cv.height = size;
  var c = cv.getContext("2d");
  var id = c.createImageData(size, size);
  var mn = safeMin(data), mx = safeMax(data), rng = mx - mn + 1e-7;
  for (var i = 0; i < size * size; i++) {
    var v = Math.pow(Math.max(0, (data[offset + i] - mn) / rng), 0.55);
    id.data[i*4  ] = Math.round(v * 220 + (1-v) * 8);
    id.data[i*4+1] = Math.round(v * 185 + (1-v) * 55);
    id.data[i*4+2] = Math.round(v * 20  + (1-v) * 130);
    id.data[i*4+3] = Math.round(170 + v * 85);
  }
  c.putImageData(id, 0, 0);
  return new THREE.CanvasTexture(cv);
}

function input2tex(data) {
  var cv = document.createElement("canvas"); cv.width = 28; cv.height = 28;
  var c = cv.getContext("2d"); var id = c.createImageData(28, 28);
  for (var i = 0; i < 28*28; i++) {
    var v = Math.round(data[i] * 255);
    id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
  }
  c.putImageData(id, 0, 0); return new THREE.CanvasTexture(cv);
}

function bars2tex(probs) {
  var W=80, H=140;
  var cv = document.createElement("canvas"); cv.width=W; cv.height=H;
  var c = cv.getContext("2d");
  c.fillStyle="#04090f"; c.fillRect(0,0,W,H);
  var top = probs.indexOf(Math.max.apply(null, probs));
  for (var i=0; i<10; i++) {
    var bh = Math.max(2, probs[i]*115);
    var x  = 3 + i * 7.5;
    c.fillStyle = i===top ? "#00ff9d" : "#00c8ff";
    c.fillRect(x, H-20-bh, 6, bh);
    c.fillStyle="#2e4d6a"; c.font="7px monospace";
    c.textAlign="center"; c.fillText(i, x+3, H-6);
  }
  return new THREE.CanvasTexture(cv);
}

// ───────────────────────────────────────────────────────────
//  LAYER MANAGEMENT
// ───────────────────────────────────────────────────────────
function resetLayers() {
  liveMeshes.forEach(function(m) { scene.remove(m); });
  liveMeshes = [];
  skeletonGroups.forEach(function(g) { g.visible = true; });
}

function buildActLayer(def, texFn, chCount, delay0) {
  var skelIdx = LD.findIndex(function(d) { return d.id === def.id; });
  if (skelIdx >= 0 && skeletonGroups[skelIdx]) {
    skeletonGroups[skelIdx].visible = false;
  }

  var visCount = Math.min(chCount, 16);
  for (var ci = 0; ci < visCount; ci++) {
    (function(c) {
      var tex  = texFn(c);
      var geo  = new THREE.PlaneGeometry(def.pw, def.ph);
      var mat  = new THREE.MeshStandardMaterial({
        map: tex, transparent: true, opacity: 0.93,
        side: THREE.DoubleSide, roughness: 0.35, metalness: 0.12,
      });
      var mesh = new THREE.Mesh(geo, mat);
      var off  = (c - visCount / 2 + 0.5);
      mesh.position.set(def.x, off * def.spread, off * def.zSlant);
      mesh.rotation.y = def.tilt;

      // Glowing cyan edge
      var eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
      var em = new THREE.LineBasicMaterial({ color: 0x00c8ff, transparent: true, opacity: 0.45 });
      var edge = new THREE.LineSegments(eg, em);
      edge.position.copy(mesh.position);
      edge.rotation.y = def.tilt;

      mesh.scale.set(0.01, 0.01, 0.01);
      edge.scale.set(0.01, 0.01, 0.01);

      setTimeout(function() {
        scene.add(mesh); scene.add(edge);
        liveMeshes.push(mesh, edge);
        easeScale(mesh, 1, 340);
        easeScale(edge, 1, 340);
      }, delay0 + c * 55);
    })(ci);
  }
}

function easeScale(obj, to, dur) {
  var t0   = performance.now();
  var from = obj.scale.x;
  function tick(now) {
    var t = Math.min((now - t0) / dur, 1);
    var e = 1 - Math.pow(1 - t, 3);
    var s = from + (to - from) * e;
    obj.scale.set(s, s, s);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// 2×2 max-pool
function maxPool(data, ch, inSz) {
  var os  = inSz / 2;
  var out = new Float32Array(ch * os * os);
  for (var c=0; c<ch; c++)
    for (var i=0; i<os; i++)
      for (var j=0; j<os; j++) {
        var b = c*inSz*inSz + i*2*inSz + j*2;
        out[c*os*os + i*os + j] = Math.max(
          data[b]||0, data[b+1]||0, data[b+inSz]||0, data[b+inSz+1]||0);
      }
  return out;
}

// single floating plane helper
function addSinglePlane(def, tex, glowColor, delay) {
  setTimeout(function() {
    var geo = new THREE.PlaneGeometry(def.pw, def.ph);
    var mat = new THREE.MeshStandardMaterial({ map:tex, transparent:true, opacity:.93, side:THREE.DoubleSide });
    var m   = new THREE.Mesh(geo, mat);
    m.position.set(def.x, 0, 0); m.rotation.y = def.tilt;
    m.scale.set(0.01,0.01,0.01); easeScale(m, 1, 340);
    scene.add(m); liveMeshes.push(m);

    var eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
    var em = new THREE.LineBasicMaterial({ color: glowColor, transparent:true, opacity:0.8 });
    var el = new THREE.LineSegments(eg, em);
    el.position.copy(m.position); el.rotation.y = def.tilt;
    el.scale.set(0.01,0.01,0.01); easeScale(el, 1, 340);
    scene.add(el); liveMeshes.push(el);

    var skelIdx = LD.findIndex(function(d){ return d.id === def.id; });
    if (skelIdx>=0 && skeletonGroups[skelIdx]) skeletonGroups[skelIdx].visible = false;
  }, delay);
}

// ───────────────────────────────────────────────────────────
//  PREDICT
// ───────────────────────────────────────────────────────────
async function predict() {
  setStatus("busy", "Running inference\u2026");
  autoRotate = false;
  document.getElementById("hint").style.opacity = "0";

  var tmp = document.createElement("canvas");
  tmp.width = 28; tmp.height = 28;
  var tc = tmp.getContext("2d");
  tc.fillStyle = "#000"; tc.fillRect(0,0,28,28);
  tc.drawImage(canvas, 0, 0, 28, 28);
  var px  = tc.getImageData(0,0,28,28).data;
  var inp = new Float32Array(28*28);
  for (var i=0; i<28*28; i++) inp[i] = px[i*4]/255;

  try {
    var tensor  = new ort.Tensor("float32", inp, [1,1,28,28]);
    var session = await ort.InferenceSession.create("model/lenet.onnx");
    var feeds   = {}; feeds[session.inputNames[0]] = tensor;
    var res     = await session.run(feeds);

    var rawOut = res["output"].data;
    var conv1  = res["conv1"].data;
    var conv2  = res["conv2"].data;

    var probs  = softmax(Array.from(rawOut));
    var topIdx = probs.indexOf(Math.max.apply(null, probs));

    // UI updates
    document.getElementById("res-text").innerHTML =
      "LeNet thinks you write a <span class=\"big\">" + topIdx + "</span>";
    document.querySelectorAll(".dc").forEach(function(c){ c.classList.remove("on"); });
    document.getElementById("d" + topIdx).classList.add("on");
    updateCBars(probs, topIdx);

    // 3D — staggered layer reveal
    resetLayers();
    var D = 120;

    // INPUT
    addSinglePlane(LD[0], input2tex(inp), 0x00c8ff, 0);

    // CONV1 (6 × 24×24)
    buildActLayer(LD[1], function(c){ return activation2tex(conv1, c*24*24, 24); }, 6, D);

    // POOL1
    var pool1 = maxPool(conv1, 6, 24);
    buildActLayer(LD[2], function(c){ return activation2tex(pool1, c*12*12, 12); }, 6, D*2);

    // CONV2 (16 × 8×8)
    buildActLayer(LD[3], function(c){ return activation2tex(conv2, c*8*8, 8); }, 16, D*3);

    // POOL2
    var pool2 = maxPool(conv2, 16, 8);
    buildActLayer(LD[4], function(c){ return activation2tex(pool2, c*4*4, 4); }, 16, D*4);

    // FC1 (placeholder probability bar)
    addSinglePlane(LD[5], bars2tex(probs), 0x00c8ff, D*5);

    // OUTPUT bar
    addSinglePlane(LD[6], bars2tex(probs), 0x00ff9d, D*6);

    // Camera ease to diagonal view
    setTimeout(function() {
      camTo = {
        pos: new THREE.Vector3(50, 60, 230),
        tgt: new THREE.Vector3(-20, 0, 0),
      };
    }, D*3);

    setStatus("idle", "Predicted: " + topIdx + "  (" + (probs[topIdx]*100).toFixed(1) + "%)");

  } catch(err) {
    console.error(err);
    setStatus("idle", "Error \u2014 see console");
    document.getElementById("res-text").textContent = "Model error \u2014 check console";
  }
}

// ───────────────────────────────────────────────────────────
//  BOOT
// ───────────────────────────────────────────────────────────
setLoad(15, "Initializing Three.js\u2026");
init3D();
setLoad(55, "Building network skeleton\u2026");
setTimeout(function() {
  setLoad(100, "Ready.");
  setTimeout(function() { document.getElementById("loader").classList.add("gone"); }, 450);
}, 500);