// ═══════════════════════════════════════════════════════════
//  LeNet Visualizer — app.js
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
  const mx = Math.max(...arr), ex = arr.map(x => Math.exp(x - mx));
  const s = ex.reduce((a, b) => a + b, 0);
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
//  LAYER DEFINITIONS
//  collapsible: only layers with spreadFull > 0 can collapse (fan layers)
//  For connection lines:
//    fan layers  → one node per visible plane (up to 8), node pos = plane world pos
//    single-plane layers → nodes distributed vertically within plane height
// ───────────────────────────────────────────────────────────
const LD = [
  { id:"input", label:"INPUT",  x:-230, ch: 1, sz:28, pw:9.5, ph:9.5, spreadFull:0,   zSlant:0,   tilt:Math.PI*0.18, collapsible:false },
  { id:"conv1", label:"CONV1",  x:-140, ch: 6, sz:24, pw:9,   ph:9,   spreadFull:6.5, zSlant:3.5, tilt:Math.PI*0.18, collapsible:true  },
  { id:"pool1", label:"POOL1",  x: -70, ch: 6, sz:12, pw:5.5, ph:5.5, spreadFull:6.5, zSlant:3.5, tilt:Math.PI*0.18, collapsible:true  },
  { id:"conv2", label:"CONV2",  x:  10, ch:16, sz: 8, pw:6,   ph:6,   spreadFull:4.5, zSlant:2.5, tilt:Math.PI*0.18, collapsible:true  },
  { id:"pool2", label:"POOL2",  x:  75, ch:16, sz: 4, pw:3.5, ph:3.5, spreadFull:4.5, zSlant:2.5, tilt:Math.PI*0.18, collapsible:true  },
  { id:"fc1",   label:"FC1",    x: 135, ch: 1, sz: 1, pw:3,   ph:20,  spreadFull:0,   zSlant:0,   tilt:0,            collapsible:false },
  { id:"out",   label:"OUTPUT", x: 185, ch: 1, sz: 1, pw:5,   ph:28,  spreadFull:0,   zSlant:0,   tilt:0,            collapsible:false },
];

// Live spread values (animated)
var layerSpread    = LD.map(d => d.spreadFull);
var layerCollapsed = LD.map(() => false);

// ───────────────────────────────────────────────────────────
//  THREE.JS SCENE
// ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let autoRotate = true, rotY = 0;
let frameCount = 0, lastFpsT = performance.now();

let liveMeshes     = [];
let liveMeshesByLayer = {};  // di → array of {mesh, edge, channelOff} for repositioning
let skeletonGroups = [];   // one Group per LD index
let connLineGroups = [];   // one Group per adjacent pair
let labelSprites   = [];

const raycaster   = new THREE.Raycaster();
const mouse       = new THREE.Vector2(-9999, -9999);
const meshToLayer = new Map();

let hoveredLayerIdx = -1;
let camTo = null;
var spreadAnims = [];

// ───────────────────────────────────────────────────────────
//  INIT
// ───────────────────────────────────────────────────────────
function init3D() {
  const vp = document.getElementById("viewport");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x040d18);
  scene.fog = new THREE.FogExp2(0x040d18, 0.0015);

  camera = new THREE.PerspectiveCamera(52, vp.clientWidth / vp.clientHeight, 0.5, 3000);
  camera.position.set(-10, 65, 270);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(vp.clientWidth, vp.clientHeight);
  vp.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.minDistance = 50; controls.maxDistance = 700;
  controls.target.set(-20, 0, 0);
  controls.addEventListener("start", () => { autoRotate = false; });

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  var pl1 = new THREE.PointLight(0x55ddff, 1.8, 900); pl1.position.set(-60, 180, 160);  scene.add(pl1);
  var pl2 = new THREE.PointLight(0x00ffaa, 1.0, 900); pl2.position.set(120,-100,-140);  scene.add(pl2);
  var pl3 = new THREE.PointLight(0xffffff, 0.7, 600); pl3.position.set(0,-200,60);      scene.add(pl3);

  var grid = new THREE.GridHelper(700, 50, 0x0c2030, 0x0c2030);
  grid.position.y = -45; grid.material.transparent = true; grid.material.opacity = 0.7;
  scene.add(grid);

  buildSkeleton();
  addLayerLabels();
  buildConnLines();

  vp.addEventListener("mousemove", function(e) {
    const r = vp.getBoundingClientRect();
    mouse.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
    mouse.y = -((e.clientY - r.top)  / r.height) * 2 + 1;
  });
  vp.addEventListener("mouseleave", function() {
    mouse.set(-9999, -9999); setHoveredLayer(-1);
  });
  window.addEventListener("resize", function() {
    camera.aspect = vp.clientWidth / vp.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(vp.clientWidth, vp.clientHeight);
  });

  animate();
}

// ───────────────────────────────────────────────────────────
//  SKELETON
// ───────────────────────────────────────────────────────────
function buildSkeleton() {
  skeletonGroups.forEach(g => scene.remove(g));
  skeletonGroups = [];
  meshToLayer.clear();

  LD.forEach(function(def, di) {
    var g     = new THREE.Group();
    var sp    = layerSpread[di];
    var count = Math.min(def.ch, 16);
    var ratio = def.spreadFull > 0 ? sp / def.spreadFull : 1;

    for (var i = 0; i < count; i++) {
      var off  = count === 1 ? 0 : (i - count / 2 + 0.5);
      var yPos = off * sp;
      var zPos = off * def.zSlant * ratio;

      var geo = new THREE.PlaneGeometry(def.pw, def.ph);
      var mat = new THREE.MeshStandardMaterial({
        color: 0x112840, transparent: true, opacity: 0.75,
        side: THREE.DoubleSide, roughness: 0.7,
        emissive: 0x061420, emissiveIntensity: 0.4,
      });
      var m = new THREE.Mesh(geo, mat);
      m.position.set(def.x, yPos, zPos);
      m.rotation.y = def.tilt;
      meshToLayer.set(m.uuid, di);
      g.add(m);

      var eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
      var em = new THREE.LineBasicMaterial({ color: 0x1e5a80, transparent: true, opacity: 0.9 });
      var el = new THREE.LineSegments(eg, em);
      el.position.set(def.x, yPos, zPos);
      el.rotation.y = def.tilt;
      g.add(el);
    }
    scene.add(g);
    skeletonGroups.push(g);
  });

  // Subtle axis connectors
  for (var i = 0; i < LD.length - 1; i++) {
    var pts = [new THREE.Vector3(LD[i].x, 0, 0), new THREE.Vector3(LD[i+1].x, 0, 0)];
    var geo = new THREE.BufferGeometry().setFromPoints(pts);
    var mat = new THREE.LineBasicMaterial({ color: 0x1a3a55, transparent: true, opacity: 0.5 });
    scene.add(new THREE.Line(geo, mat));
  }
}

// ───────────────────────────────────────────────────────────
//  LABELS
// ───────────────────────────────────────────────────────────
function addLayerLabels() {
  labelSprites = [];
  LD.forEach(function(def, di) {
    var cv = document.createElement("canvas");
    cv.width = 256; cv.height = 48;
    var c = cv.getContext("2d");
    c.font = "bold 22px 'Courier New'";
    c.fillStyle = "#4a9cc0";
    c.textAlign = "center";
    c.fillText(def.label, 128, 30);
    var tex = new THREE.CanvasTexture(cv);
    var sp  = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.95 }));
    sp.scale.set(18, 3.5, 1);
    sp.position.set(def.x, computeLabelY(di), 0);
    scene.add(sp);
    labelSprites.push(sp);
  });
}

function computeLabelY(di) {
  var def   = LD[di];
  var sp    = layerSpread[di];
  var count = Math.min(def.ch, 8);
  var bottomY = count === 1 ? 0 : -(count / 2 - 0.5) * sp;
  return bottomY - def.ph / 2 - 3.5;
}

// ───────────────────────────────────────────────────────────
//  PLANE WORLD POSITIONS  (used for connection line anchors)
//  Returns array of {x,y,z} for each visible plane of a layer.
//  For fan layers: each plane's centre.
//  For single-plane layers: distributed points along Y inside the plane,
//  offset in Z by the tilt so lines emerge from the plane face.
// ───────────────────────────────────────────────────────────
function getLayerNodes(di, side) {
  var def   = LD[di];
  var sp    = layerSpread[di];
  // Match the actual rendered plane count exactly — buildActLayer uses min(ch,16), skeleton uses min(ch,8)
  // For connection lines we want one node per rendered plane, so use min(ch,16)
  var count = Math.min(def.ch, 16);
  var ratio = def.spreadFull > 0 ? sp / def.spreadFull : 1;
  var nodes = [];

  // For a plane rotated by tilt (Y-axis): local +X → world (cos(tilt), 0, -sin(tilt))
  // Note: THREE.js PlaneGeometry faces +Z by default. After rotation.y = tilt:
  //   local +X → world ( cos(tilt), 0, -sin(tilt) )
  //   right face of plane = +X edge in local space
  var faceSign  = side === "right" ? 1 : -1;
  var faceLocal = faceSign * (def.pw / 2 + 0.4);
  var faceWX    =  Math.cos(def.tilt) * faceLocal;
  var faceWZ    = -Math.sin(def.tilt) * faceLocal;   // was +sin — wrong sign

  if (def.spreadFull > 0) {
    // Fan layer — one node per plane at its centre face edge
    for (var i = 0; i < count; i++) {
      var off = count === 1 ? 0 : (i - count / 2 + 0.5);
      nodes.push(new THREE.Vector3(
        def.x + faceWX,
        off * sp,
        off * def.zSlant * ratio + faceWZ
      ));
    }
  } else if (def.spreadFull === 0 && def.ph <= 12) {
    // Small single plane (INPUT, tilt > 0) — single centre node at face
    nodes.push(new THREE.Vector3(def.x + faceWX, 0, faceWZ));
  } else {
    // Tall single plane (FC1, OUTPUT — tilt = 0) — distribute nodes vertically
    var nodeCount = def.id === "out" ? 10 : 8;
    var halfH     = def.ph * 0.42;
    for (var j = 0; j < nodeCount; j++) {
      var yy = nodeCount === 1 ? 0 : (j - (nodeCount-1)/2) * (2 * halfH / Math.max(nodeCount-1, 1));
      nodes.push(new THREE.Vector3(def.x + faceWX, yy, faceWZ));
    }
  }
  return nodes;
}

// ───────────────────────────────────────────────────────────
//  CONNECTION LINES
// ───────────────────────────────────────────────────────────
function buildConnLines() {
  connLineGroups.forEach(g => scene.remove(g));
  connLineGroups = [];

  for (var pi = 0; pi < LD.length - 1; pi++) {
    (function(fromIdx, toIdx) {
      var g = new THREE.Group();
      g.visible = false;
      g.userData.fromIdx = fromIdx;
      g.userData.toIdx   = toIdx;

      var fromNodes = getLayerNodes(fromIdx, "right");
      var toNodes   = getLayerNodes(toIdx,   "left");

      for (var fi = 0; fi < fromNodes.length; fi++) {
        for (var ti = 0; ti < toNodes.length; ti++) {
          var pts = [ fromNodes[fi].clone(), toNodes[ti].clone() ];
          var geo = new THREE.BufferGeometry().setFromPoints(pts);
          var mat = new THREE.LineBasicMaterial({
            color: 0x00ccff, transparent: true, opacity: 0.0,
          });
          g.add(new THREE.Line(geo, mat));
        }
      }

      scene.add(g);
      connLineGroups.push(g);
    })(pi, pi + 1);
  }
}

// ───────────────────────────────────────────────────────────
//  HOVER
// ───────────────────────────────────────────────────────────
function setHoveredLayer(idx) {
  if (idx === hoveredLayerIdx) return;
  hoveredLayerIdx = idx;

  skeletonGroups.forEach(function(g, gi) {
    g.children.forEach(function(child) {
      if (child.type === "Mesh") {
        child.material.opacity          = (idx === -1 || gi === idx) ? 0.75 : 0.20;
        child.material.emissiveIntensity = (gi === idx) ? 1.5 : 0.3;
      }
    });
  });

  liveMeshes.forEach(function(m) {
    if (m.type === "Mesh") {
      var li = meshToLayer.get(m.uuid);
      m.material.opacity = (idx === -1 || li === idx) ? 0.93 : 0.15;
    }
  });

  connLineGroups.forEach(function(g) {
    var show = (idx !== -1) && (g.userData.fromIdx === idx || g.userData.toIdx === idx);
    g.visible = show;
    if (show) {
      g.children.forEach(function(line) { line.material.opacity = 0.25; });
    }
  });
}

// ───────────────────────────────────────────────────────────
//  COLLAPSE / EXPAND  (only for fan layers)
// ───────────────────────────────────────────────────────────
function toggleLayer(idx) {
  if (!LD[idx].collapsible) return;
  var isCollapsed = layerCollapsed[idx];
  layerCollapsed[idx] = !isCollapsed;

  var target = isCollapsed ? LD[idx].spreadFull : 0;
  spreadAnims = spreadAnims.filter(function(a) { return a.idx !== idx; });
  spreadAnims.push({ idx: idx, from: layerSpread[idx], to: target, t0: performance.now(), dur: 450 });

  var btn = document.getElementById("cbtn-" + idx);
  if (btn) {
    btn.classList.toggle("collapsed", !isCollapsed);
    btn.querySelector(".cico").textContent = !isCollapsed ? "▶" : "▼";
    btn.title = (!isCollapsed ? "Expand " : "Collapse ") + LD[idx].label;
  }
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

  var anyChanged = false;
  spreadAnims = spreadAnims.filter(function(a) {
    var t = Math.min((performance.now() - a.t0) / a.dur, 1);
    layerSpread[a.idx] = a.from + (a.to - a.from) * (1 - Math.pow(1 - t, 3));
    if (t >= 1) layerSpread[a.idx] = a.to;
    anyChanged = true;
    return t < 1;
  });

  if (anyChanged) {
    rebuildSkeletonPositions();
    rebuildConnLinePositions();
  }

  // Raycasting
  raycaster.setFromCamera(mouse, camera);
  var hittable = [];
  skeletonGroups.forEach(function(g) {
    g.children.forEach(function(c) { if (c.type === "Mesh") hittable.push(c); });
  });
  liveMeshes.forEach(function(m) { if (m.type === "Mesh") hittable.push(m); });

  var hits = raycaster.intersectObjects(hittable, false);
  if (hits.length > 0) {
    var hitLayer = meshToLayer.get(hits[0].object.uuid);
    if (hitLayer !== undefined) setHoveredLayer(hitLayer);
  } else {
    setHoveredLayer(-1);
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
//  REBUILD ON SPREAD CHANGE
// ───────────────────────────────────────────────────────────
function rebuildSkeletonPositions() {
  LD.forEach(function(def, di) {
    var g     = skeletonGroups[di];
    if (!g) return;
    var sp    = layerSpread[di];
    var count = Math.min(def.ch, 16);
    var ratio = def.spreadFull > 0 ? sp / def.spreadFull : 1;
    var meshes = g.children.filter(function(c) { return c.type === "Mesh"; });
    var edges  = g.children.filter(function(c) { return c.type === "LineSegments"; });

    for (var i = 0; i < count; i++) {
      var off  = count === 1 ? 0 : (i - count / 2 + 0.5);
      var yPos = off * sp;
      var zPos = off * def.zSlant * ratio;
      if (meshes[i]) meshes[i].position.set(def.x, yPos, zPos);
      if (edges[i])  edges[i].position.set(def.x, yPos, zPos);
    }

    // Also reposition live activation planes for this layer
    var liveEntries = liveMeshesByLayer[di];
    if (liveEntries) {
      liveEntries.forEach(function(entry) {
        var yPos = entry.off * sp;
        var zPos = entry.off * def.zSlant * ratio;
        entry.mesh.position.set(def.x, yPos, zPos);
        entry.edge.position.set(def.x, yPos, zPos);
      });
    }

    if (labelSprites[di]) {
      labelSprites[di].position.y = computeLabelY(di);
    }
  });
}

function rebuildConnLinePositions() {
  connLineGroups.forEach(function(g) {
    var fi       = g.userData.fromIdx;
    var ti       = g.userData.toIdx;
    var fromNodes = getLayerNodes(fi, "right");
    var toNodes   = getLayerNodes(ti, "left");
    var lineIdx   = 0;

    for (var fni = 0; fni < fromNodes.length; fni++) {
      for (var tni = 0; tni < toNodes.length; tni++) {
        var line = g.children[lineIdx++];
        if (!line) continue;
        var pos = line.geometry.attributes.position.array;
        pos[0] = fromNodes[fni].x; pos[1] = fromNodes[fni].y; pos[2] = fromNodes[fni].z;
        pos[3] = toNodes[tni].x;   pos[4] = toNodes[tni].y;   pos[5] = toNodes[tni].z;
        line.geometry.attributes.position.needsUpdate = true;
      }
    }
  });
}

// ───────────────────────────────────────────────────────────
//  TEXTURE GENERATORS
// ───────────────────────────────────────────────────────────
function activation2tex(data, offset, size) {
  var cv = document.createElement("canvas"); cv.width = size; cv.height = size;
  var c = cv.getContext("2d"), id = c.createImageData(size, size);
  var mn = safeMin(data), mx = safeMax(data), rng = mx - mn + 1e-7;
  for (var i = 0; i < size * size; i++) {
    var v = Math.pow(Math.max(0, (data[offset + i] - mn) / rng), 0.5);
    id.data[i*4  ] = Math.round(v * 230 + (1-v) * 15);
    id.data[i*4+1] = Math.round(v * 195 + (1-v) * 70);
    id.data[i*4+2] = Math.round(v * 30  + (1-v) * 140);
    id.data[i*4+3] = Math.round(185 + v * 70);
  }
  c.putImageData(id, 0, 0); return new THREE.CanvasTexture(cv);
}

function input2tex(data) {
  var cv = document.createElement("canvas"); cv.width = 28; cv.height = 28;
  var c = cv.getContext("2d"), id = c.createImageData(28, 28);
  for (var i = 0; i < 28*28; i++) {
    var v = Math.round(data[i] * 255);
    id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
  }
  c.putImageData(id, 0, 0); return new THREE.CanvasTexture(cv);
}

function bars2tex(probs) {
  var W=80, H=140, cv = document.createElement("canvas");
  cv.width=W; cv.height=H;
  var c = cv.getContext("2d");
  c.fillStyle="#040d18"; c.fillRect(0,0,W,H);
  var top = probs.indexOf(Math.max.apply(null, probs));
  for (var i=0; i<10; i++) {
    var bh = Math.max(3, probs[i]*115), x = 3 + i * 7.5;
    c.fillStyle = i===top ? "#00ffaa" : "#33ccff";
    c.fillRect(x, H-22-bh, 6, bh);
    c.fillStyle = i===top ? "#00ffaa" : "#5599bb";
    c.font = "bold 8px monospace"; c.textAlign="center";
    c.fillText(i, x+3, H-5);
  }
  return new THREE.CanvasTexture(cv);
}

// ───────────────────────────────────────────────────────────
//  LAYER MANAGEMENT
// ───────────────────────────────────────────────────────────
function resetLayers() {
  liveMeshes.forEach(function(m) { scene.remove(m); });
  liveMeshes = [];
  liveMeshesByLayer = {};
  meshToLayer.clear();
  skeletonGroups.forEach(function(g, di) {
    g.visible = true;
    g.children.forEach(function(c) { if (c.type === "Mesh") meshToLayer.set(c.uuid, di); });
  });
  connLineGroups.forEach(function(g) { g.visible = false; });
  hoveredLayerIdx = -1;
}

function buildActLayer(def, texFn, chCount, delay0) {
  var defIdx = LD.findIndex(function(d) { return d.id === def.id; });
  if (defIdx >= 0 && skeletonGroups[defIdx]) skeletonGroups[defIdx].visible = false;
  if (!liveMeshesByLayer[defIdx]) liveMeshesByLayer[defIdx] = [];

  var visCount = Math.min(chCount, 16);
  for (var ci = 0; ci < visCount; ci++) {
    (function(c) {
      var sp    = layerSpread[defIdx];
      var ratio = def.spreadFull > 0 ? sp / def.spreadFull : 1;
      var off   = visCount === 1 ? 0 : (c - visCount / 2 + 0.5);

      var tex  = texFn(c);
      var geo  = new THREE.PlaneGeometry(def.pw, def.ph);
      var mat  = new THREE.MeshStandardMaterial({
        map: tex, transparent: true, opacity: 0.93,
        side: THREE.DoubleSide, roughness: 0.3, metalness: 0.1,
      });
      var mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(def.x, off * sp, off * def.zSlant * ratio);
      mesh.rotation.y = def.tilt;
      meshToLayer.set(mesh.uuid, defIdx);

      var eg   = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
      var em   = new THREE.LineBasicMaterial({ color: 0x33ddff, transparent: true, opacity: 0.6 });
      var edge = new THREE.LineSegments(eg, em);
      edge.position.copy(mesh.position); edge.rotation.y = def.tilt;

      mesh.scale.set(0.01, 0.01, 0.01); edge.scale.set(0.01, 0.01, 0.01);
      setTimeout(function() {
        scene.add(mesh); scene.add(edge);
        liveMeshes.push(mesh, edge);
        // Store ref with channel offset for collapse repositioning
        liveMeshesByLayer[defIdx].push({ mesh: mesh, edge: edge, off: off });
        easeScale(mesh, 1, 340); easeScale(edge, 1, 340);
      }, delay0 + c * 55);
    })(ci);
  }
}

function easeScale(obj, to, dur) {
  var t0 = performance.now(), from = obj.scale.x;
  function tick(now) {
    var t = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - t, 3);
    var s = from + (to - from) * e; obj.scale.set(s, s, s);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function maxPool(data, ch, inSz) {
  var os = inSz / 2, out = new Float32Array(ch * os * os);
  for (var c=0; c<ch; c++)
    for (var i=0; i<os; i++)
      for (var j=0; j<os; j++) {
        var b = c*inSz*inSz + i*2*inSz + j*2;
        out[c*os*os + i*os + j] = Math.max(data[b]||0, data[b+1]||0, data[b+inSz]||0, data[b+inSz+1]||0);
      }
  return out;
}

function addSinglePlane(def, tex, glowColor, delay) {
  var defIdx = LD.findIndex(function(d) { return d.id === def.id; });
  setTimeout(function() {
    var geo = new THREE.PlaneGeometry(def.pw, def.ph);
    var mat = new THREE.MeshStandardMaterial({ map:tex, transparent:true, opacity:.95, side:THREE.DoubleSide, roughness:0.3 });
    var m   = new THREE.Mesh(geo, mat);
    m.position.set(def.x, 0, 0); m.rotation.y = def.tilt;
    m.scale.set(0.01,0.01,0.01); easeScale(m, 1, 340);
    meshToLayer.set(m.uuid, defIdx);
    scene.add(m); liveMeshes.push(m);

    var eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(def.pw, def.ph));
    var em = new THREE.LineBasicMaterial({ color: glowColor, transparent:true, opacity:0.9 });
    var el = new THREE.LineSegments(eg, em);
    el.position.copy(m.position); el.rotation.y = def.tilt;
    el.scale.set(0.01,0.01,0.01); easeScale(el, 1, 340);
    scene.add(el); liveMeshes.push(el);

    if (defIdx >= 0 && skeletonGroups[defIdx]) skeletonGroups[defIdx].visible = false;
  }, delay);
}

// ───────────────────────────────────────────────────────────
//  PREDICT
// ───────────────────────────────────────────────────────────
async function predict() {
  setStatus("busy", "Running inference\u2026");
  autoRotate = false;
  document.getElementById("hint").style.opacity = "0";

  var tmp = document.createElement("canvas"); tmp.width = 28; tmp.height = 28;
  var tc  = tmp.getContext("2d");
  tc.fillStyle = "#000"; tc.fillRect(0,0,28,28); tc.drawImage(canvas, 0, 0, 28, 28);
  var px  = tc.getImageData(0,0,28,28).data;
  var inp = new Float32Array(28*28);
  for (var i=0; i<28*28; i++) inp[i] = px[i*4]/255;

  try {
    var tensor  = new ort.Tensor("float32", inp, [1,1,28,28]);
    var session = await ort.InferenceSession.create("model/lenet.onnx");
    var feeds   = {}; feeds[session.inputNames[0]] = tensor;
    var res     = await session.run(feeds);

    var rawOut = res["output"].data, conv1 = res["conv1"].data, conv2 = res["conv2"].data;
    var probs  = softmax(Array.from(rawOut));
    var topIdx = probs.indexOf(Math.max.apply(null, probs));

    document.getElementById("res-text").innerHTML =
      "LeNet thinks you write a <span class=\"big\">" + topIdx + "</span>";
    document.querySelectorAll(".dc").forEach(function(c){ c.classList.remove("on"); });
    document.getElementById("d" + topIdx).classList.add("on");
    updateCBars(probs, topIdx);

    resetLayers();

    var D = 120;

    addSinglePlane(LD[0], input2tex(inp), 0x33ddff, 0);
    buildActLayer(LD[1], function(c){ return activation2tex(conv1, c*24*24, 24); }, 6,  D);
    var pool1 = maxPool(conv1, 6, 24);
    buildActLayer(LD[2], function(c){ return activation2tex(pool1, c*12*12, 12); }, 6, D*2);
    buildActLayer(LD[3], function(c){ return activation2tex(conv2, c*8*8,   8);  }, 16, D*3);
    var pool2 = maxPool(conv2, 16, 8);
    buildActLayer(LD[4], function(c){ return activation2tex(pool2, c*4*4,   4);  }, 16, D*4);
    addSinglePlane(LD[5], bars2tex(probs), 0x33ddff, D*5);
    addSinglePlane(LD[6], bars2tex(probs), 0x00ffaa, D*6);

    setTimeout(function() {
      camTo = { pos: new THREE.Vector3(50, 60, 230), tgt: new THREE.Vector3(-20, 0, 0) };
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