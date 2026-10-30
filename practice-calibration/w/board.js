// Shared core for the chessboard-corner widgets: a synthetic camera image of a board,
// a simplified findChessboardCorners pipeline and OpenCV's cornerSubPix iteration.
// Pure functions work in the browser and in node; drawing helpers need a DOM.
(function (G) {
const B = {};

// ---------- camera and board ----------
// Board: cols × rows squares of size s mm with a white margin. Default 6 × 9 squares → pattern (5, 8).
B.board = { cols: 6, rows: 9, s: 25, margin: 15 };
const rx = a => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
const ry = a => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
const rz = a => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
const mm = (A, C) => A.map(r => C[0].map((_, j) => r.reduce((s, v, k) => s + v * C[k][j], 0)));
function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, Bq = -(d * i - f * g), C = d * h - e * g, det = a * A + b * Bq + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
          [Bq / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
// pose: tilt angles in degrees, board centre at (tx, ty, Z) mm in camera coordinates
B.camera = function (o) {
  const d = Math.PI / 180, W = o.W || 640, H = o.H || 480, f = o.f || 600;
  const R = mm(mm(rx((o.ax || 0) * d), ry((o.ay || 0) * d)), rz((o.az || 0) * d));
  const t = [o.tx || 0, o.ty || 0, o.Z || 420], cx = (W - 1) / 2, cy = (H - 1) / 2;
  const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]];
  const Hm = mm(K, [[R[0][0], R[0][1], t[0]], [R[1][0], R[1][1], t[1]], [R[2][0], R[2][1], t[2]]]);
  return { W, H, f, cx, cy, R, t, K, Hm, Hi: inv3(Hm) };
};
B.project = function (cam, X, Y, Z = 0) {
  const R = cam.R, t = cam.t;
  const x = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0], y = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1],
        z = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
  return [cam.f * x / z + cam.cx, cam.f * y / z + cam.cy, z];
};
// inner corner (c, r): object point (c·s, r·s, 0); on the plane it sits at the board centre offset
B.cornerXY = (bd, c, r) => [(c + 1 - bd.cols / 2) * bd.s, (r + 1 - bd.rows / 2) * bd.s];
B.innerCorners = function (cam, bd = B.board) {
  const out = [];
  for (let r = 0; r < bd.rows - 1; r++) for (let c = 0; c < bd.cols - 1; c++) {
    const [X, Y] = B.cornerXY(bd, c, r), [u, v] = B.project(cam, X, Y);
    out.push({ c, r, obj: [c * bd.s, r * bd.s, 0], u, v });
  }
  return out;
};

// ---------- rendering ----------
function rng(seed) { let a = seed | 0; return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
B.rng = rng;
function blur(img, W, H, sigma) {
  if (!(sigma > 0.05)) return img;
  const r = Math.ceil(3 * sigma), k = []; let ks = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); ks += v; }
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0; for (let i = -r; i <= r; i++) s += k[i + r] * img[y * W + Math.min(W - 1, Math.max(0, x + i))]; tmp[y * W + x] = s;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0; for (let i = -r; i <= r; i++) s += k[i + r] * tmp[Math.min(H - 1, Math.max(0, y + i)) * W + x]; out[y * W + x] = s;
  }
  return out;
}
B.blur = blur;
// o: camera options + {blur, noise, seed, light, ss}; pixel (u, v) has its centre at integer coordinates
B.render = function (o, bd = B.board) {
  // stratified jittered supersampling: a regular grid would snap edges to 1/ss of a pixel
  const cam = B.camera(o), W = cam.W, H = cam.H, Hi = cam.Hi, ss = o.ss || 4, J = rng(12345);
  const img = new Float32Array(W * H), m = bd.margin / bd.s, light = o.light || 0;
  for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
    let acc = 0;
    for (let sy = 0; sy < ss; sy++) for (let sx = 0; sx < ss; sx++) {
      const px = u - 0.5 + (sx + J()) / ss, py = v - 0.5 + (sy + J()) / ss;
      const w = Hi[2][0] * px + Hi[2][1] * py + Hi[2][2];
      const X = (Hi[0][0] * px + Hi[0][1] * py + Hi[0][2]) / w, Y = (Hi[1][0] * px + Hi[1][1] * py + Hi[1][2]) / w;
      const bx = X / bd.s + bd.cols / 2, by = Y / bd.s + bd.rows / 2;
      let g;
      if (w > 0 && bx >= 0 && bx < bd.cols && by >= 0 && by < bd.rows) g = ((Math.floor(bx) + Math.floor(by)) & 1) ? 0.9 : 0.1;
      else if (w > 0 && bx >= -m && bx < bd.cols + m && by >= -m && by < bd.rows + m) g = 0.9;
      else g = 0.42 + 0.05 * Math.sin(px * 0.045 + 1) * Math.cos(py * 0.06);
      acc += g;
    }
    img[v * W + u] = acc / (ss * ss) * (1 - light * (u / W - 0.4));
  }
  let out = blur(img, W, H, o.blur || 0);
  if (o.noise > 0) {
    const R = rng(o.seed || 1);
    for (let i = 0; i < out.length; i++) out[i] += o.noise * Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  }
  return { img: out, W, H, cam, truth: B.innerCorners(cam, bd) };
};

// ---------- method 1: simplified findChessboardCorners ----------
// 1) adaptive threshold  2) black quads  3) glue quads by shared vertices  4) order, found only if count matches
B.adaptiveThreshold = function (img, W, H, block = 25, C = 0.02) {
  const S = new Float64Array((W + 1) * (H + 1)), r = block >> 1, out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) { let row = 0; for (let x = 0; x < W; x++) { row += img[y * W + x]; S[(y + 1) * (W + 1) + x + 1] = S[y * (W + 1) + x + 1] + row; } }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r), y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r);
    const s = S[(y1 + 1) * (W + 1) + x1 + 1] - S[y0 * (W + 1) + x1 + 1] - S[(y1 + 1) * (W + 1) + x0] + S[y0 * (W + 1) + x0];
    out[y * W + x] = img[y * W + x] > s / ((x1 - x0 + 1) * (y1 - y0 + 1)) - C ? 1 : 0;   // 1 = white, 0 = black
  }
  return out;
};
function erodeBlack(bin, W, H) {   // black shrinks by one pixel so diagonal squares come apart
  const out = bin.slice();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (bin[y * W + x]) continue;
    if ((x > 0 && bin[y * W + x - 1]) || (x < W - 1 && bin[y * W + x + 1]) || (y > 0 && bin[(y - 1) * W + x]) || (y < H - 1 && bin[(y + 1) * W + x])) out[y * W + x] = 1;
  }
  return out;
}
function components(bin, W, H) {
  const lab = new Int32Array(W * H).fill(-1), comps = [];
  for (let i = 0; i < W * H; i++) {
    if (bin[i] || lab[i] >= 0) continue;
    const pts = [], st = [i]; lab[i] = comps.length;
    while (st.length) {
      const k = st.pop(), x = k % W, y = (k - x) / W; pts.push([x, y]);
      for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const n = ny * W + nx; if (!bin[n] && lab[n] < 0) { lab[n] = comps.length; st.push(n); }
      }
    }
    comps.push(pts);
  }
  return { lab, comps };
}
function fitQuad(pts) {
  let cx = 0, cy = 0; for (const p of pts) { cx += p[0]; cy += p[1]; } cx /= pts.length; cy /= pts.length;
  const far = (from, f) => { let b = null, bd = -1; for (const p of pts) { const d = f(p, from); if (d > bd) { bd = d; b = p; } } return b; };
  const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
  const p1 = far([cx, cy], d2), p2 = far(p1, d2);
  const side = p => (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]);
  const p3 = far(null, p => side(p)), p4 = far(null, p => -side(p));
  const poly = [p1, p3, p2, p4];
  let area = 0; for (let i = 0; i < 4; i++) { const a = poly[i], b = poly[(i + 1) % 4]; area += a[0] * b[1] - b[0] * a[1]; }
  area = Math.abs(area) / 2;
  const edges = poly.map((p, i) => Math.sqrt(d2(p, poly[(i + 1) % 4])));
  return { poly, area, edge: edges.reduce((a, b) => a + b) / 4, minEdge: Math.min(...edges), maxEdge: Math.max(...edges) };
}
B.find = function (img, W, H, pattern, opt = {}) {
  const res = { found: false, pattern };
  res.bin = B.adaptiveThreshold(img, W, H, opt.block || 51, opt.C ?? 0.02);
  res.eroded = erodeBlack(res.bin, W, H);
  const { comps } = components(res.eroded, W, H);
  res.quads = [];
  for (const pts of comps) {
    if (pts.length < 40 || pts.length > W * H / 25) continue;
    const q = fitQuad(pts);
    const ratio = pts.length / Math.max(q.area, 1);
    if (ratio < 0.8 || ratio > 1.3 || q.minEdge < 4 || q.maxEdge > 3 * q.minEdge) continue;
    q.n = pts.length; res.quads.push(q);
  }
  // glue: a vertex of one quad close to a vertex of another quad = one inner corner
  const corners = [], vid = res.quads.map(() => [-1, -1, -1, -1]);
  for (let a = 0; a < res.quads.length; a++) for (let b = a + 1; b < res.quads.length; b++) {
    const A = res.quads[a], Bq = res.quads[b], lim = 0.5 * Math.min(A.minEdge, Bq.minEdge);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      if (vid[a][i] >= 0 || vid[b][j] >= 0) continue;
      const p = A.poly[i], q = Bq.poly[j];
      if (Math.hypot(p[0] - q[0], p[1] - q[1]) < lim) {
        vid[a][i] = vid[b][j] = corners.length;
        corners.push({ u: (p[0] + q[0]) / 2, v: (p[1] + q[1]) / 2, nb: [] });
      }
    }
  }
  res.corners = corners;
  // neighbours: two corners on the same edge of a quad
  res.quads.forEach((q, k) => { for (let i = 0; i < 4; i++) {
    const a = vid[k][i], b = vid[k][(i + 1) % 4];
    if (a >= 0 && b >= 0 && !corners[a].nb.includes(b)) { corners[a].nb.push(b); corners[b].nb.push(a); }
  } });
  const want = pattern[0] * pattern[1];
  if (corners.length !== want) { res.reason = `${corners.length} corners glued, pattern needs ${pattern[0]}×${pattern[1]} = ${want}`; return res; }
  // grid coordinates by walking along quad edges
  const g = corners.map(() => null), vec = (a, b) => [corners[b].u - corners[a].u, corners[b].v - corners[a].v];
  g[0] = [0, 0]; const queue = [0], stepOf = new Map();
  while (queue.length) {
    const a = queue.shift();
    for (const b of corners[a].nb) {
      if (g[b]) continue;
      const e = vec(a, b); let step = null;
      for (const n of corners[a].nb) {
        if (!g[n] || n === b) continue;
        const en = vec(a, n), sn = [g[n][0] - g[a][0], g[n][1] - g[a][1]];
        const cos = (e[0] * en[0] + e[1] * en[1]) / Math.hypot(...e) / Math.hypot(...en);
        if (cos > 0.7) step = sn; else if (cos < -0.7) step = [-sn[0], -sn[1]];
        else step = (en[0] * e[1] - en[1] * e[0]) > 0 ? [-sn[1], sn[0]] : [sn[1], -sn[0]];
        break;
      }
      if (!step) step = [1, 0];
      g[b] = [g[a][0] + step[0], g[a][1] + step[1]]; queue.push(b);
    }
  }
  if (g.some(x => !x)) { res.reason = 'corners do not form one connected grid'; return res; }
  const xs = g.map(p => p[0]), ys = g.map(p => p[1]), x0 = Math.min(...xs), y0 = Math.min(...ys);
  const nx = Math.max(...xs) - x0 + 1, ny = Math.max(...ys) - y0 + 1;
  let gx = g.map(p => [p[0] - x0, p[1] - y0]);
  const [w, h] = pattern;
  if (!(nx === w && ny === h) && !(nx === h && ny === w)) { res.reason = `grid is ${nx}×${ny}, pattern is ${w}×${h}`; return res; }
  if (!(nx === w && ny === h)) gx = gx.map(p => [p[1], p[0]]);          // board seen rotated: swap axes
  if (new Set(gx.map(p => p[0] + ',' + p[1])).size !== want) { res.reason = 'grid has holes'; return res; }
  // orient: rows run left → right, row index grows downwards
  const dir = (ax) => { let s = [0, 0]; gx.forEach((p, i) => gx.forEach((q, j) => { if (q[ax] === p[ax] + 1 && q[1 - ax] === p[1 - ax]) { s[0] += corners[j].u - corners[i].u; s[1] += corners[j].v - corners[i].v; } })); return s; };
  const dx = dir(0), dy = dir(1);
  if (dx[0] < 0) gx = gx.map(p => [w - 1 - p[0], p[1]]);
  if (dy[1] < 0) gx = gx.map(p => [p[0], h - 1 - p[1]]);
  res.ordered = new Array(want);
  gx.forEach((p, i) => { res.ordered[p[1] * w + p[0]] = [corners[i].u, corners[i].v]; });
  res.found = true;
  return res;
};

// ---------- method 2: cornerSubPix (same iteration as OpenCV) ----------
B.sample = function (img, W, H, x, y) {
  x = Math.min(W - 1.001, Math.max(0, x)); y = Math.min(H - 1.001, Math.max(0, y));
  const i = Math.floor(x), j = Math.floor(y), a = x - i, b = y - j, k = j * W + i;
  return (1 - a) * (1 - b) * img[k] + a * (1 - b) * img[k + 1] + (1 - a) * b * img[k + W] + a * b * img[k + W + 1];
};
// one step: gradients in a (2·win+1)² window around p, weighted by exp(−r²/win²)
B.subpixStep = function (img, W, H, p, win = 11) {
  let a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;
  const coeff = 1 / (win * win), S = (dx, dy) => B.sample(img, W, H, p[0] + dx, p[1] + dy);
  for (let y = -win; y <= win; y++) for (let x = -win; x <= win; x++) {
    const m = Math.exp(-coeff * x * x) * Math.exp(-coeff * y * y);
    const gx = S(x + 1, y) - S(x - 1, y), gy = S(x, y + 1) - S(x, y - 1);
    const gxx = gx * gx * m, gxy = gx * gy * m, gyy = gy * gy * m;
    a += gxx; b += gxy; c += gyy; bb1 += gxx * x + gxy * y; bb2 += gxy * x + gyy * y;
  }
  const det = a * c - b * b;
  if (Math.abs(det) < 1e-12) return null;
  return [p[0] + (c * bb1 - b * bb2) / det, p[1] + (-b * bb1 + a * bb2) / det];
};
B.subpix = function (img, W, H, p0, win = 11, maxIter = 30, eps = 0.001) {
  const path = [{ p: p0.slice(), shift: null }];
  let p = p0.slice(), reason = `max_iter = ${maxIter} reached`;
  for (let it = 0; it < maxIter; it++) {
    const q = B.subpixStep(img, W, H, p, win);
    if (!q) { reason = 'gradient matrix is singular'; break; }
    const shift = Math.hypot(q[0] - p[0], q[1] - p[1]);
    p = q; path.push({ p: p.slice(), shift });
    if (shift <= eps) { reason = `shift ${shift.toFixed(4)} px < eps = ${eps}`; break; }
  }
  // like OpenCV: a result that ran away by more than the window is rejected
  if (Math.hypot(p[0] - p0[0], p[1] - p0[1]) > win) { p = p0.slice(); reason = 'ran away from the window, start point kept'; }
  return { p, path, reason };
};
B.gradient = (img, W, H, x, y) => [(B.sample(img, W, H, x + 1, y) - B.sample(img, W, H, x - 1, y)) / 2, (B.sample(img, W, H, x, y + 1) - B.sample(img, W, H, x, y - 1)) / 2];

// ---------- drawing helpers (browser only) ----------
if (typeof document !== 'undefined') {
  B.css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  B.toCanvas = function (img, W, H, map) {
    const o = document.createElement('canvas'); o.width = W; o.height = H;
    const c = o.getContext('2d'), d = c.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const g = map ? map(i) : Math.round(255 * Math.max(0, Math.min(1, img[i])));
      if (Array.isArray(g)) d.data.set([g[0], g[1], g[2], 255], 4 * i); else d.data.set([g, g, g, 255], 4 * i);
    }
    c.putImageData(d, 0, 0); return o;
  };
  // size a canvas to its CSS width with the given aspect ratio; returns [ctx, cssWidth, cssHeight]
  B.prep = function (cv, aspect) {
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = w * aspect;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px';
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); return [c, w, h];
  };
  // OpenCV drawChessboardCorners row colours (BGR table converted to RGB)
  B.rowColors = ['#ff0000', '#ff8000', '#c8c800', '#00ff00', '#00c8c8', '#0000ff', '#ff00ff'];
  B.drawOrder = function (c, pts, w, S) {
    for (let i = 0; i < pts.length; i++) {
      const col = B.rowColors[Math.floor(i / w) % 7], [x, y] = S(pts[i]);
      if (i) { const [px, py] = S(pts[i - 1]); c.strokeStyle = col; c.lineWidth = 1.5; c.beginPath(); c.moveTo(px, py); c.lineTo(x, y); c.stroke(); }
      c.strokeStyle = col; c.lineWidth = 1.5; c.beginPath(); c.arc(x, y, 4, 0, 7); c.stroke();
    }
  };
  B.postHeight = function () {
    const post = () => parent.postMessage({ cvNotesHeight: Math.ceil(document.body.getBoundingClientRect().height) + 8 }, '*');
    addEventListener('load', post); if (window.ResizeObserver) new ResizeObserver(post).observe(document.body); setTimeout(post, 300);
  };
}
G.Board = B;
})(typeof window !== 'undefined' ? window : globalThis);
