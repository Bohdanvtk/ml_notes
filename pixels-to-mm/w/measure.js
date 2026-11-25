// measure.py in the browser: a synthetic top view (10 px = 1 mm) of countersunk screws on paper,
// then the same pipeline: GaussianBlur 5×5 → Otsu → MORPH_CLOSE → contours → minAreaRect → thickness profile.
(function (G) {
const M = {};
M.PPM = 10;                                           // px per mm in the top view
// countersunk screws: d = nominal Ø, dh = head Ø, p = thread pitch (mm)
M.TYPES = { M3: { d: 3, dh: 5.5, p: 0.5 }, M4: { d: 4, dh: 7.5, p: 0.7 }, M5: { d: 5, dh: 9.2, p: 0.8 } };
M.STD_D = [3, 4, 5, 6];
M.STD_L = [6, 8, 10, 12, 14, 16, 20, 25, 30];

// radius of the screw silhouette at distance s (mm) from the top of the head
M.radius = function (sc, s) {
  const t = M.TYPES[sc.type], hh = (t.dh - t.d) / 2, depth = 0.05 * t.d, ch = 0.5;   // 90° head cone, thread depth, tip chamfer
  if (s < 0 || s > sc.L) return -1;
  if (s < hh) return t.dh / 2 - s;                                        // cone of the head
  if (s > sc.L - ch) return t.d / 2 - (s - (sc.L - ch));                  // chamfer at the tip
  const f = ((s - hh) / t.p) % 1;                                         // thread: sawtooth between crest and root
  return t.d / 2 - depth * (f < 0.5 ? 2 * f : 2 - 2 * f);
};

function rng(seed) { let a = seed | 0; return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
// scene: {W, H (mm), screws: [{type, L, x, y, ang}], glare (mm, width of the bright stripe), shadow (bool), dust: n, seed}
M.render = function (o) {
  const W = Math.round(o.W * M.PPM), H = Math.round(o.H * M.PPM), img = new Float32Array(W * H), R = rng(o.seed || 1), ss = 3;
  const local = (sc, X, Y) => { const a = sc.ang * Math.PI / 180, dx = X - sc.x, dy = Y - sc.y; return [dx * Math.cos(a) + dy * Math.sin(a), -dx * Math.sin(a) + dy * Math.cos(a)]; };
  const dust = []; for (let i = 0; i < (o.dust || 0); i++) dust.push([R() * o.W, R() * o.H, 0.15 + R() * 0.45]);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    let acc = 0;
    for (let sy = 0; sy < ss; sy++) for (let sx = 0; sx < ss; sx++) {
      const X = (i + (sx + 0.5) / ss) / M.PPM, Y = (j + (sy + 0.5) / ss) / M.PPM;
      let g = 230 + 4 * Math.sin(X * 1.3) * Math.cos(Y * 1.1);            // paper with a faint texture
      for (const sc of o.screws) {
        if (o.shadow) { const [s, t] = local(sc, X - 0.7, Y - 0.7), r = M.radius(sc, s); if (r > 0 && Math.abs(t) <= r + 0.2) g = Math.min(g, 140); }
      }
      for (const sc of o.screws) {
        const [s, t] = local(sc, X, Y), r = M.radius(sc, s);
        if (r > 0 && Math.abs(t) <= r) {
          g = 75 + 18 * (1 - (t / r) ** 2);                               // dark steel, a bit lighter in the middle
          if (o.glare > 0 && t > -0.15 * r - o.glare / 2 && t < -0.15 * r + o.glare / 2) g = 215;   // shiny stripe along the axis
        }
      }
      for (const [dx, dy, dr] of dust) if ((X - dx) ** 2 + (Y - dy) ** 2 < dr * dr) g = 95;
      acc += g;
    }
    img[j * W + i] = acc / (ss * ss) + 5 * Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
  }
  return { img, W, H };
};

// ---------- step 1: GaussianBlur 5×5 (σ = 1.1, what OpenCV uses for ksize 5, sigma 0) ----------
M.blur5 = function (img, W, H) {
  const s = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8, k = [-2, -1, 0, 1, 2].map(x => Math.exp(-x * x / (2 * s * s))), ks = k.reduce((a, b) => a + b);
  const kn = k.map(v => v / ks), tmp = new Float32Array(W * H), out = new Float32Array(W * H), cl = (v, n) => v < 0 ? -v : v >= n ? 2 * n - v - 2 : v;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let a = 0; for (let i = -2; i <= 2; i++) a += kn[i + 2] * img[y * W + cl(x + i, W)]; tmp[y * W + x] = a; }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let a = 0; for (let i = -2; i <= 2; i++) a += kn[i + 2] * tmp[cl(y + i, H) * W + x]; out[y * W + x] = a; }
  return { out, kernel: kn, sigma: s };
};
// ---------- step 2: Otsu threshold + BINARY_INV ----------
M.otsu = function (img) {
  const hist = new Array(256).fill(0); for (const v of img) hist[Math.max(0, Math.min(255, Math.round(v)))]++;
  const n = img.length; let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let wB = 0, sB = 0, best = -1, T = 0; const score = [];
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) { score.push(0); continue; } const wF = n - wB; if (!wF) { score.push(0); break; }
    sB += t * hist[t]; const mB = sB / wB, mF = (sum - sB) / wF, v = wB * wF * (mB - mF) ** 2; score.push(v);
    if (v > best) { best = v; T = t; }
  }
  // the two class means at the chosen T
  let a = 0, b = 0, na = 0, nb = 0; for (let i = 0; i < 256; i++) { if (i <= T) { a += i * hist[i]; na += hist[i]; } else { b += i * hist[i]; nb += hist[i]; } }
  return { T, hist, score, meanDark: a / na, meanLight: b / nb, shareDark: na / n };
};
M.thresholdInv = (img, T) => { const m = new Uint8Array(img.length); for (let i = 0; i < img.length; i++) m[i] = Math.round(img[i]) > T ? 0 : 1; return m; };
// ---------- step 3: MORPH_CLOSE = dilate, then erode, with an elliptic kernel ----------
M.disk = function (size) {
  const r = (size - 1) / 2, pts = []; for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if ((x * x) / ((r + 0.5) ** 2) + (y * y) / ((r + 0.5) ** 2) <= 1) pts.push([x, y]);
  return pts;
};
function morph(m, W, H, K, dilate) {
  const out = new Uint8Array(m.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = dilate ? 0 : 1;
    for (const [dx, dy] of K) { const xx = x + dx, yy = y + dy; const p = (xx < 0 || yy < 0 || xx >= W || yy >= H) ? (dilate ? 0 : 1) : m[yy * W + xx];
      if (dilate && p) { v = 1; break; } if (!dilate && !p) { v = 0; break; } }
    out[y * W + x] = v;
  }
  return out;
}
M.dilate = (m, W, H, size) => morph(m, W, H, M.disk(size), true);
M.erode = (m, W, H, size) => morph(m, W, H, M.disk(size), false);
M.close = (m, W, H, size) => size <= 1 ? m.slice() : M.erode(M.dilate(m, W, H, size), W, H, size);
// ---------- step 4: blobs (what RETR_EXTERNAL contours enclose) ----------
M.blobs = function (m, W, H) {
  const lab = new Int32Array(W * H).fill(-1), out = [];
  for (let i = 0; i < W * H; i++) {
    if (!m[i] || lab[i] >= 0) continue;
    const pts = [], st = [i]; lab[i] = out.length;
    while (st.length) { const k = st.pop(), x = k % W, y = (k - x) / W; pts.push([x, y]);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const n = ny * W + nx; if (m[n] && lab[n] < 0) { lab[n] = out.length; st.push(n); } } }
    let cx = 0, cy = 0; pts.forEach(p => { cx += p[0]; cy += p[1]; });
    out.push({ pts, area: pts.length, cx: cx / pts.length, cy: cy / pts.length });
  }
  return { lab, blobs: out };
};
// holes = background regions not connected to the image border
M.holes = function (m, W, H) {
  const inv = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) inv[i] = m[i] ? 0 : 1;
  const { blobs } = M.blobs(inv, W, H);
  return blobs.filter(b => !b.pts.some(([x, y]) => x === 0 || y === 0 || x === W - 1 || y === H - 1)).length;
};
// ---------- step 5: minAreaRect via the convex hull (rotating calipers) ----------
M.minAreaRect = function (pts) {
  const P = pts.map(p => [p[0] + 0.5, p[1] + 0.5]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for (const p of P) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of P.slice().reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  const hull = lo.slice(0, -1).concat(up.slice(0, -1));
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length], L = Math.hypot(b[0] - a[0], b[1] - a[1]); if (L < 1e-9) continue;
    const ux = [(b[0] - a[0]) / L, (b[1] - a[1]) / L], uy = [-ux[1], ux[0]];
    let mn = [1e9, 1e9], mx = [-1e9, -1e9];
    for (const p of hull) { const s = p[0] * ux[0] + p[1] * ux[1], t = p[0] * uy[0] + p[1] * uy[1]; mn = [Math.min(mn[0], s), Math.min(mn[1], t)]; mx = [Math.max(mx[0], s), Math.max(mx[1], t)]; }
    const w = mx[0] - mn[0], h = mx[1] - mn[1];
    if (!best || w * h < best.area) best = { area: w * h, ux, uy, mn, mx, w, h };
  }
  // long side direction and the 4 corners
  const long = best.w >= best.h ? best.ux : best.uy, corner = (s, t) => [s * best.ux[0] + t * best.uy[0], s * best.ux[1] + t * best.uy[1]];
  const box = [corner(best.mn[0], best.mn[1]), corner(best.mx[0], best.mn[1]), corner(best.mx[0], best.mx[1]), corner(best.mn[0], best.mx[1])];
  let ang = Math.atan2(long[1], long[0]) * 180 / Math.PI; if (ang > 90) ang -= 180; if (ang <= -90) ang += 180;
  return { box, angle: ang, length: Math.max(best.w, best.h), width: Math.min(best.w, best.h), hull };
};
// ---------- step 6: rotate the blob so its long side is horizontal, then count white pixels per column ----------
M.rotateBlob = function (pts, angle) {
  const a = -angle * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  let cx = 0, cy = 0; pts.forEach(p => { cx += p[0] + 0.5; cy += p[1] + 0.5; }); cx /= pts.length; cy /= pts.length;
  const set = new Set(pts.map(p => p[0] + ',' + p[1])), rp = pts.map(p => { const x = p[0] + 0.5 - cx, y = p[1] + 0.5 - cy; return [x * c - y * s, x * s + y * c]; });
  const xs = rp.map(p => p[0]), ys = rp.map(p => p[1]), x0 = Math.floor(Math.min(...xs)) - 1, x1 = Math.ceil(Math.max(...xs)) + 1, y0 = Math.floor(Math.min(...ys)) - 1, y1 = Math.ceil(Math.max(...ys)) + 1;
  const W = x1 - x0, H = y1 - y0, m = new Uint8Array(W * H);
  // nearest-neighbour inverse mapping (like warpAffine with INTER_NEAREST)
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const x = x0 + i + 0.5, y = y0 + j + 0.5, ox = x * c + y * s + cx, oy = -x * s + y * c + cy;
    if (set.has(Math.floor(ox) + ',' + Math.floor(oy))) m[j * W + i] = 1;
  }
  return { m, W, H };
};
M.profile = function (r) {
  const prof = []; for (let i = 0; i < r.W; i++) { let n = 0; for (let j = 0; j < r.H; j++) n += r.m[j * r.W + i]; prof.push(n); }
  let a = 0, b = prof.length - 1; while (a < b && !prof[a]) a++; while (b > a && !prof[b]) b--;
  return prof.slice(a, b + 1);
};
const pct = (arr, q) => { const s = arr.slice().sort((x, y) => x - y), i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i); return s[lo] + (s[hi] - s[lo]) * (i - lo); };
M.percentile = pct;
// ---------- step 7: head side, head end, L, Ø, standard ----------
M.analyze = function (profIn) {
  const n = profIn.length, k = Math.max(1, Math.round(n / 6)), mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const first = mean(profIn.slice(0, k)), last = mean(profIn.slice(n - k)), flipped = last > first;
  const prof = flipped ? profIn.slice().reverse() : profIn.slice();
  const mid = prof.slice(Math.round(0.4 * n), Math.round(0.9 * n)), med = pct(mid, 0.5), lim = 1.25 * med;
  let headEnd = 0; for (let i = 0; i < Math.round(0.4 * n); i++) if (prof[i] > lim) headEnd = i + 1;
  const tipStart = Math.round(0.95 * n), shaft = prof.slice(headEnd, tipStart);
  const p95 = pct(shaft, 0.95), avg = mean(shaft), mx = Math.max(...shaft);
  const L = n / M.PPM, D = p95 / M.PPM;
  const snap = (v, list, tol) => { const s = list.reduce((b, x) => Math.abs(x - v) < Math.abs(b - v) ? x : b); return Math.abs(s - v) <= tol ? s : null; };
  const sd = snap(D, M.STD_D, 0.4), sl = snap(L, M.STD_L, 1);
  return { prof, n, k, first, last, flipped, med, lim, headEnd, tipStart, p95, avg, mx, L, D, sd, sl, label: sd && sl ? `M${sd}×${sl}` : '?' };
};

// ---------- drawing helpers ----------
if (typeof document !== 'undefined') {
  M.grayCanvas = function (img, W, H, map) {
    const o = document.createElement('canvas'); o.width = W; o.height = H; const c = o.getContext('2d'), d = c.createImageData(W, H);
    for (let i = 0; i < W * H; i++) { const v = map ? map(i) : Math.max(0, Math.min(255, Math.round(img[i]))); if (Array.isArray(v)) d.data.set([v[0], v[1], v[2], 255], 4 * i); else d.data.set([v, v, v, 255], 4 * i); }
    c.putImageData(d, 0, 0); return o;
  };
}
G.Measure = M;
})(typeof window !== 'undefined' ? window : globalThis);
