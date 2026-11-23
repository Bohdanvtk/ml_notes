// Shared scene for the "Pixels to millimetres" widgets: the A4 marker sheet (same layout as sheet.pdf),
// screws lying on it, a phone camera that can be raised, tilted and rolled, and the homography maths.
(function (G) {
const S = {};

// ---------- the sheet, in mm (origin = top-left corner of the paper, X right, Y down) ----------
S.SHEET = { w: 210, h: 297 };
S.MSIZE = 40;                                   // marker side, mm
// top-left corner of each marker; ids as on the printed sheet: 0 TL, 1 TR, 2 BL, 3 BR
S.MARKERS = [
  { id: 0, x: 10, y: 10, bits: [[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]] },
  { id: 1, x: 160, y: 10, bits: [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]] },
  { id: 2, x: 10, y: 247, bits: [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1]] },
  { id: 3, x: 160, y: 247, bits: [[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]] }];
S.ID_COLORS = ['#2F6FDE', '#E8692E', '#1D9E75', '#B04FC4'];
// four corners of a marker, in the order ArUco returns them: TL, TR, BR, BL
S.markerCorners = m => [[m.x, m.y], [m.x + S.MSIZE, m.y], [m.x + S.MSIZE, m.y + S.MSIZE], [m.x, m.y + S.MSIZE]];
// what sheet_layout.json stores (illustrative structure)
S.LAYOUT = { paper: 'A4', size_mm: [210, 297], marker_mm: 40, markers: { 0: [10, 10], 1: [160, 10], 2: [10, 247], 3: [160, 247] } };
// screws: shaft from (x, y) along angle (deg), length L, diameter d; a head sits before the start
S.SCREWS = [
  { name: 'M3×20 (top)', x: 40, y: 100, ang: 0, L: 20, d: 3 },
  { name: 'M3×20 (bottom)', x: 40, y: 205, ang: 0, L: 20, d: 3 },
  { name: 'M4×30', x: 150, y: 120, ang: 90, L: 30, d: 4 }];
S.screwEnds = s => { const a = s.ang * Math.PI / 180; return [[s.x, s.y], [s.x + s.L * Math.cos(a), s.y + s.L * Math.sin(a)]]; };

// ---------- phone camera ----------
S.PHOTO = { W: 3024, H: 4032, f: 3000 };       // portrait 12 MP photo; f in px
const rx = a => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
const ry = a => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
const rz = a => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
const mm3 = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
// pose: d = distance to the sheet centre along the optical axis (mm), tilt/roll/yaw in degrees, k1 = lens distortion
S.camera = function (o = {}) {
  const d2r = Math.PI / 180, f = o.f || S.PHOTO.f, W = S.PHOTO.W, H = S.PHOTO.H;
  const R = mm3(mm3(rz((o.roll || 0) * d2r), rx((o.tilt || 0) * d2r)), ry((o.yaw || 0) * d2r));
  const T = [S.SHEET.w / 2 + (o.ox || 0), S.SHEET.h / 2 + (o.oy || 0), 0], d = o.d || 300;
  const C = [0, 1, 2].map(i => T[i] - d * R[2][i]);           // camera centre: back along the optical axis
  const k1 = o.k1 || 0, cx = W / 2, cy = H / 2;
  const cam = { R, C, f, W, H, cx, cy, k1, d };
  cam.toCam = P => { const q = [P[0] - C[0], P[1] - C[1], (P[2] || 0) - C[2]]; return R.map(r => r[0] * q[0] + r[1] * q[1] + r[2] * q[2]); };
  cam.project = (P, noDist) => {
    const c = cam.toCam(P), x = c[0] / c[2], y = c[1] / c[2], k = noDist ? 0 : k1, fac = 1 + k * (x * x + y * y);
    return [f * x * fac + cx, f * y * fac + cy];
  };
  cam.depth = P => cam.toCam(P)[2];
  return cam;
};

// ---------- homography ----------
function solve(A, b) {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
S.inv3 = function (m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2], A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
};
// the two rows one pair adds: unknowns h11 h12 h13 h21 h22 h23 h31 h32 (h33 = 1)
S.pairRows = (p, q) => {
  const [u, v] = p, [X, Y] = q;
  return [{ a: [u, v, 1, 0, 0, 0, -u * X, -v * X], b: X }, { a: [0, 0, 0, u, v, 1, -u * Y, -v * Y], b: Y }];
};
// H maps src (photo px) to dst (sheet mm). Points are normalised first so the numbers stay well conditioned.
S.homography = function (src, dst) {
  const norm = P => { const m = [0, 0]; P.forEach(p => { m[0] += p[0] / P.length; m[1] += p[1] / P.length; });
    const s = Math.SQRT2 / (P.reduce((a, p) => a + Math.hypot(p[0] - m[0], p[1] - m[1]), 0) / P.length);
    return { T: [[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]], P: P.map(p => [s * (p[0] - m[0]), s * (p[1] - m[1])]) }; };
  const a = norm(src), b = norm(dst), rows = [], rhs = [];
  a.P.forEach((p, i) => S.pairRows(p, b.P[i]).forEach(r => { rows.push(r.a); rhs.push(r.b); }));
  let h;
  if (rows.length === 8) h = solve(rows, rhs);
  else { const AtA = [...Array(8)].map((_, i) => [...Array(8)].map((_, j) => rows.reduce((s, r) => s + r[i] * r[j], 0)));
    h = solve(AtA, [...Array(8)].map((_, i) => rows.reduce((s, r, k) => s + r[i] * rhs[k], 0))); }
  const Hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  let H = mm3(mm3(S.inv3(b.T), Hn), a.T);
  H = H.map(r => r.map(x => x / H[2][2]));
  return H;
};
// apply H step by step: [x', y', w] = H·[u, v, 1], then divide by w
S.applyH = (H, p) => {
  const xp = H[0][0] * p[0] + H[0][1] * p[1] + H[0][2], yp = H[1][0] * p[0] + H[1][1] * p[1] + H[1][2], w = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
  return { xp, yp, w, X: xp / w, Y: yp / w };
};
// detected marker corners in the photo (what the program finds) and their partners from the layout
S.pairs = function (cam, all16) {
  const out = [];
  S.MARKERS.forEach(m => (all16 ? S.markerCorners(m) : [S.markerCorners(m)[0]]).forEach((c, k) =>
    out.push({ id: m.id, corner: k, mm: c, px: cam.project(c) })));
  return out;
};

// ---------- drawing ----------
// sc = canvas px per photo px; edges are split so a distorting lens bends them
S.path = function (ctx, cam, pts, sc, closed = true) {
  const n = cam.k1 ? 12 : 1, all = [];
  for (let i = 0; i < pts.length - (closed ? 0 : 1); i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    for (let k = 0; k < n; k++) all.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  if (!closed) all.push(pts[pts.length - 1]);
  ctx.beginPath(); all.forEach((p, i) => { const q = cam.project(p); i ? ctx.lineTo(q[0] * sc, q[1] * sc) : ctx.moveTo(q[0] * sc, q[1] * sc); });
  if (closed) ctx.closePath();
};
S.drawScrew = function (ctx, cam, s, sc, col = '#2a2a2a') {
  const a = s.ang * Math.PI / 180, ux = [Math.cos(a), Math.sin(a)], nx = [-ux[1], ux[0]], P = (t, n) => [s.x + ux[0] * t + nx[0] * n, s.y + ux[1] * t + nx[1] * n];
  ctx.fillStyle = col;
  S.path(ctx, cam, [P(0, -s.d / 2), P(s.L, -s.d / 2), P(s.L + s.d * 0.3, 0), P(s.L, s.d / 2), P(0, s.d / 2)], sc); ctx.fill();
  S.path(ctx, cam, [P(-s.d * 0.7, -s.d * 0.95), P(0, -s.d * 0.95), P(0, s.d * 0.95), P(-s.d * 0.7, s.d * 0.95)], sc); ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.lineWidth = 1;
  for (let t = 1; t < s.L; t += 0.7) { S.path(ctx, cam, [P(t, -s.d / 2), P(t + 0.35, s.d / 2)], sc, false); ctx.stroke(); }
};
// o: {markers, ruler, dashed, screws, table}
S.drawPhoto = function (ctx, cam, sc, o = {}) {
  const W = cam.W * sc, H = cam.H * sc;
  ctx.fillStyle = o.table || '#8f8578'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#f6f6f3'; S.path(ctx, cam, [[0, 0], [210, 0], [210, 297], [0, 297]], sc); ctx.fill();
  if (o.markers !== false) S.MARKERS.forEach(m => {
    ctx.fillStyle = '#111'; S.path(ctx, cam, S.markerCorners(m), sc); ctx.fill();
    const c = S.MSIZE / 6; ctx.fillStyle = '#f6f6f3';
    m.bits.forEach((row, r) => row.forEach((b, k) => { if (!b) return; const x = m.x + (k + 1) * c, y = m.y + (r + 1) * c;
      S.path(ctx, cam, [[x, y], [x + c, y], [x + c, y + c], [x, y + c]], sc); ctx.fill(); }));
  });
  ctx.strokeStyle = '#222'; ctx.lineWidth = Math.max(0.6, 0.35 * cam.f / cam.d * sc);
  if (o.ruler !== false) for (let i = 0; i <= 100; i++) { const x = 55 + i, l = i % 10 ? (i % 5 ? 1.5 : 2.5) : 4.5; S.path(ctx, cam, [[x, 30], [x, 30 - l]], sc, false); ctx.stroke(); }
  if (o.ruler !== false) { S.path(ctx, cam, [[55, 30], [155, 30]], sc, false); ctx.stroke(); }
  if (o.dashed !== false) {
    const seg = (a, b) => { const L = Math.hypot(b[0] - a[0], b[1] - a[1]); for (let t = 0; t < L; t += 10) { const t2 = Math.min(L, t + 6);
      S.path(ctx, cam, [[a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L], [a[0] + (b[0] - a[0]) * t2 / L, a[1] + (b[1] - a[1]) * t2 / L]], sc, false); ctx.stroke(); } };
    const r = [[10, 60.5], [200, 60.5], [200, 238], [10, 238]]; for (let i = 0; i < 4; i++) seg(r[i], r[(i + 1) % 4]);
  }
  if (o.screws !== false) S.SCREWS.forEach(s => S.drawScrew(ctx, cam, s, sc));
};

// ---------- page helpers ----------
if (typeof document !== 'undefined') {
  S.css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  S.prep = function (cv, aspect) {
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = w * aspect;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px';
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); return [c, w, h];
  };
  S.dot = (c, x, y, r, col, label, lc) => { c.fillStyle = col; c.beginPath(); c.arc(x, y, r, 0, 7); c.fill();
    if (label) { c.font = '600 12px system-ui,sans-serif'; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,.85)'; c.strokeText(label, x + r + 3, y - r - 2); c.fillStyle = lc || col; c.fillText(label, x + r + 3, y - r - 2); } };
  S.postHeight = function () {
    const post = () => parent.postMessage({ cvNotesHeight: Math.ceil(document.body.getBoundingClientRect().height) + 8 }, '*');
    addEventListener('load', post); if (window.ResizeObserver) new ResizeObserver(post).observe(document.body); setTimeout(post, 300);
  };
}
// number formatting for matrices: big numbers plain, small ones as a·10^k (LaTeX)
S.fmt = v => { const a = Math.abs(v); if (a < 5e-13) return '0'; if (Math.abs(v - Math.round(v)) < 1e-9 && a < 1e6) return String(Math.round(v)); if (a >= 100) return v.toFixed(0); if (a >= 0.01) return v.toFixed(a >= 10 ? 1 : 3);
  const e = Math.floor(Math.log10(a)); return `${(v / 10 ** e).toFixed(2)}\\!\\cdot\\!10^{${e}}`; };
G.Scene = S;
})(typeof window !== 'undefined' ? window : globalThis);
