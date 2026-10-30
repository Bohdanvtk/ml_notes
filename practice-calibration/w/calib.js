// Camera calibration core: a synthetic true camera with distortion, several photos of the board,
// and the same recipe as cv2.calibrateCamera: Zhang's closed form for a start (no distortion),
// then Levenberg–Marquardt over all parameters to minimise the reprojection error.
(function (G) {
const C = {};
C.PATTERN = [5, 8]; C.SQUARE = 21.5; C.SIZE = [640, 480];
C.TRUE = { fx: 610, fy: 600, cx: 326, cy: 236, k1: -0.25, k2: 0.08 };

// object points: (c·s, r·s, 0), row by row like cv2.findChessboardCorners
C.objPoints = function () { const o = []; for (let r = 0; r < C.PATTERN[1]; r++) for (let c = 0; c < C.PATTERN[0]; c++) o.push([c * C.SQUARE, r * C.SQUARE, 0]); return o; };

// ---------- small linear algebra ----------
const rot = (ax, ay, az) => {
  const [a, b, c] = [ax, ay, az].map(d => d * Math.PI / 180);
  const X = [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const Y = [[Math.cos(b), 0, Math.sin(b)], [0, 1, 0], [-Math.sin(b), 0, Math.cos(b)]];
  const Z = [[Math.cos(c), -Math.sin(c), 0], [Math.sin(c), Math.cos(c), 0], [0, 0, 1]];
  return mm(mm(X, Y), Z);
};
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
function solve(A, b) {           // Gaussian elimination with partial pivoting
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
function lstsq(rows, rhs) {       // normal equations
  const n = rows[0].length, A = [...Array(n)].map(() => Array(n).fill(0)), b = Array(n).fill(0);
  rows.forEach((r, k) => { for (let i = 0; i < n; i++) { b[i] += r[i] * rhs[k]; for (let j = 0; j < n; j++) A[i][j] += r[i] * r[j]; } });
  return solve(A, b);
}
function smallestEig(S) {         // Jacobi eigen-decomposition of a symmetric matrix
  const n = S.length, A = S.map(r => r.slice()), V = [...Array(n)].map((_, i) => [...Array(n)].map((_, j) => +(i === j)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  let m = 0; for (let i = 1; i < n; i++) if (A[i][i] < A[m][m]) m = i;
  return V.map(r => r[m]);
}

// ---------- camera model (the same one OpenCV uses, radial part) ----------
C.project = function (cam, R, t, X) {
  const p = mv(R, X).map((v, i) => v + t[i]), x = p[0] / p[2], y = p[1] / p[2], r2 = x * x + y * y;
  const d = 1 + cam.k1 * r2 + cam.k2 * r2 * r2;
  return [cam.fx * x * d + cam.cx, cam.fy * y * d + cam.cy];
};

// ---------- synthetic photos ----------
function rng(seed) { let a = seed | 0; return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
C.POSES = [[18, -22, 6, -40, -10, 260], [-25, 15, -8, 40, 15, 260], [10, 30, 20, 0, 0, 245], [-15, -30, -15, 40, -20, 265],
  [30, 5, 0, -50, 30, 265], [-5, -10, 35, 15, 0, 245], [25, 25, -20, 55, 25, 275], [-30, -5, 10, -50, -15, 270],
  [5, 20, -30, 25, 15, 265], [-20, 30, 25, -25, -25, 270]];
C.photos = function (n, cam, noise, seed = 1) {
  const R0 = rng(seed), g = () => Math.sqrt(-2 * Math.log(R0() + 1e-12)) * Math.cos(2 * Math.PI * R0()), obj = C.objPoints();
  const cx = (C.PATTERN[0] - 1) * C.SQUARE / 2, cy = (C.PATTERN[1] - 1) * C.SQUARE / 2;
  return C.POSES.slice(0, n).map(([ax, ay, az, tx, ty, Z]) => {
    const R = rot(ax, ay, az), t = mv(R, [-cx, -cy, 0]).map((v, i) => v + [tx, ty, Z][i]);   // board centre at (tx, ty, Z)
    return { R, t, pts: obj.map(X => { const u = C.project(cam, R, t, X); return [u[0] + noise * g(), u[1] + noise * g()]; }) };
  });
};

// ---------- calibration ----------
function homography(obj, img) {
  const norm = P => { const m = [0, 0]; P.forEach(p => { m[0] += p[0] / P.length; m[1] += p[1] / P.length; });
    const s = Math.SQRT2 / (P.reduce((a, p) => a + Math.hypot(p[0] - m[0], p[1] - m[1]), 0) / P.length);
    return { T: [[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]], P: P.map(p => [s * (p[0] - m[0]), s * (p[1] - m[1])]) }; };
  const a = norm(obj.map(X => [X[0], X[1]])), b = norm(img), rows = [], rhs = [];
  a.P.forEach((p, i) => { const [x, y] = p, [u, v] = b.P[i];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.push(u);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.push(v); });
  const h = lstsq(rows, rhs), Hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  const Tbi = [[1 / b.T[0][0], 0, -b.T[0][2] / b.T[0][0]], [0, 1 / b.T[1][1], -b.T[1][2] / b.T[1][1]], [0, 0, 1]];
  return mm(mm(Tbi, Hn), a.T);
}
function zhangK(Hs) {
  const v = (H, i, j) => [H[0][i] * H[0][j], H[0][i] * H[1][j] + H[1][i] * H[0][j], H[1][i] * H[1][j],
    H[2][i] * H[0][j] + H[0][i] * H[2][j], H[2][i] * H[1][j] + H[1][i] * H[2][j], H[2][i] * H[2][j]];
  const rows = [];
  Hs.forEach(H => { const s = Math.hypot(...H.flat()), Hh = H.map(r => r.map(x => x / s));
    rows.push(v(Hh, 0, 1)); const a = v(Hh, 0, 0), c = v(Hh, 1, 1); rows.push(a.map((x, k) => x - c[k])); });
  rows.push([0, 1, 0, 0, 0, 0]);                                         // zero skew, like OpenCV
  const S = [...Array(6)].map((_, i) => [...Array(6)].map((_, j) => rows.reduce((t, r) => t + r[i] * r[j], 0)));
  let [B11, B12, B22, B13, B23, B33] = smallestEig(S);
  if (B11 < 0) [B11, B12, B22, B13, B23, B33] = [B11, B12, B22, B13, B23, B33].map(x => -x);
  const v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12), lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
  const fx = Math.sqrt(lam / B11), fy = Math.sqrt(lam * B11 / (B11 * B22 - B12 * B12));
  return { fx, fy, cx: -B13 * fx * fx / lam, cy: v0 };
}
function extrinsics(K, H) {
  const Ki = h => [(h[0] - K.cx * h[2]) / K.fx, (h[1] - K.cy * h[2]) / K.fy, h[2]], col = j => [H[0][j], H[1][j], H[2][j]];
  let r1 = Ki(col(0)), r2 = Ki(col(1)), t = Ki(col(2)); const l = 1 / Math.hypot(...r1);
  r1 = r1.map(x => x * l); r2 = r2.map(x => x * l); t = t.map(x => x * l);
  if (t[2] < 0) { r1 = r1.map(x => -x); r2 = r2.map(x => -x); t = t.map(x => -x); }
  // nearest rotation: orthonormalise r1, r2 and complete with r3 = r1 × r2
  const n2 = Math.hypot(...r2), d = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
  r2 = r2.map((x, i) => x - d * r1[i]); const m = Math.hypot(...r2); r2 = r2.map(x => x / m);
  const r3 = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]];
  return { R: [[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]], t, scaleCheck: n2 };
}
C.rodrigues = function (R) {
  const th = Math.acos(Math.max(-1, Math.min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)));
  if (th < 1e-9) return [0, 0, 0];
  const k = th / (2 * Math.sin(th));
  return [(R[2][1] - R[1][2]) * k, (R[0][2] - R[2][0]) * k, (R[1][0] - R[0][1]) * k];
};
C.rotFromVec = function (w) {
  const th = Math.hypot(...w); if (th < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [x, y, z] = w.map(v => v / th), c = Math.cos(th), s = Math.sin(th), C1 = 1 - c;
  return [[c + x * x * C1, x * y * C1 - z * s, x * z * C1 + y * s], [y * x * C1 + z * s, c + y * y * C1, y * z * C1 - x * s], [z * x * C1 - y * s, z * y * C1 + x * s, c + z * z * C1]];
};
// parameter vector: [fx, fy, cx, cy, k1, k2,  rvec_1, tvec_1,  rvec_2, tvec_2, …]
function residuals(par, obj, imgPoints) {
  const cam = { fx: par[0], fy: par[1], cx: par[2], cy: par[3], k1: par[4], k2: par[5] }, out = [];
  imgPoints.forEach((P, i) => { const o = 6 + 6 * i, R = C.rotFromVec(par.slice(o, o + 3)), t = par.slice(o + 3, o + 6);
    P.forEach((p, j) => { const q = C.project(cam, R, t, obj[j]); out.push(q[0] - p[0], q[1] - p[1]); }); });
  return out;
}
C.calibrate = function (imgPoints, maxIter = 40) {
  const obj = C.objPoints(), Hs = imgPoints.map(P => homography(obj, P)), K0 = zhangK(Hs);
  let par = [K0.fx, K0.fy, K0.cx, K0.cy, 0, 0];
  Hs.forEach(H => { const v = extrinsics(K0, H); par.push(...C.rodrigues(v.R), ...v.t); });
  const init = { ...K0, k1: 0, k2: 0 };
  let r = residuals(par, obj, imgPoints), cost = r.reduce((a, x) => a + x * x, 0), mu = 1e-3;
  const n = par.length, history = [Math.sqrt(cost / (r.length / 2))];
  for (let it = 0; it < maxIter; it++) {
    // numeric Jacobian; each view only touches its own 6 columns, but n is small enough to do it plainly
    const J = [];
    for (let k = 0; k < n; k++) { const h = 1e-6 * Math.max(1, Math.abs(par[k])), p2 = par.slice(); p2[k] += h;
      const r2 = residuals(p2, obj, imgPoints); J.push(r2.map((x, m) => (x - r[m]) / h)); }
    const A = [...Array(n)].map((_, a) => [...Array(n)].map((_, b) => { let t = 0; for (let m = 0; m < r.length; m++) t += J[a][m] * J[b][m]; return t; }));
    const g = [...Array(n)].map((_, a) => { let t = 0; for (let m = 0; m < r.length; m++) t += J[a][m] * r[m]; return t; });
    let improved = false;
    for (let tries = 0; tries < 10; tries++) {
      const Ad = A.map((row, a) => row.map((x, b) => a === b ? x * (1 + mu) : x));
      const step = solve(Ad, g.map(x => -x)), p2 = par.map((x, k) => x + step[k]);
      const r2 = residuals(p2, obj, imgPoints), c2 = r2.reduce((a, x) => a + x * x, 0);
      if (c2 < cost) { par = p2; r = r2; improved = cost - c2 > 1e-12 * cost; cost = c2; mu = Math.max(mu / 3, 1e-9); break; }
      mu *= 4;
    }
    history.push(Math.sqrt(cost / (r.length / 2)));
    if (!improved) break;
  }
  const cam = { fx: par[0], fy: par[1], cx: par[2], cy: par[3], k1: par[4], k2: par[5] };
  const rvecs = [], tvecs = [], views = [];
  imgPoints.forEach((_, i) => { const o = 6 + 6 * i, w = par.slice(o, o + 3), t = par.slice(o + 3, o + 6); rvecs.push(w); tvecs.push(t); views.push({ R: C.rotFromVec(w), t }); });
  const reproj = imgPoints.map((P, i) => obj.map(X => C.project(cam, views[i].R, views[i].t, X)));
  return { K: cam, init, dist: [cam.k1, cam.k2, 0, 0, 0], rvecs, tvecs, views, reproj, rms: history[history.length - 1], history };
};
G.Calib = C;
})(typeof window !== 'undefined' ? window : globalThis);
