// CNN(작가 판별·원본 대조) + 이미지 캡셔닝 학습 — TensorFlow.js (WebAssembly 백엔드, 네이티브 DLL 없음)
//
// 구조:
//   MobileNetV2 (ImageNet 사전학습, 고정, public/models/backbone) ─┬─ 작가 판별 헤드 Dense(15, softmax)
//                                                                ├─ 원본 대조 임베딩 (PCA 256차원) → 소장품과 코사인 유사도
//                                                                └─ 캡션 초기값 (h0, ctx) → GRU 디코더가 토큰을 하나씩 만든다
// 전이학습: CNN 본체는 고정하고, 그림마다 보기 4개(전체·좌우 반전·잘라 보기 2개)의 특징을 한 번 뽑아 헤드만 학습한다.
//
// 내보내는 파일 (public/models/) — 앱은 이 파일만으로 인터넷 없이 추론한다:
//   heads/      특징(1280) → [임베딩(256), 작가 확률(15), 캡션 ctx(64), 캡션 h0(256)]
//   decoder/    캡션 한 걸음: [ctx, 토큰, 상태] → [다음 토큰 점수, 새 상태]
//   refs.bin    소장품 임베딩 int8 (그림 수 × 256),  refs.json 소장품 정보
//   meta.json   작가 이름·캡션 단어장·시험 정확도·대조 기준값,  files.json 기기에 저장할 파일 목록
//
// 사용: cd training && npm install && npm run train   (먼저 python prepare_data.py)
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import jpeg from "jpeg-js";
import fs from "node:fs";
import path from "node:path";

const DATA = path.resolve("data");
const PUBLIC = path.resolve("../public");
const OUT = path.join(PUBLIC, "models");
const WORK = path.resolve("work");
const VIEWS = 4;
const IMG = 224;
const FEAT = 1280;
const EMB_DIM = 256;
const CAP_UNITS = 256;
const CTX_DIM = 64;
const MAX_LEN = 9; // <start> 포함

// 재현 가능한 난수
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

setWasmPaths(path.resolve("node_modules/@tensorflow/tfjs-backend-wasm/dist") + path.sep);
await tf.setBackend("wasm");
await tf.ready();
console.log("백엔드:", tf.getBackend());

// ---------- 파일 입출력 (tfjs-node 없이) ----------
function loadGraphFromDir(dir) {
  const json = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const data = Buffer.concat(json.weightsManifest.flatMap((g) => g.paths.map((p) => fs.readFileSync(path.join(dir, p)))));
  return tf.loadGraphModel(tf.io.fromMemory({
    modelTopology: json.modelTopology, format: json.format,
    weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
    weightData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
  }));
}

async function saveLayers(model, dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  await model.save(tf.io.withSaveHandler(async (a) => {
    fs.writeFileSync(path.join(dir, "weights.bin"), Buffer.from(a.weightData));
    fs.writeFileSync(path.join(dir, "model.json"), JSON.stringify({
      modelTopology: a.modelTopology, format: a.format, generatedBy: a.generatedBy, convertedBy: a.convertedBy,
      weightsManifest: [{ paths: ["weights.bin"], weights: a.weightSpecs }],
    }));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
  }));
}

function readView(id, k) {
  const { data } = jpeg.decode(fs.readFileSync(path.join(DATA, "views", `${id}_${k}.jpg`)), { useTArray: true, formatAsRGBA: false });
  return data; // RGB uint8, 224*224*3
}

// ---------- 1) 특징 추출 (한 번만, work/에 저장) ----------
const meta = JSON.parse(fs.readFileSync(path.join(DATA, "metadata.json"), "utf8"));
const { artists } = meta;
const items = meta.items.filter((it) => fs.existsSync(path.join(DATA, "views", `${it.id}_${VIEWS - 1}.jpg`)));
const N = items.length;
console.log(`그림 ${N}점, 보기 ${N * VIEWS}장`);

// 특징은 그림마다 work/feats/<id>.f32 (보기 4개 × 1280)에 저장해 두고, 그림을 더하면 새 그림만 계산한다
const FEAT_DIR = path.join(WORK, "feats");
fs.mkdirSync(FEAT_DIR, { recursive: true });
const featPath = (id) => path.join(FEAT_DIR, `${id}.f32`);
const feats = new Float32Array(N * VIEWS * FEAT); // 행 = i*VIEWS + k
const todo = [];
items.forEach((it, i) => {
  if (fs.existsSync(featPath(it.id))) {
    const b = fs.readFileSync(featPath(it.id));
    feats.set(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), i * VIEWS * FEAT);
  } else todo.push(i);
});
if (todo.length) {
  console.log(`새 그림 ${todo.length}점의 특징을 뽑는다`);
  const backbone = await loadGraphFromDir(path.join(OUT, "backbone"));
  const PER = 8; // 그림 8점 = 보기 32장씩
  const started = Date.now();
  for (let s = 0; s < todo.length; s += PER) {
    const chunk = todo.slice(s, s + PER);
    const buf = new Uint8Array(chunk.length * VIEWS * IMG * IMG * 3);
    chunk.forEach((i, c) => { for (let k = 0; k < VIEWS; k++) buf.set(readView(items[i].id, k), (c * VIEWS + k) * IMG * IMG * 3); });
    const out = tf.tidy(() => backbone.execute(tf.tensor4d(buf, [chunk.length * VIEWS, IMG, IMG, 3], "int32").toFloat().div(255)));
    const data = await out.data();
    out.dispose();
    chunk.forEach((i, c) => {
      const part = data.slice(c * VIEWS * FEAT, (c + 1) * VIEWS * FEAT);
      feats.set(part, i * VIEWS * FEAT);
      fs.writeFileSync(featPath(items[i].id), Buffer.from(part.buffer));
    });
    if ((s / PER) % 25 === 0) {
      const done = s + chunk.length;
      console.log(`  특징 추출 ${done}/${todo.length} (남은 시간 약 ${Math.round((Date.now() - started) / done * (todo.length - done) / 60000)}분)`);
    }
  }
}
const row = (i, k) => feats.subarray((i * VIEWS + k) * FEAT, (i * VIEWS + k + 1) * FEAT);
const center = new Float32Array(N * FEAT);
for (let i = 0; i < N; i++) center.set(row(i, 0), i * FEAT);

// 그림 단위로 학습·시험을 나눈다 (같은 그림의 보기가 양쪽에 섞이지 않게)
function split(indices, ratio, keyOf) {
  const groups = new Map();
  for (const i of indices) { const k = keyOf ? keyOf(i) : "all"; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i); }
  const train = [], test = [];
  for (const g of groups.values()) { shuffle(g); const n = Math.max(1, Math.floor(g.length * ratio)); test.push(...g.slice(0, n)); train.push(...g.slice(n)); }
  return [train.sort((a, b) => a - b), test.sort((a, b) => a - b)];
}
function gather(indices, views) {
  const out = new Float32Array(indices.length * views.length * FEAT);
  let o = 0;
  for (const i of indices) for (const k of views) { out.set(row(i, k), o * FEAT); o++; }
  return out;
}

// ---------- 2) 작가 판별 헤드 ----------
const labeled = items.map((it, i) => (it.label ? i : -1)).filter((i) => i >= 0);
const [artTr, artTe] = split(labeled, 0.2, (i) => items[i].label);
const yOf = (i) => artists.indexOf(items[i].label);
const xTr = tf.tensor2d(gather(artTr, [0, 1, 2, 3]), [artTr.length * VIEWS, FEAT]);
const yTr = tf.tensor1d(artTr.flatMap((i) => Array(VIEWS).fill(yOf(i))), "float32");
const xTe = tf.tensor2d(gather(artTe, [0]), [artTe.length, FEAT]);
const yTe = artTe.map(yOf);

const artistHead = tf.sequential({ name: "artist_head", layers: [
  tf.layers.dropout({ inputShape: [FEAT], rate: 0.4 }),
  tf.layers.dense({ units: artists.length, activation: "softmax", kernelRegularizer: tf.regularizers.l2({ l2: 1e-3 }) }),
] });
artistHead.compile({ optimizer: tf.train.adam(1e-3), loss: "sparseCategoricalCrossentropy" });

function artistAccuracy() {
  const p = tf.tidy(() => artistHead.predict(xTe)).arraySync();
  let top1 = 0, top3 = 0;
  p.forEach((probs, r) => {
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
    top1 += order[0] === yTe[r];
    top3 += order.slice(0, 3).includes(yTe[r]);
  });
  return [top1 / p.length, top3 / p.length];
}
let best = { top1: -1 };
for (let epoch = 1; epoch <= 80; epoch++) {
  await artistHead.fit(xTr, yTr, { epochs: 1, batchSize: 64, shuffle: true, verbose: 0 });
  const [top1, top3] = artistAccuracy();
  if (top1 > best.top1) best = { top1, top3, epoch, weights: artistHead.getWeights().map((w) => w.clone()) };
  if (epoch - best.epoch >= 15) break;
}
artistHead.setWeights(best.weights);
console.log(`작가 판별 시험: top-1 ${best.top1.toFixed(3)}, top-3 ${best.top3.toFixed(3)} (시험 그림 ${artTe.length}점, ${best.epoch}번째 학습에서 최고)`);

// 여러 보기 평균(TTA): 그림 전체·좌우 반전·잘라 보기의 확률을 평균하면 구도보다 붓질·질감 쪽을 더 본다
function meanProbs(indices) {
  return tf.tidy(() => tf.stack([0, 1, 2, 3].map((k) =>
    artistHead.predict(tf.tensor2d(gather(indices, [k]), [indices.length, FEAT])))).mean(0).arraySync());
}
const tePTta = meanProbs(artTe);
let ttaTop1 = 0, ttaTop3 = 0;
tePTta.forEach((probs, r) => {
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  ttaTop1 += order[0] === yTe[r];
  ttaTop3 += order.slice(0, 3).includes(yTe[r]);
});
ttaTop1 /= artTe.length; ttaTop3 /= artTe.length;
console.log(`작가 판별 시험 (보기 4개 평균): top-1 ${ttaTop1.toFixed(3)}, top-3 ${ttaTop3.toFixed(3)}`);

// "모름" 기준값: 15명의 시험 그림(학습 범위 안)과 15명 밖 작가의 그림(학습 범위 밖)에서 최고 확률을 비교해
// 둘을 가장 잘 가르는 값을 고른다 (양쪽 정답률의 평균이 가장 높은 값)
const outside = items.map((it, i) => (it.label ? -1 : i)).filter((i) => i >= 0);
const maxOf = (rows) => rows.map((p) => Math.max(...p));
const inMax = maxOf(tePTta);
const outMax = maxOf(meanProbs(outside));
let calib = { balanced: -1 };
for (let t = 0.2; t <= 0.95; t += 0.01) {
  const inKeep = inMax.filter((m) => m >= t).length / inMax.length;
  const outReject = outMax.filter((m) => m < t).length / outMax.length;
  if ((inKeep + outReject) / 2 > calib.balanced) calib = { threshold: +t.toFixed(2), inKeep, outReject, balanced: (inKeep + outReject) / 2 };
}
console.log(`'모름' 기준: 최고 확률 ${calib.threshold} 미만이면 학습 범위 밖 — 범위 안 그림 ${(calib.inKeep * 100).toFixed(0)}% 통과, ` +
  `범위 밖 그림 ${(calib.outReject * 100).toFixed(0)}% 걸러냄 (범위 밖 ${outside.length}점)`);

// ---------- 3) 원본 대조 임베딩: PCA 256차원 (부분공간 반복법) ----------
const [pcaW, pcaB] = tf.tidy(() => {
  const X = tf.tensor2d(center, [N, FEAT]);
  const mean = X.mean(0);
  const Xc = X.sub(mean);
  const C = Xc.transpose().matMul(Xc).div(N - 1);
  let Q = tf.linalg.qr(tf.randomNormal([FEAT, EMB_DIM], 0, 1, "float32", 7))[0];
  for (let it = 0; it < 40; it++) Q = tf.linalg.qr(C.matMul(Q))[0];
  return [Q, mean.expandDims(0).matMul(Q).neg().reshape([EMB_DIM])];
});
const l2norm = (t) => tf.tidy(() => t.div(t.norm("euclidean", -1, true).add(1e-8)));
const project = (x) => tf.tidy(() => l2norm(x.matMul(pcaW).add(pcaB)));

const refs = project(tf.tensor2d(center, [N, FEAT]));
const allViews = project(tf.tensor2d(feats, [N * VIEWS, FEAT]));
const refArr = await refs.data();
const viewArr = await allViews.data();
// 기준값: 같은 작품을 뒤집거나 잘라 찍은 사진 vs 서로 다른 작품
const same = [];
for (let i = 0; i < N; i++) for (let k = 1; k < VIEWS; k++) {
  let s = 0; for (let j = 0; j < EMB_DIM; j++) s += viewArr[(i * VIEWS + k) * EMB_DIM + j] * refArr[i * EMB_DIM + j];
  same.push(s);
}
const simMat = tf.tidy(() => refs.matMul(refs.transpose()).sub(tf.eye(N).mul(3)).max(1));
const nearestOther = Array.from(await simMat.data());
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p / 100 * s.length))]; };
const thresholds = {
  same_work_p10: +pct(same, 10).toFixed(3), same_work_median: +pct(same, 50).toFixed(3),
  different_work_p99: +pct(nearestOther, 99).toFixed(3), different_work_median: +pct(nearestOther, 50).toFixed(3),
};
console.log("원본 대조 기준값:", thresholds);

// ---------- 4) 이미지 캡셔닝: CNN 특징 → GRU 디코더 ----------
// 단어는 원-핫 벡터로 넣고 Dense(64, 편향 없음)로 임베딩한다 — Embedding 층과 계산은 같지만,
// WebAssembly 백엔드에 Embedding의 기울기 연산(UnsortedSegmentSum)이 없어서 이렇게 하면 빠른 WASM으로 학습할 수 있다
const vocab = ["<pad>", "<start>", "<end>", ...[...new Set(items.flatMap((it) => it.caption))].sort()];
const tok = Object.fromEntries(vocab.map((t, i) => [t, i]));
const V = vocab.length;
const seqs = items.map((it) => {
  const ids = [tok["<start>"], ...it.caption.slice(0, MAX_LEN - 1).map((t) => tok[t]), tok["<end>"]];
  return [...ids, ...Array(MAX_LEN + 1 - ids.length).fill(0)];
});
const [capTr, capTe] = split(items.map((_, i) => i), 0.1);

const fIn = tf.input({ shape: [FEAT], name: "features" });
const ctxL = tf.layers.dense({ units: CTX_DIM, activation: "relu", name: "cap_ctx" });
const h0L = tf.layers.dense({ units: CAP_UNITS, activation: "tanh", name: "cap_h0" });
const embedL = tf.layers.dense({ units: 64, useBias: false, name: "cap_embed" });
const gruL = tf.layers.gru({ units: CAP_UNITS, returnSequences: true, returnState: true, name: "cap_gru" });
const logitsL = tf.layers.dense({ units: V, name: "cap_logits" });

const tIn = tf.input({ shape: [MAX_LEN, V], name: "tokens" });   // 원-핫
const ctxOut = ctxL.apply(fIn);
const h0Out = h0L.apply(fIn);
const seqIn = tf.layers.concatenate().apply([embedL.apply(tIn), tf.layers.repeatVector({ n: MAX_LEN }).apply(ctxOut)]);
const [seqOut] = gruL.apply(seqIn, { initialState: [h0Out] });
const capModel = tf.model({ inputs: [fIn, tIn], outputs: logitsL.apply(seqOut) });

const capRows = capTr.flatMap((i) => [0, 1, 2, 3].map((k) => [i, k]));
const oneHot = (ids) => tf.tidy(() => tf.oneHot(tf.tensor(ids, undefined, "int32"), V).toFloat());
const optimizer = tf.train.adam(2e-3);
function capBatch(rowsB) {
  const f = new Float32Array(rowsB.length * FEAT);
  const xIn = [], yOut = [];
  rowsB.forEach(([i, k], r) => { f.set(row(i, k), r * FEAT); xIn.push(seqs[i].slice(0, MAX_LEN)); yOut.push(seqs[i].slice(1)); });
  return [tf.tensor2d(f, [rowsB.length, FEAT]), oneHot(xIn), tf.tensor2d(yOut, undefined, "int32")];
}
function capLoss(f, x, y, training) {
  const logits = capModel.apply([f, x], { training });
  const mask = y.notEqual(0).toFloat();
  return tf.losses.softmaxCrossEntropy(tf.oneHot(y, V), logits, mask);
}
const [teF, teX, teY] = capBatch(capTe.map((i) => [i, 0]));
let bestCap = { loss: Infinity };
for (let epoch = 1; epoch <= 25; epoch++) {
  const epochStart = Date.now();
  shuffle(capRows);
  let total = 0, batches = 0;
  for (let s = 0; s < capRows.length; s += 128) {
    const [f, x, y] = capBatch(capRows.slice(s, s + 128));
    const loss = optimizer.minimize(() => capLoss(f, x, y, true), true);
    total += (await loss.data())[0]; batches++;
    tf.dispose([f, x, y, loss]);
  }
  const val = tf.tidy(() => capLoss(teF, teX, teY, false).dataSync()[0]);
  console.log(`  캡션 학습 ${epoch}: 학습 손실 ${(total / batches).toFixed(3)}, 시험 손실 ${val.toFixed(3)} (${Math.round((Date.now() - epochStart) / 1000)}초)`);
  if (val < bestCap.loss) bestCap = { loss: val, epoch, weights: capModel.getWeights().map((w) => w.clone()) };
  if (epoch - bestCap.epoch >= 3) break;
}
capModel.setWeights(bestCap.weights);

// 한 걸음 디코더 (기기에서 토큰을 하나씩 만든다) — 학습한 층을 그대로 공유한다
const ctxIn = tf.input({ shape: [CTX_DIM], name: "ctx" });
const tokIn = tf.input({ shape: [1, V], name: "token" });   // 원-핫 (앱도 같은 모양으로 넣는다)
const stIn = tf.input({ shape: [CAP_UNITS], name: "state" });
const stepIn = tf.layers.concatenate().apply([embedL.apply(tokIn), tf.layers.reshape({ targetShape: [1, CTX_DIM] }).apply(ctxIn)]);
const [stepOut, stepState] = gruL.apply(stepIn, { initialState: [stIn] });
const decoder = tf.model({ name: "decoder_step", inputs: [ctxIn, tokIn, stIn],
  outputs: [tf.layers.reshape({ targetShape: [V] }).apply(logitsL.apply(stepOut)), stepState] });

function greedy(ctxV, hV) {
  return tf.tidy(() => {
    let t = tok["<start>"], h = hV;
    const words = [];
    for (let s = 0; s < MAX_LEN; s++) {
      const [lg, nh] = decoder.predict([ctxV, oneHot([[t]]), h]);
      h = nh;
      t = lg.argMax(1).dataSync()[0];
      if (t === tok["<end>"]) break;
      if (words.length && words[words.length - 1] === vocab[t]) continue;
      words.push(vocab[t]);
    }
    return words;
  });
}
const ctxModel = tf.model({ inputs: fIn, outputs: [ctxOut, h0Out] });
let exact = 0, slotHits = 0, slots = 0;
const samples = [];
const evalIdx = capTe.slice(0, 300);
for (const i of evalIdx) {
  const [cv, hv] = ctxModel.predict(tf.tensor2d(row(i, 0), [1, FEAT]));
  const pred = greedy(cv, hv), gold = items[i].caption;
  tf.dispose([cv, hv]);
  exact += pred.join(" ") === gold.join(" ");
  for (let s = 0; s < Math.min(3, gold.length); s++) slotHits += pred[s] === gold[s];
  slots += Math.min(3, gold.length);
  if (samples.length < 8) samples.push({ pred, gold });
}
const capExact = exact / evalIdx.length, capSlot = slotHits / slots;
console.log(`캡션 시험: 완전 일치 ${capExact.toFixed(3)}, 시대·문화권·기법 칸 정확도 ${capSlot.toFixed(3)}`);
samples.forEach((s) => console.log("   ", s.pred.join(" "), " | 정답:", s.gold.join(" ")));

// ---------- 5) 내보내기 ----------
const pcaL = tf.layers.dense({ units: EMB_DIM, name: "pca", weights: [pcaW, pcaB] });
const heads = tf.model({ name: "heads", inputs: fIn,
  outputs: [pcaL.apply(fIn), artistHead.apply(fIn), ctxOut, h0Out] });
await saveLayers(heads, path.join(OUT, "heads"));
await saveLayers(decoder, path.join(OUT, "decoder"));

const refInt8 = Int8Array.from(refArr, (v) => Math.max(-127, Math.min(127, Math.round(v * 127))));
fs.writeFileSync(path.join(OUT, "refs.bin"), Buffer.from(refInt8.buffer));
fs.writeFileSync(path.join(OUT, "refs.json"), JSON.stringify(items.map((it) => ({
  id: it.id, title: it.title, artist: it.artist, date: it.date, source: it.source, url: it.source_url }))));
const r3 = (x) => +x.toFixed(3);
fs.writeFileSync(path.join(OUT, "meta.json"), JSON.stringify({
  image_size: IMG, feature_dim: FEAT, emb_dim: EMB_DIM, cap_units: CAP_UNITS, ctx_dim: CTX_DIM, max_len: MAX_LEN,
  backbone: "MobileNetV2 1.0 224 (ImageNet), 입력 0~1 RGB",
  artists, artist_ko: meta.artist_ko, vocab,
  metrics: { artist_top1: r3(best.top1), artist_top3: r3(best.top3), artist_test_images: artTe.length,
    artist_top1_tta: r3(ttaTop1), artist_top3_tta: r3(ttaTop3),
    caption_exact: r3(capExact), caption_slot_accuracy: r3(capSlot), train_images: N },
  // 작가 최고 확률이 이 값 미만이면 "학습 범위 밖의 그림" — 학습 범위 안 그림 in_keep 비율은 통과, 범위 밖 그림 out_reject 비율은 걸러냄
  artist_scope: { threshold: calib.threshold, in_keep: r3(calib.inKeep), out_reject: r3(calib.outReject), outside_images: outside.length },
  match_thresholds: thresholds, caption_samples: samples,
}, null, 1));

// 기기에 미리 저장할 파일 목록 (앱 + 모델). 썸네일은 볼 때마다 저장한다.
const listDir = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) =>
  e.isDirectory() ? listDir(path.join(d, e.name)) : [path.join(d, e.name)]);
const files = listDir(OUT).filter((f) => !f.endsWith("files.json"))
  .map((f) => path.relative(PUBLIC, f).split(path.sep).join("/"));
fs.writeFileSync(path.join(OUT, "files.json"), JSON.stringify(files, null, 1));
console.log("내보냄:", files.join(", "));
