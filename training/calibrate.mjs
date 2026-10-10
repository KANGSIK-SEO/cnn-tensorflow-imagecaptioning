// 앱이 실제로 쓰는 보기 조합으로 작가 판별 정확도와 "모름" 기준값을 다시 재서 public/models/meta.json에 적는다.
// (train.mjs가 잰 4개 평균은 무작위로 잘라 본 보기가 섞여 정확도가 오히려 떨어졌다 — 2026-10-11)
// 시험 그림은 train.mjs와 같은 난수·같은 순서로 나눠 학습에 쓰지 않은 그림만 쓴다.
// 사용: cd training && node calibrate.mjs   (train.mjs 다음에)
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import fs from "node:fs";
import path from "node:path";

setWasmPaths(path.resolve("node_modules/@tensorflow/tfjs-backend-wasm/dist") + path.sep);
await tf.setBackend("wasm");
const VIEWS = 4, FEAT = 1280;
const MODELS = path.resolve("../public/models");

let seed = 7;   // train.mjs와 같은 난수
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
function split(indices, ratio, keyOf) {
  const groups = new Map();
  for (const i of indices) { const k = keyOf ? keyOf(i) : "all"; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i); }
  const train = [], test = [];
  for (const g of groups.values()) { shuffle(g); const n = Math.max(1, Math.floor(g.length * ratio)); test.push(...g.slice(0, n)); train.push(...g.slice(n)); }
  return [train.sort((a, b) => a - b), test.sort((a, b) => a - b)];
}

const data = JSON.parse(fs.readFileSync("data/metadata.json", "utf8"));
const items = data.items.filter((it) => fs.existsSync(path.join("data", "views", `${it.id}_${VIEWS - 1}.jpg`)));
const artists = data.artists;
const labeled = items.map((it, i) => (it.label ? i : -1)).filter((i) => i >= 0);
const [, artTe] = split(labeled, 0.2, (i) => items[i].label);
const outside = items.map((it, i) => (it.label ? -1 : i)).filter((i) => i >= 0);
const featOf = (i) => { const b = fs.readFileSync(path.join("work", "feats", `${items[i].id}.f32`)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

const handler = (dir) => ({ load: async () => {
  const json = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const buf = Buffer.concat(json.weightsManifest.flatMap((g) => g.paths.map((p) => fs.readFileSync(path.join(dir, p)))));
  return { modelTopology: json.modelTopology, format: json.format, weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
    weightData: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
} });
const heads = await tf.loadLayersModel(handler(path.join(MODELS, "heads")));

function probsFor(indices, views) {
  const out = [];
  for (let s = 0; s < indices.length; s += 256) {
    const chunk = indices.slice(s, s + 256);
    const per = views.map((k) => {
      const x = new Float32Array(chunk.length * FEAT);
      chunk.forEach((i, r) => x.set(featOf(i).subarray(k * FEAT, (k + 1) * FEAT), r * FEAT));
      return x;
    });
    const p = tf.tidy(() => tf.stack(per.map((x) => heads.predict(tf.tensor2d(x, [chunk.length, FEAT]))[1])).mean(0).arraySync());
    out.push(...p);
  }
  return out;
}
const accuracy = (rows, idx) => {
  let t1 = 0, t3 = 0;
  rows.forEach((p, r) => {
    const y = artists.indexOf(items[idx[r]].label);
    const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
    t1 += order[0] === y; t3 += order.slice(0, 3).includes(y);
  });
  return [t1 / rows.length, t3 / rows.length];
};

const options = { "전체 한 장": [0], "전체+좌우 반전 평균": [0, 1] };
let chosen = null;
for (const [name, views] of Object.entries(options)) {
  const [top1, top3] = accuracy(probsFor(artTe, views), artTe);
  console.log(`${name}: top-1 ${top1.toFixed(3)}, top-3 ${top3.toFixed(3)}`);
  if (!chosen || top1 > chosen.top1 || (top1 === chosen.top1 && top3 > chosen.top3)) chosen = { name, views, top1, top3 };
}
console.log(`앱에서 쓸 보기: ${chosen.name}`);

const inP = probsFor(artTe, chosen.views);
const outP = probsFor(outside, chosen.views);
const inMax = inP.map((p) => Math.max(...p));
const outMax = outP.map((p) => Math.max(...p));
// 범위 안 그림 중 1위가 맞은 것만 "통과해야 할 것"으로 센다 — 틀린 답을 자신 있게 말하는 것도 막아야 하므로
const inCorrect = inP.map((p, r) => p.indexOf(Math.max(...p)) === artists.indexOf(items[artTe[r]].label));
let calib = { balanced: -1 };
for (let t = 0.2; t <= 0.95; t += 0.01) {
  const keepCorrect = inMax.filter((m, r) => inCorrect[r] && m >= t).length / Math.max(1, inCorrect.filter(Boolean).length);
  const outReject = outMax.filter((m) => m < t).length / outMax.length;
  const inKeep = inMax.filter((m) => m >= t).length / inMax.length;
  const balanced = (keepCorrect + outReject) / 2;
  if (balanced > calib.balanced) calib = { threshold: +t.toFixed(2), keepCorrect, inKeep, outReject, balanced };
}
const keptAcc = (() => {
  const kept = inMax.map((m, r) => (m >= calib.threshold ? r : -1)).filter((r) => r >= 0);
  return kept.length ? kept.filter((r) => inCorrect[r]).length / kept.length : 0;
})();
console.log(`'모름' 기준 ${calib.threshold}: 맞힌 범위 안 그림 ${(calib.keepCorrect * 100).toFixed(0)}% 통과, ` +
  `범위 밖 그림 ${(calib.outReject * 100).toFixed(0)}% 걸러냄, 통과한 그림의 1위 정답률 ${(keptAcc * 100).toFixed(0)}%`);

const meta = JSON.parse(fs.readFileSync(path.join(MODELS, "meta.json"), "utf8"));
const r3 = (x) => +x.toFixed(3);
meta.inference_views = chosen.views;   // 0: 영역 전체, 1: 좌우 반전 (app.js toCanvas와 같은 번호)
meta.metrics.artist_top1 = r3(chosen.top1);
meta.metrics.artist_top3 = r3(chosen.top3);
delete meta.metrics.artist_top1_tta;
delete meta.metrics.artist_top3_tta;
meta.artist_scope = { threshold: calib.threshold, in_keep: r3(calib.inKeep), correct_keep: r3(calib.keepCorrect),
  out_reject: r3(calib.outReject), kept_top1: r3(keptAcc), outside_images: outside.length };
fs.writeFileSync(path.join(MODELS, "meta.json"), JSON.stringify(meta, null, 1));
console.log("meta.json 갱신:", JSON.stringify(meta.artist_scope));
