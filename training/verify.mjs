// 내보낸 모델을 앱(public/app.js)과 같은 방식으로 다시 불러와 그림 몇 장을 추론해 본다.
// 저장·불러오기가 깨지지 않았는지, 원본 대조가 자기 자신을 1등으로 찾는지 확인한다.
// 사용: cd training && node verify.mjs
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import jpeg from "jpeg-js";
import fs from "node:fs";
import path from "node:path";

setWasmPaths(path.resolve("node_modules/@tensorflow/tfjs-backend-wasm/dist") + path.sep);
await tf.setBackend("wasm");
const MODELS = path.resolve("../public/models");

function handler(dir) {
  return { load: async () => {
    const json = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
    const data = Buffer.concat(json.weightsManifest.flatMap((g) => g.paths.map((p) => fs.readFileSync(path.join(dir, p)))));
    return { modelTopology: json.modelTopology, format: json.format, weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
      weightData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) };
  } };
}
const meta = JSON.parse(fs.readFileSync(path.join(MODELS, "meta.json"), "utf8"));
const refs = JSON.parse(fs.readFileSync(path.join(MODELS, "refs.json"), "utf8"));
const refVecs = new Int8Array(fs.readFileSync(path.join(MODELS, "refs.bin")).buffer.slice(0));
const backbone = await tf.loadGraphModel(handler(path.join(MODELS, "backbone")));
const heads = await tf.loadLayersModel(handler(path.join(MODELS, "heads")));
const decoder = await tf.loadLayersModel(handler(path.join(MODELS, "decoder")));
const { vocab } = meta;

let selfTop1 = 0;
const picks = [0, 1, 2, 50, 400, 1000, 2000, 3000].filter((i) => i < refs.length);
for (const i of picks) {
  const { data } = jpeg.decode(fs.readFileSync(path.join("data", "views", `${refs[i].id}_1.jpg`)), { useTArray: true, formatAsRGBA: false });
  const x = tf.tensor4d(data, [1, 224, 224, 3], "int32").toFloat().div(255);   // 좌우 반전 보기로 시험
  const [pca, probs, ctx, h0] = heads.predict(backbone.execute(x));
  const emb = await pca.div(pca.norm("euclidean", -1, true)).data();
  let best = -1, bestS = -2;
  for (let r = 0; r < refs.length; r++) {
    let s = 0; for (let j = 0; j < meta.emb_dim; j++) s += emb[j] * refVecs[r * meta.emb_dim + j];
    if (s > bestS) { bestS = s; best = r; }
  }
  selfTop1 += best === i;
  const p = await probs.data();
  const topArtist = meta.artists[[...p.keys()].sort((a, b) => p[b] - p[a])[0]];
  let t = vocab.indexOf("<start>"), state = h0;
  const words = [];
  for (let s = 0; s < meta.max_len; s++) {
    const [lg, ns] = decoder.predict([ctx, tf.oneHot(tf.tensor2d([[t]], [1, 1], "int32"), vocab.length).toFloat(), state]);
    state = ns; t = (await lg.argMax(1).data())[0];
    if (vocab[t] === "<end>") break;
    words.push(vocab[t]);
  }
  console.log(`#${i} ${refs[i].title} (${refs[i].artist}) → 대조 1등 #${best} 유사도 ${(bestS / 127).toFixed(2)}, ` +
    `화풍 ${topArtist}, 캡션 "${words.join(" ")}"`);
}
console.log(`원본 대조: 좌우 반전 사진 ${picks.length}장 중 ${selfTop1}장이 자기 원본을 1등으로 찾음`);
