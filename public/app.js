// 명화 진위 참고 검사 — 모든 추론을 이 기기 안(TensorFlow.js)에서 한다. 서버·API 키·인터넷 없음.
// 모델 (public/models, 처음 열 때 기기에 저장된다):
//   backbone/  MobileNetV2 CNN — 사진(224×224, 0~1) → 특징 1280
//   heads/     특징 → [원본 대조 임베딩 256, 작가 확률 15, 캡션 ctx 64, 캡션 h0 256]
//   decoder/   캡션 한 걸음 — [ctx, 토큰, 상태] → [다음 토큰 점수, 새 상태]
"use strict";

const $ = (id) => document.getElementById(id);
const M = {};   // 불러온 모델과 데이터

function setStatus(text) { $("status").textContent = text; }

async function boot() {
  try {
    await tf.ready();
    const [meta, refs, refsBin] = await Promise.all([
      fetch("models/meta.json").then((r) => r.json()),
      fetch("models/refs.json").then((r) => r.json()),
      fetch("models/refs.bin").then((r) => r.arrayBuffer()),
    ]);
    Object.assign(M, { meta, refs, refVecs: new Int8Array(refsBin) });
    [M.backbone, M.heads, M.decoder] = await Promise.all([
      tf.loadGraphModel("models/backbone/model.json"),
      tf.loadLayersModel("models/heads/model.json"),
      tf.loadLayersModel("models/decoder/model.json"),
    ]);
    // 첫 실행은 준비 작업으로 느리므로 빈 그림으로 한 번 돌려 둔다
    tf.tidy(() => M.heads.predict(M.backbone.execute(tf.zeros([1, meta.image_size, meta.image_size, 3]))));

    const select = $("claimed");
    meta.artists.forEach((a, i) => select.add(new Option(`${meta.artist_ko[a]} (${a})`, String(i))));
    $("ref-count").textContent = `${refs.length.toLocaleString()}`;
    const m = meta.metrics;
    $("metrics").textContent =
      `시험 성적 — 작가 판별 1순위 정답 ${(m.artist_top1 * 100).toFixed(0)}%, 3순위 안 ${(m.artist_top3 * 100).toFixed(0)}% ` +
      `(학습에 안 쓴 그림 ${m.artist_test_images}점), 캡션의 시대·문화권·기법 정답 ${(m.caption_slot_accuracy * 100).toFixed(0)}%. ` +
      `실행 장치: ${tf.getBackend()}.`;
    setStatus("준비됐어요. 그림 사진을 넣어 주세요.");
    checkOffline();
  } catch (e) {
    console.error(e);
    setStatus("모델을 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침해 주세요.");
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

// 학습 때와 같게: 그림 전체를 정사각형으로 늘려 224×224, 0~1 RGB
function toInput(img) {
  const size = M.meta.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, size, size);
  return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0));
}

async function analyze(file) {
  if (!M.heads) return;
  const img = await loadImage(file);
  $("preview").src = img.src;
  $("preview-wrap").hidden = false;
  setStatus("분석하는 중…");
  await tf.nextFrame();

  const started = performance.now();
  const input = toInput(img);
  const outs = tf.tidy(() => {
    const [pca, artistProbs, capCtx, capH0] = M.heads.predict(M.backbone.execute(input));
    return [pca.div(pca.norm("euclidean", -1, true)), artistProbs, capCtx, capH0];   // 임베딩은 단위 벡터로
  });
  const [emb, probs, ctx, h0] = await Promise.all(outs.map((t) => t.data()));
  tf.dispose([...outs, input]);

  const caption = await generateCaption(ctx, h0);
  showMatches(emb);
  showArtists(probs);
  showCaption(caption);
  $("result").hidden = false;
  setStatus(`분석 완료 (${Math.round(performance.now() - started)}ms, 이 기기 안에서 처리)`);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- 1. 미술관 소장품 대조 ----------
function showMatches(emb) {
  const dim = M.meta.emb_dim;
  const vecs = M.refVecs;
  const n = vecs.length / dim;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) s += emb[j] * vecs[off + j];
    scores[i] = s / 127;
  }
  const top = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, 5);
  const best = scores[top[0]];
  const th = M.meta.match_thresholds;
  const verdict = $("match-verdict");
  const ref = M.refs[top[0]];
  if (best >= th.same_work_p10 && best > th.different_work_p99) {
    verdict.className = "verdict bad";
    verdict.textContent = `미술관 소장품 「${ref.title}」(${ref.artist || "작가 미상"})과 같은 작품의 사진으로 보입니다 ` +
      `(유사도 ${best.toFixed(2)}). 이 원작은 미술관에 있으므로, '진품'으로 팔린다면 복제품일 가능성이 큽니다.`;
  } else if (best > th.different_work_p99) {
    verdict.className = "verdict warn";
    verdict.textContent = `「${ref.title}」과 매우 비슷합니다 (유사도 ${best.toFixed(2)}). 같은 작품인지 직접 비교해 보세요.`;
  } else {
    verdict.className = "verdict ok";
    verdict.textContent = `등록된 소장품 ${n.toLocaleString()}점 중 같은 작품은 찾지 못했습니다. 가장 비슷한 그림들:`;
  }
  $("matches").replaceChildren(...top.map((i) => {
    const r = M.refs[i];
    const li = document.createElement("li");
    const thumb = document.createElement("img");
    thumb.src = `thumbs/${r.id}.webp`;
    thumb.onerror = () => { thumb.style.visibility = "hidden"; };   // 오프라인에서 아직 안 본 썸네일
    thumb.alt = r.title;
    thumb.loading = "lazy";
    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement(r.url ? "a" : "span");
    title.className = "title";
    title.textContent = r.title;
    if (r.url) { title.href = r.url; title.target = "_blank"; title.rel = "noopener"; }
    const by = document.createElement("span");
    by.className = "by";
    by.textContent = `${r.artist || "작가 미상"}${r.date ? ` · ${r.date}` : ""} · 유사도 ${scores[i].toFixed(2)}`;
    meta.append(title, document.createElement("br"), by);
    li.append(thumb, meta);
    return li;
  }));
}

// ---------- 2. 화풍으로 본 작가 ----------
function showArtists(probs) {
  const { artists, artist_ko: ko } = M.meta;
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const claimed = $("claimed").value === "" ? null : Number($("claimed").value);
  const shown = order.slice(0, 3);
  if (claimed !== null && !shown.includes(claimed)) shown.push(claimed);

  const verdict = $("artist-verdict");
  if (claimed === null) {
    verdict.className = "verdict";
    verdict.textContent = `학습한 작가 15명 중에서는 ${ko[artists[order[0]]]}의 화풍에 가장 가깝습니다. ` +
      "목록에 없는 작가의 그림이라도 15명 중 하나로 답하므로 참고만 하세요.";
  } else {
    const rank = order.indexOf(claimed) + 1;
    const p = probs[claimed];
    const name = ko[artists[claimed]];
    if (rank === 1 && p >= 0.5) {
      verdict.className = "verdict ok";
      verdict.textContent = `화풍이 ${name}와(과) 잘 맞습니다 (${(p * 100).toFixed(0)}%, 15명 중 1위).`;
    } else if (rank <= 3) {
      verdict.className = "verdict warn";
      verdict.textContent = `화풍이 ${name}와(과) 어느 정도 맞습니다 (${(p * 100).toFixed(0)}%, ${rank}위). 더 확인이 필요합니다.`;
    } else {
      verdict.className = "verdict bad";
      verdict.textContent = `화풍이 ${name}와(과) 잘 맞지 않습니다 (${(p * 100).toFixed(1)}%, 15명 중 ${rank}위). 전문가 감정을 권합니다.`;
    }
  }
  $("artists").replaceChildren(...shown.map((i) => {
    const li = document.createElement("li");
    if (i === claimed) li.className = "claimed";
    li.innerHTML = '<div class="row"><span></span><span></span></div><div class="track"><div class="fill"></div></div>';
    li.querySelector(".row span:first-child").textContent = `${ko[artists[i]]}${i === claimed ? " (판매자 주장)" : ""}`;
    li.querySelector(".row span:last-child").textContent = `${(probs[i] * 100).toFixed(1)}%`;
    li.querySelector(".fill").style.width = `${Math.max(1, probs[i] * 100)}%`;
    return li;
  }));
}

// ---------- 3. 이미지 캡셔닝 (GRU 디코더를 한 걸음씩) ----------
async function generateCaption(ctxArr, h0Arr) {
  const { vocab, ctx_dim: ctxDim, cap_units: units, max_len: maxLen } = M.meta;
  const START = vocab.indexOf("<start>");
  const END = vocab.indexOf("<end>");
  const ctx = tf.tensor2d(ctxArr, [1, ctxDim]);
  let state = tf.tensor2d(h0Arr, [1, units]);
  let token = START;
  const words = [];
  for (let step = 0; step < maxLen; step++) {
    const tokenT = tf.tidy(() => tf.oneHot(tf.tensor2d([token], [1, 1], "int32"), vocab.length).toFloat());   // 원-핫 [1, 1, V]
    const [logits, next] = M.decoder.predict([ctx, tokenT, state]);
    token = (await logits.argMax(1).data())[0];
    tf.dispose([tokenT, logits, state]);
    state = next;
    if (token === END) break;
    if (words.length && words[words.length - 1] === vocab[token]) continue;   // 같은 말 반복 막기
    words.push(vocab[token]);
  }
  ctx.dispose();
  state.dispose();
  return words;
}

function objectParticle(word) {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  return code >= 0 && code <= 11171 && code % 28 !== 0 ? "을" : "를";
}

function showCaption(words) {
  const sep = words.indexOf("<sep>");
  const kind = (sep === -1 ? words : words.slice(0, sep)).filter((w) => w !== "기타");
  const subjects = sep === -1 ? [] : words.slice(sep + 1);
  let sentence = `${kind.join(" ") || "회화"} 작품으로 보입니다.`;
  if (subjects.length) {
    const last = subjects[subjects.length - 1];
    sentence = `${kind.join(" ") || "회화"} 작품으로, ${subjects.join("·")}${objectParticle(last)} 그린 것으로 보입니다.`;
  }
  $("caption").textContent = sentence;
  $("caption-tokens").textContent = `모델이 만든 토큰: ${words.join(" ")}`;
}

// ---------- 이벤트 ----------
$("photo").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) analyze(file).catch((err) => { console.error(err); setStatus("이 사진을 분석하지 못했어요. 다른 사진으로 해 보세요."); });
});
$("claimed").addEventListener("change", () => {
  const file = $("photo").files && $("photo").files[0];
  if (file && !$("result").hidden) analyze(file);
});

// ---------- 인터넷 없이 쓰기 ----------
// 서비스 워커가 처음 열 때 앱과 모델 파일 전부(models/files.json)를 기기에 저장한다.
async function checkOffline() {
  const el = $("offline-state");
  if (!("serviceWorker" in navigator) || !("caches" in window)) { el.textContent = "이 브라우저는 오프라인 저장을 지원하지 않아요"; return; }
  try {
    await navigator.serviceWorker.ready;
    const files = await fetch("models/files.json").then((r) => r.json());
    const cache = await caches.open("art-verify-v1");
    const missing = [];
    for (const f of ["index.html", "app.js", "style.css", "vendor/tf.min.js", ...files]) if (!(await cache.match(f))) missing.push(f);
    el.textContent = missing.length ? `오프라인 준비 중… (${missing.length}개 남음)` : "✓ 인터넷 없이 쓸 수 있어요";
    if (missing.length) setTimeout(checkOffline, 3000);
  } catch { el.textContent = "오프라인 준비 상태를 확인하지 못했어요"; }
}

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
boot();
