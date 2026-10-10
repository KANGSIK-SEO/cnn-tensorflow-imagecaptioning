// 명화 진위 참고 검사 — 모든 추론을 이 기기 안(TensorFlow.js)에서 한다. 서버·API 키·인터넷 없음.
// 모델 (public/models, 처음 열 때 기기에 저장된다):
//   backbone/  MobileNetV2 CNN — 사진(224×224, 0~1) → 특징 1280
//   heads/     특징 → [원본 대조 임베딩 256, 작가 확률 15, 캡션 ctx 64, 캡션 h0 256]
//   decoder/   캡션 한 걸음 — [ctx, 원-핫 토큰, 상태] → [다음 토큰 점수, 새 상태]
// 판정·문장 만들기는 logic.js (테스트: tests/logic.test.js). 모든 안내는 한국어와 영어를 함께 보여 준다.
"use strict";

// ---------- 설정 ----------
const MODELS = "models";
const LOAD_BUDGET_MS = 90_000;   // 모델 내려받기 전체 시간 예산 (느린 모바일 망 고려, 저장된 뒤에는 금방 끝난다)
const FETCH_RETRIES = 1;         // 일시 장애(5xx·연결 끊김)만 한 번 더 — 시간 초과·4xx는 다시 해도 같다 (명화 추천 챗봇 PR #23)
const CACHE_NAME = "art-verify-v4";

const $ = (id) => document.getElementById(id);
const state = { meta: null, refs: null, refVecs: null, backbone: null, heads: null, decoder: null,
  ready: false, busy: false, pending: false, run: 0,
  img: null, rect: { x: 0, y: 0, w: 1, h: 1 } };   // 고른 사진과 분석할 영역(사진 크기에 대한 비율)
const DEFAULT_INSET = 0.06;   // 처음 상자는 가장자리를 6%씩 덜어 낸다 (액자·벽이 대개 가장자리에 있다)
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---------- 상태 메시지 (PR #39 공통 컴포넌트) ----------
const STATUS_KINDS = new Set(["info", "busy", "success", "warning", "error"]);
function setStatus(kind, ko, en, { retry = false } = {}) {
  if (!STATUS_KINDS.has(kind)) throw new TypeError(`unknown status kind: ${kind}`);
  const el = $("status");
  el.className = `status ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.querySelector(".ko").textContent = ko;
  el.querySelector(".en").textContent = en;
  $("retry").hidden = !retry;
}

function setProgress(fraction) {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  $("progress").hidden = pct >= 100;
  $("progress").setAttribute("aria-valuenow", String(pct));
  $("progress-fill").style.width = `${pct}%`;
}

// 단계별 소요 시간 기록 (PR #27 형식). 사진·결과 내용은 남기지 않는다.
function logStage(run, stage, started, ok = true) {
  console.info(`stage=${stage} run=${run} latency_ms=${Math.round(performance.now() - started)} ok=${ok}`);
}

// 화면이 그려질 틈을 준다. tf.nextFrame(requestAnimationFrame)은 탭이 가려지면 멈추므로 setTimeout을 쓴다.
const yieldToUi = () => new Promise((r) => setTimeout(r, 16));

// ---------- 내려받기: 일시 장애만 재시도, 전체 시간 예산 (PR #23) ----------
function makeFetch(deadline) {
  return async function fetchWithRetry(url, init) {
    for (let attempt = 0; ; attempt++) {
      const remaining = deadline - performance.now();
      if (remaining < 2000) throw Object.assign(new Error("timeout"), { code: "TIMEOUT" });
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), remaining);
      try {
        const res = await fetch(url, { ...init, signal: ctrl.signal });
        if (res.status >= 500 && attempt < FETCH_RETRIES) { await new Promise((r) => setTimeout(r, 500 * (attempt + 1))); continue; }
        if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { code: "HTTP" });
        return res;
      } catch (e) {
        if (e.name === "AbortError") throw Object.assign(new Error("timeout"), { code: "TIMEOUT" });
        if (e.code === "HTTP" || attempt >= FETCH_RETRIES) throw Object.assign(e, { code: e.code || "NETWORK" });
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      } finally {
        clearTimeout(timer);
      }
    }
  };
}

// 무엇이 잘못됐고 다음에 무엇을 하면 되는지 (PR #29)
const LOAD_ERRORS = {
  NETWORK: ["모델을 내려받지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요. 한 번 받은 뒤에는 인터넷 없이도 돼요.",
    "Couldn't download the models. Check your connection and try again — after the first download it works offline."],
  TIMEOUT: ["내려받기가 너무 오래 걸려요. 와이파이에서 다시 시도해 주세요.", "The download is taking too long. Try again on Wi-Fi."],
  HTTP: ["모델 파일을 찾지 못했어요. 잠시 뒤 새로고침해 주세요.", "Model files are unavailable. Reload in a moment."],
  INVALID: ["모델 파일이 서로 맞지 않아요. 새로고침해 주세요. 계속되면 알려 주세요.", "Model files don't match. Reload; if it persists, please report it."],
  UNKNOWN: ["앱을 준비하지 못했어요. 새로고침해 주세요.", "The app couldn't start. Please reload."],
};

// ---------- 시작 ----------
async function boot() {
  const run = ++state.run;
  const started = performance.now();
  setStatus("busy", "모델을 불러오는 중…", "Loading models…");
  setProgress(0);
  try {
    if (typeof tf === "undefined" || typeof tf.ready !== "function") throw Object.assign(new Error("tf missing"), { code: "UNKNOWN" });
    await tf.ready();
    if (tf.getBackend() !== "webgl") console.warn("backend", tf.getBackend());
    const deadline = performance.now() + LOAD_BUDGET_MS;
    const fetchFunc = makeFetch(deadline);
    const json = (path) => fetchFunc(`${MODELS}/${path}`).then((r) => r.json());
    const [meta, refs, refBuf] = await Promise.all([
      json("meta.json"), json("refs.json"), fetchFunc(`${MODELS}/refs.bin`).then((r) => r.arrayBuffer()),
    ]);
    // 진행률: 세 모델 가중치 크기에 비례 (백본 약 8.6MB, 헤드 2.9MB, 디코더 1.3MB)
    const parts = { backbone: 0, heads: 0, decoder: 0 };
    const weight = { backbone: 8.6, heads: 2.9, decoder: 1.3 };
    const total = 12.8;
    const onProgress = (name) => (f) => {
      parts[name] = f;
      setProgress(Object.keys(parts).reduce((s, k) => s + parts[k] * weight[k], 0) / total);
    };
    const [backbone, heads, decoder] = await Promise.all([
      tf.loadGraphModel(`${MODELS}/backbone/model.json`, { fetchFunc, onProgress: onProgress("backbone") }),
      tf.loadLayersModel(`${MODELS}/heads/model.json`, { fetchFunc, onProgress: onProgress("heads") }),
      tf.loadLayersModel(`${MODELS}/decoder/model.json`, { fetchFunc, onProgress: onProgress("decoder") }),
    ]);
    const problems = Logic.validateAssets({
      meta, refs, refBytes: refBuf.byteLength,
      headsOutputs: heads.outputs.map((o) => o.shape[1]), decoderOutputs: decoder.outputs.map((o) => o.shape[1]),
    });
    if (problems.length) throw Object.assign(new Error(problems.join("; ")), { code: "INVALID" });
    Object.assign(state, { meta, refs, refVecs: new Int8Array(refBuf), backbone, heads, decoder });
    setProgress(1);
    logStage(run, "load", started);

    // 첫 실행은 준비 작업으로 느리므로 빈 그림으로 한 번 돌려 둔다
    const warm = performance.now();
    tf.tidy(() => {
      heads.predict(backbone.execute(tf.zeros([1, meta.image_size, meta.image_size, 3])));
      decoder.predict([tf.zeros([1, meta.ctx_dim]), tf.zeros([1, 1, meta.vocab.length]), tf.zeros([1, meta.cap_units])]);
    });
    logStage(run, "warmup", warm);

    fillStaticTexts();
    state.ready = true;
    setStatus("success", "준비됐어요. 그림 사진을 넣어 주세요.", "Ready. Add a photo of a painting.");
    checkOffline();
    if (state.pending) { state.pending = false; analyze(); }
  } catch (e) {
    console.error(e);
    logStage(run, "load", started, false);
    const [ko, en] = LOAD_ERRORS[e.code] || LOAD_ERRORS.UNKNOWN;
    setStatus("error", ko, en, { retry: true });
  }
}

function fillStaticTexts() {
  const { meta, refs } = state;
  const select = $("claimed");
  if (select.options.length <= 1) {
    meta.artists.forEach((a, i) => select.add(new Option(`${meta.artist_ko[a]} · ${a}`, String(i))));
  }
  $("ref-count").textContent = refs.length.toLocaleString();
  const m = meta.metrics;
  const pct = (x) => `${Math.round(x * 100)}%`;
  $("metrics").textContent =
    `시험 성적 · Test accuracy — 작가 판별 1순위 ${pct(m.artist_top1)}, 3순위 안 ${pct(m.artist_top3)} ` +
    `(학습에 안 쓴 그림 ${m.artist_test_images}점 · top-1 / top-3 on ${m.artist_test_images} held-out paintings), ` +
    `캡션 시대·문화권·기법 ${pct(m.caption_slot_accuracy)} · caption era/culture/technique. ` +
    (meta.artist_scope ? `학습 범위 밖 판정 — 범위 밖 그림 ${pct(meta.artist_scope.out_reject)}를 걸러내고, 통과한 판정은 ${pct(meta.artist_scope.kept_top1)}가 정답 · ` +
      `"outside scope" catches ${pct(meta.artist_scope.out_reject)} of unfamiliar paintings; confident answers are ${pct(meta.artist_scope.kept_top1)} correct. ` : "") +
    `실행 장치 · Backend: ${tf.getBackend()}.`;
}

// ---------- 사진 → 입력 ----------
function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(Object.assign(new Error("decode"), { code: "DECODE" }));
    img.src = URL.createObjectURL(file);
  });
}

// 학습 때와 같게: 고른 영역을 정사각형으로 늘려 224×224, 0~1 RGB.
// 보기 번호 — 0: 영역 전체, 1: 좌우 반전, 2: 가운데 75% 확대. 어떤 보기를 평균할지는 시험 정확도로 정해
// meta.inference_views에 적혀 있다 (training/calibrate.mjs). 첫 보기(0)의 결과로 대조·캡션을 한다.
function toCanvas(img, rect, view) {
  const size = state.meta.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  let { x, y, w, h } = rect;
  if (view === 2) { x += w * 0.125; y += h * 0.125; w *= 0.75; h *= 0.75; }
  if (view === 1) { ctx.translate(size, 0); ctx.scale(-1, 1); }
  const W = img.naturalWidth, H = img.naturalHeight;
  ctx.drawImage(img, x * W, y * H, w * W, h * H, 0, 0, size, size);
  return canvas;
}
function toInputs(img, rect) {
  const views = state.meta.inference_views || [0];
  const canvases = views.map((v) => toCanvas(img, rect, v));
  $("analyzed-img").src = canvases[0].toDataURL("image/jpeg", 0.8);
  return tf.tidy(() => tf.stack(canvases.map((c) => tf.browser.fromPixels(c))).toFloat().div(255));
}

// ---------- 분석 ----------
// 사진을 고르면 먼저 분석할 영역을 고르게 한다
async function choosePhoto(file) {
  const check = Logic.checkFile(file);
  if (!check.ok) { setStatus("warning", check.ko, check.en); return; }
  try {
    const img = await loadImage(file);
    state.img = img;
    $("crop-img").src = img.src;
    state.rect = { x: DEFAULT_INSET, y: DEFAULT_INSET, w: 1 - 2 * DEFAULT_INSET, h: 1 - 2 * DEFAULT_INSET };
    $("crop").hidden = false;
    $("result").hidden = true;
    drawCropBox();
    setStatus(state.ready ? "info" : "busy",
      "네모를 그림에 맞춘 뒤 '이 영역 분석'을 눌러 주세요.", "Fit the box to the painting, then tap 'Analyze this area'.");
    $("crop").scrollIntoView({ behavior: "auto", block: "start" });
  } catch {
    setStatus("error", "이 사진을 열지 못했어요. JPG나 PNG 사진으로 다시 골라 주세요.", "Couldn't open this photo. Try a JPG or PNG.");
  }
}

async function analyze() {
  if (!state.img) return;
  if (!state.ready) {
    state.pending = true;   // 준비가 끝나면 바로 분석한다 — 버튼을 눌렀는데 반응이 없는 일이 없게
    setStatus("busy", "모델 준비가 끝나면 바로 분석할게요…", "Will analyze as soon as the models are ready…");
    return;
  }
  if (state.busy) return;
  state.busy = true;
  $("analyze-btn").disabled = true;
  const run = ++state.run;
  const started = performance.now();
  $("skeleton").hidden = false;
  $("result").hidden = true;
  try {
    let t = performance.now();
    setStatus("busy", "분석하는 중…", "Analyzing…");
    await yieldToUi();
    const input = toInputs(state.img, state.rect);
    logStage(run, "preprocess", t);

    t = performance.now();
    const outs = tf.tidy(() => {
      const [pca, artistProbs, capCtx, capH0] = state.heads.predict(state.backbone.execute(input));
      const first = (x) => x.slice([0, 0], [1, -1]);
      return [l2(first(pca)), artistProbs.mean(0, true), first(capCtx), first(capH0)];   // 작가 확률만 보기들의 평균
    });
    const [emb, probs, ctx, h0] = outs.map((x) => x.dataSync());   // 작은 결과라 동기로 (캡션 단계와 같은 이유)
    tf.dispose([...outs, input]);
    logStage(run, "cnn", t);

    t = performance.now();
    const words = generateCaption(ctx, h0);
    logStage(run, "caption", t);

    const matchKind = showMatches(emb);
    const scope = Logic.scopeVerdict(probs, state.meta.artist_scope, matchKind);
    showScope(scope);
    showArtists(probs, scope);
    showCaption(words);
    const now = new Date();
    const when = $("analyzed-at");
    when.dateTime = now.toISOString();
    when.textContent = `분석 시각 · Analyzed ${now.toLocaleString("ko-KR", { hour: "2-digit", minute: "2-digit" })}`;
    $("result").hidden = false;
    const ms = Math.round(performance.now() - started);
    setStatus("success", `분석 완료 (${ms}ms, 이 기기 안에서 처리)`, `Done in ${ms} ms, on this device`);
    logStage(run, "total", started);
    $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (e) {
    console.error(e);
    logStage(run, "total", started, false);
    setStatus("error", "분석하지 못했어요. 다른 사진으로 해 보거나 새로고침해 주세요.", "Analysis failed. Try another photo or reload.");
  } finally {
    $("skeleton").hidden = true;
    $("analyze-btn").disabled = false;
    state.busy = false;
  }
}

const l2 = (x) => x.div(x.norm("euclidean", -1, true));

// ---------- 그림 영역 상자 (끌어서 옮기기·모서리로 크기 바꾸기·키보드 화살표) ----------
function drawCropBox() {
  const { x, y, w, h } = state.rect;
  const box = $("crop-box");
  box.style.left = `${x * 100}%`;
  box.style.top = `${y * 100}%`;
  box.style.width = `${w * 100}%`;
  box.style.height = `${h * 100}%`;
}

function moveRect(r, handle, dx, dy) {
  const MIN = 0.1;
  let { x, y, w, h } = r;
  if (handle === "move") {
    x = clamp(x + dx, 0, 1 - w);
    y = clamp(y + dy, 0, 1 - h);
  } else {
    if (handle.includes("w")) { const nx = clamp(x + dx, 0, x + w - MIN); w += x - nx; x = nx; }
    if (handle.includes("e")) w = clamp(w + dx, MIN, 1 - x);
    if (handle.includes("n")) { const ny = clamp(y + dy, 0, y + h - MIN); h += y - ny; y = ny; }
    if (handle.includes("s")) h = clamp(h + dy, MIN, 1 - y);
  }
  return { x, y, w, h };
}

(function initCrop() {
  const stage = $("crop-stage");
  const box = $("crop-box");
  box.tabIndex = 0;
  box.setAttribute("aria-description", "화살표: 옮기기, Shift+화살표: 크기 · Arrows move, Shift+arrows resize");
  let drag = null;
  box.addEventListener("pointerdown", (e) => {
    drag = { handle: e.target.dataset.h || "move", sx: e.clientX, sy: e.clientY, r: { ...state.rect } };
    box.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  box.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = (e.clientX - drag.sx) / stage.clientWidth;
    const dy = (e.clientY - drag.sy) / stage.clientHeight;
    state.rect = moveRect(drag.r, drag.handle, dx, dy);
    drawCropBox();
  });
  const end = () => { drag = null; };
  box.addEventListener("pointerup", end);
  box.addEventListener("pointercancel", end);
  box.addEventListener("keydown", (e) => {
    const step = 0.02;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (!d) return;
    e.preventDefault();
    state.rect = moveRect(state.rect, e.shiftKey ? "se" : "move", d[0], d[1]);
    drawCropBox();
  });
})();

// ---------- 1. 미술관 소장품 대조 ----------
function setVerdict(el, kind, ko, en) {
  el.className = `verdict ${kind}`;
  el.replaceChildren(document.createTextNode(ko));
  const span = document.createElement("span");
  span.className = "en";
  span.lang = "en";
  span.textContent = en;
  el.append(span);
}

function showMatches(emb) {
  const { emb_dim: dim, match_thresholds: th } = state.meta;
  const vecs = state.refVecs;
  const n = vecs.length / dim;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) s += emb[j] * vecs[off + j];
    scores[i] = s / 127;
  }
  const top = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, 5);
  const ref = state.refs[top[0]];
  const best = scores[top[0]].toFixed(2);
  const who = ref.artist || "작가 미상 · Unknown";
  const verdict = Logic.matchVerdict(top.map((i) => scores[i]), th);
  const el = $("match-verdict");
  if (verdict === "same") {
    setVerdict(el, "bad",
      `미술관 소장품 「${ref.title}」(${who})과 같은 작품의 사진으로 보입니다 (유사도 ${best}). 원작은 미술관에 있으므로, '진품'으로 팔린다면 복제품일 가능성이 큽니다.`,
      `Looks like the museum work "${ref.title}" (${who}), similarity ${best}. The original is in a museum, so if this is sold as an original it is likely a reproduction.`);
  } else if (verdict === "similar") {
    setVerdict(el, "maybe",
      `「${ref.title}」과 비슷합니다 (유사도 ${best}). 같은 작품인지 직접 비교해 보세요.`,
      `Similar to "${ref.title}" (similarity ${best}). Compare them yourself.`);
  } else {
    setVerdict(el, "good",
      `등록된 소장품 ${n.toLocaleString()}점 중 같은 작품은 찾지 못했습니다. 가장 비슷한 그림들:`,
      `No identical work among ${n.toLocaleString()} museum pieces. Most similar:`);
  }
  $("matches").replaceChildren(...top.map((i) => {
    const r = state.refs[i];
    const li = document.createElement("li");
    const thumb = document.createElement("img");
    thumb.decoding = "async";
    thumb.alt = r.title;
    thumb.onload = () => thumb.classList.add("loaded");
    thumb.onerror = () => { thumb.style.visibility = "hidden"; };   // 오프라인에서 아직 안 본 썸네일
    thumb.src = `thumbs/${r.id}.webp`;
    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement(r.url ? "a" : "span");
    title.className = "title";
    title.textContent = r.title;
    if (r.url) { title.href = r.url; title.target = "_blank"; title.rel = "noopener"; }
    const by = document.createElement("span");
    by.className = "by";
    by.textContent = `${r.artist || "작가 미상 · Unknown"}${r.date ? ` · ${r.date}` : ""} · 유사도 similarity ${scores[i].toFixed(2)}`;
    meta.append(title, by);
    li.append(thumb, meta);
    return li;
  }));
  return verdict;
}

// ---------- 학습 범위 밖 ("모름"을 먼저 말한다) ----------
function showScope(scope) {
  const el = $("scope-verdict");
  el.hidden = !scope.outside;
  if (!scope.outside) return;
  const pct = Math.round(scope.maxProb * 100);
  el.replaceChildren(document.createTextNode(
    `학습 범위 밖의 그림으로 보입니다. 학습한 작가 15명 누구와도 뚜렷하게 닮지 않았고(최고 ${pct}%), 등록된 소장품과도 같은 작품이 없습니다. 아래 작가 확률은 참고만 하세요.`));
  const en = document.createElement("span");
  en.className = "en";
  en.lang = "en";
  en.textContent = `This painting seems outside what the model has learned: no clear match with the 15 artists (top ${pct}%) and no identical museum work. Treat the artist scores below as hints only.`;
  el.append(en);
}

// ---------- 2. 화풍으로 본 작가 ----------
function showArtists(probs, scope) {
  const { artists, artist_ko: ko } = state.meta;
  const claimed = $("claimed").value === "" ? null : Number($("claimed").value);
  const v = Logic.artistVerdict(probs, claimed);
  const shown = v.order.slice(0, 3);
  if (claimed !== null && !shown.includes(claimed)) shown.push(claimed);
  const el = $("artist-verdict");
  const nameKo = (i) => ko[artists[i]];
  const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
  $("artists").classList.toggle("unsure", scope.outside);
  if (scope.outside && claimed !== null) {
    setVerdict(el, "maybe", `학습 범위 밖의 그림이라 ${nameKo(claimed)} 화풍인지 판단하기 어렵습니다 (${pct(v.p, 1)}, ${v.rank}위). 전문가 감정을 권합니다.`,
      `Outside the model's scope, so it can't judge whether this is ${artists[claimed]}'s style (${pct(v.p, 1)}, ranked ${v.rank}). Get a professional opinion.`);
  } else if (scope.outside) {
    setVerdict(el, "", `15명 중 가장 가까운 화풍은 ${nameKo(v.order[0])}이지만 확률이 낮아 의미가 크지 않습니다.`,
      `The nearest of the 15 styles is ${artists[v.order[0]]}, but the score is too low to mean much.`);
  } else if (v.kind === "none") {
    setVerdict(el, "", `학습한 작가 15명 중에서는 ${nameKo(v.order[0])}의 화풍에 가장 가깝습니다. 목록에 없는 작가의 그림이라도 15명 중 하나로 답하므로 참고만 하세요.`,
      `Closest style among the 15 artists: ${artists[v.order[0]]}. Paintings by other artists are still mapped to one of the 15 — use as a hint only.`);
  } else if (v.kind === "good") {
    setVerdict(el, "good", `화풍이 ${nameKo(claimed)}와(과) 잘 맞습니다 (${pct(v.p)}, 15명 중 1위).`,
      `The style matches ${artists[claimed]} well (${pct(v.p)}, ranked 1st of 15).`);
  } else if (v.kind === "maybe") {
    setVerdict(el, "maybe", `화풍이 ${nameKo(claimed)}와(과) 어느 정도 맞습니다 (${pct(v.p)}, ${v.rank}위). 더 확인이 필요합니다.`,
      `Partly consistent with ${artists[claimed]} (${pct(v.p)}, ranked ${v.rank}). Needs more checking.`);
  } else {
    setVerdict(el, "bad", `화풍이 ${nameKo(claimed)}와(과) 잘 맞지 않습니다 (${pct(v.p, 1)}, 15명 중 ${v.rank}위). 전문가 감정을 권합니다.`,
      `The style doesn't match ${artists[claimed]} (${pct(v.p, 1)}, ranked ${v.rank} of 15). Get a professional opinion.`);
  }
  $("artists").replaceChildren(...shown.map((i) => {
    const li = document.createElement("li");
    if (i === claimed) li.className = "claimed";
    const row = document.createElement("div");
    row.className = "row";
    const name = document.createElement("span");
    name.textContent = nameKo(i);
    const en = document.createElement("span");
    en.className = "en";
    en.lang = "en";
    en.textContent = `${artists[i]}${i === claimed ? " · 판매자 주장 claimed" : ""}`;
    name.append(en);
    const val = document.createElement("span");
    val.textContent = pct(probs[i], 1);
    row.append(name, val);
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${Math.max(1, probs[i] * 100)}%`;
    track.append(fill);
    li.append(row, track);
    li.setAttribute("aria-label", `${nameKo(i)} ${artists[i]} ${pct(probs[i], 1)}`);
    return li;
  }));
}

// ---------- 3. 이미지 캡셔닝 (GRU 디코더를 한 걸음씩) ----------
function generateCaption(ctxArr, h0Arr) {
  const { vocab, ctx_dim: ctxDim, cap_units: units, max_len: maxLen } = state.meta;
  const START = vocab.indexOf("<start>");
  const END = vocab.indexOf("<end>");
  const ctx = tf.tensor2d(ctxArr, [1, ctxDim]);
  let st = tf.tensor2d(h0Arr, [1, units]);
  let token = START;
  const words = [];
  for (let step = 0; step < maxLen; step++) {
    const tokenT = tf.tidy(() => tf.oneHot(tf.tensor2d([token], [1, 1], "int32"), vocab.length).toFloat());   // 원-핫 [1, 1, V]
    const [logits, next] = state.decoder.predict([ctx, tokenT, st]);
    // 숫자 하나만 읽으므로 동기로 읽는다. 비동기(data())는 GPU 완료를 타이머로 확인하는데, 화면이 가려지거나
    // 절전 중이면 타이머가 1초씩 밀려 9단어에 7초가 걸렸다 (동기로는 0.1초대, 2026-10-11 측정)
    token = tf.tidy(() => logits.argMax(1).dataSync()[0]);
    tf.dispose([tokenT, logits, st]);
    st = next;
    if (token === END) break;
    words.push(vocab[token]);
  }
  tf.dispose([ctx, st]);
  return words;
}

function showCaption(words) {
  const s = Logic.captionSentence(words);
  $("caption").textContent = s.ko;
  $("caption-en").textContent = s.en;
  $("caption-tokens").textContent = `모델이 만든 토큰 · Model tokens: ${words.join(" ")}`;
}

// ---------- 인터넷 없이 쓰기 ----------
// 서비스 워커가 처음 열 때 앱과 모델 파일 전부(models/files.json)를 기기에 저장한다.
async function checkOffline() {
  const el = $("offline-state");
  if (!("serviceWorker" in navigator) || !("caches" in window)) {
    el.textContent = "이 브라우저는 오프라인 저장을 지원하지 않아요 · Offline mode not supported";
    return;
  }
  try {
    await navigator.serviceWorker.ready;
    const files = await fetch(`${MODELS}/files.json`).then((r) => r.json());
    const cache = await caches.open(CACHE_NAME);
    let missing = 0;
    for (const f of ["index.html", "app.js", "logic.js", "style.css", "vendor/tf.min.js", ...files]) if (!(await cache.match(f))) missing++;
    el.textContent = missing
      ? `오프라인 준비 중… (${missing}개 남음) · Preparing offline (${missing} left)`
      : "✓ 인터넷 없이 쓸 수 있어요 · Works offline";
    if (missing) setTimeout(checkOffline, 3000);
  } catch {
    el.textContent = "오프라인 준비 상태를 확인하지 못했어요 · Couldn't check offline status";
  }
}

// ---------- 테마 (PR #107: 시스템 설정을 따르다가 고르면 기억) ----------
$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = isDark ? "light" : "dark";
  try { localStorage.setItem("av-theme", root.dataset.theme); } catch (_) { /* 저장 못 해도 이번 화면은 바뀐다 */ }
});

// ---------- 이벤트 ----------
$("photo").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) choosePhoto(file);
});
$("analyze-btn").addEventListener("click", () => analyze());
$("whole-btn").addEventListener("click", () => { state.rect = { x: 0, y: 0, w: 1, h: 1 }; drawCropBox(); analyze(); });
$("claimed").addEventListener("change", () => { if (state.img && !$("result").hidden) analyze(); });
$("retry").addEventListener("click", () => { $("retry").hidden = true; boot(); });

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
boot();
