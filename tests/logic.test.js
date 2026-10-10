// 판정·문장 만들기(public/logic.js) 시험 — node --test tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../public/logic.js");

const TH = { same_work_p10: 0.573, same_work_median: 0.802, different_work_p99: 0.93, different_work_median: 0.587 };

test("캡션: 한국어·영어 문장을 함께 만들고 받침에 맞는 조사를 붙인다", () => {
  const s = L.captionSentence(["19세기", "유럽", "유화", "<sep>", "풍경", "나무"]);
  assert.equal(s.ko, "19세기 유럽 유화 작품으로, 풍경·나무를 그린 것으로 보입니다.");
  assert.equal(s.en, "Looks like a 19th-century European oil painting depicting a landscape and trees.");
  assert.equal(L.captionSentence(["17세기", "유럽", "유화", "<sep>", "초상", "남성", "말"]).ko.endsWith("말을 그린 것으로 보입니다."), true);
});

test("캡션: 반복 단어·'기타'·주제 없음도 자연스럽게", () => {
  assert.equal(L.captionSentence(["19세기", "유럽", "유화", "<sep>", "풍경", "구름", "풍경"]).ko,
    "19세기 유럽 유화 작품으로, 풍경·구름을 그린 것으로 보입니다.");
  assert.equal(L.captionSentence(["기타", "회화"]).ko, "회화 작품으로 보입니다.");
  assert.equal(L.captionSentence([]).en, "Looks like a painting.");
  assert.equal(L.toEn("21세기"), "21st-century");
  assert.equal(L.toEn("11세기"), "11th-century");
});

test("소장품 대조: 같은 작품·비슷함·없음을 1등 점수와 2등과의 차이로 가른다", () => {
  assert.equal(L.matchVerdict([0.93, 0.60, 0.55], TH), "same");
  assert.equal(L.matchVerdict([0.72, 0.60], TH), "similar");
  assert.equal(L.matchVerdict([0.56, 0.50], TH), "none");
});

test("소장품 대조: 연작처럼 1·2등이 비슷하면 같은 작품이라 단정하지 않는다", () => {
  assert.equal(L.matchVerdict([0.90, 0.89], TH), "similar");
  assert.notEqual(L.matchVerdict([0.85, 0.83], TH), "same");
});

test("작가 판정: 판매자 주장 작가의 순위와 확률로 나눈다", () => {
  const probs = [0.6, 0.2, 0.1, 0.05, 0.05];
  assert.equal(L.artistVerdict(probs, 0).kind, "good");
  assert.equal(L.artistVerdict(probs, 2).kind, "maybe");
  assert.equal(L.artistVerdict(probs, 4).kind, "bad");
  assert.equal(L.artistVerdict(probs, null).kind, "none");
  assert.equal(L.artistVerdict([0.4, 0.35, 0.25], 0).kind, "maybe");   // 1위라도 50% 미만이면 확정하지 않는다
});

test("사진 확인: 막아야 할 것과 통과시켜야 할 것", () => {
  assert.equal(L.checkFile(null).ok, false);
  assert.equal(L.checkFile({ type: "application/pdf", size: 10 }).ok, false);
  assert.equal(L.checkFile({ type: "image/jpeg", size: 31 * 1024 * 1024 }).ok, false);
  assert.equal(L.checkFile({ type: "image/heic", size: 5_000_000 }).ok, true);
  assert.equal(L.checkFile({ type: "", size: 5_000_000 }).ok, true);   // 일부 폰은 형식을 비워 보낸다
});

test("모델 점검: 맞지 않는 곳을 한 번에 모두 알린다", () => {
  const meta = { artists: Array(15).fill("a"), vocab: Array(62).fill("w"), emb_dim: 256 };
  const refs = Array(10).fill({});
  assert.deepEqual(L.validateAssets({ meta, refs, refBytes: 2560, headsOutputs: [256, 15, 64, 256], decoderOutputs: [62, 256] }), []);
  assert.equal(L.validateAssets({ meta, refs, refBytes: 100, headsOutputs: [256, 14], decoderOutputs: [60] }).length, 3);
});

test("학습 범위 밖: 확률이 낮고 소장품에도 없으면 '모름'", () => {
  const scope = { threshold: 0.6 };
  assert.equal(L.scopeVerdict([0.538, 0.2, 0.262], scope, "none").outside, true);   // 브론치노 → 루벤스 53.8% 같은 경우
  assert.equal(L.scopeVerdict([0.9, 0.05, 0.05], scope, "none").outside, false);
  assert.equal(L.scopeVerdict([0.3, 0.3, 0.4], scope, "same").outside, false);       // 소장품과 같은 작품이면 아는 그림
  assert.equal(L.scopeVerdict([0.3, 0.3, 0.4], undefined, "none").threshold, 0.5);   // 기준값이 없으면 0.5
});

test("캡션 단어장의 모든 단어에 영어가 있다 (한영 병기가 빠지지 않게)", () => {
  const { vocab } = require("../public/models/meta.json");
  const missing = vocab.filter((w) => !w.startsWith("<") && /[가-힣]/.test(L.toEn(w)));
  assert.deepEqual(missing, []);
});
