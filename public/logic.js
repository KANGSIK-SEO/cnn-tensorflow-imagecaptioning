// 화면과 상관없는 판단·문장 만들기 (브라우저와 테스트(node --test) 양쪽에서 쓴다).
// 모든 사용자 문장은 한국어와 영어를 함께 만든다 (한영 병기).
"use strict";

(function (root) {
  // ---------- 캡션 단어 → 영어 ----------
  const EN = {
    "<sep>": "|",
    유럽: "European", 미국: "American", 중국: "Chinese", 일본: "Japanese", 한국: "Korean", 인도: "Indian",
    페르시아: "Persian", 기타: "",
    이탈리아: "Italian", 플랑드르: "Flemish", 네덜란드: "Dutch", 스페인: "Spanish", 프랑스: "French", 영국: "British",
    독일어권: "German-speaking",
    유화: "oil painting", 템페라: "tempera painting", 과슈: "gouache painting", 수채: "watercolor",
    파스텔: "pastel", 수묵: "ink painting", 수묵채색: "ink-and-color painting", 프레스코: "fresco", 회화: "painting",
    강: "river", 개: "dog", 건물: "buildings", 과일: "fruit", 구름: "clouds", 군인: "soldiers", 꽃: "flowers",
    나무: "trees", 남성: "a man", 누드: "a nude", 동물: "animals", 말: "horses", 물: "water", 배: "boats",
    불교: "Buddhist themes", 산: "mountains", 새: "birds", 성모: "the Virgin Mary", 성인: "saints", 소: "cattle",
    실내: "an interior", 아이: "children", 여성: "a woman", 예수: "Christ", 음악: "music", 정물: "a still life",
    죽음: "death", 천사: "angels", 초상: "a portrait", 풍경: "a landscape", 하늘: "the sky",
  };
  const centuryEn = (w) => {
    const n = parseInt(w, 10);
    const suffix = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] || "th";
    return `${n}${suffix}-century`;
  };
  const toEn = (w) => (/^\d+세기$/.test(w) ? centuryEn(w) : (EN[w] ?? w));

  // 받침이 있으면 "을", 없으면 "를"
  function objectParticle(word) {
    const code = word.charCodeAt(word.length - 1) - 0xac00;
    return code >= 0 && code <= 11171 && code % 28 !== 0 ? "을" : "를";
  }

  function englishList(items) {
    if (items.length <= 1) return items.join("");
    return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
  }

  // 모델이 만든 토큰열 → { ko, en } 문장. 예: ["19세기","유럽","유화","<sep>","풍경","나무"]
  function captionSentence(words) {
    const clean = [];
    for (const w of words) if (typeof w === "string" && w && !clean.includes(w)) clean.push(w);   // 반복 제거
    const sep = clean.indexOf("<sep>");
    const kind = (sep === -1 ? clean : clean.slice(0, sep)).filter((w) => w !== "기타");
    const subjects = sep === -1 ? [] : clean.slice(sep + 1);
    const kindKo = kind.join(" ") || "회화";
    const kindEn = kind.map(toEn).filter(Boolean).join(" ") || "painting";
    if (!subjects.length) {
      return { ko: `${kindKo} 작품으로 보입니다.`, en: `Looks like a ${kindEn}.` };
    }
    const last = subjects[subjects.length - 1];
    return {
      ko: `${kindKo} 작품으로, ${subjects.join("·")}${objectParticle(last)} 그린 것으로 보입니다.`,
      en: `Looks like a ${kindEn} depicting ${englishList(subjects.map(toEn))}.`,
    };
  }

  // ---------- 1. 소장품 대조 판정 ----------
  // 기준값(meta.match_thresholds)은 학습 데이터에서 잰 것: 같은 작품을 뒤집거나 잘라 찍은 사진의 중앙값은 약 0.80,
  // 서로 다른 작품끼리 가장 비슷한 쌍의 중앙값은 약 0.59. 1등과 2등의 차이(margin)도 함께 본다 —
  // 연작(예: 모네 「건초더미」)처럼 비슷한 그림이 여럿이면 1등만 높아서는 같은 작품이라 단정하지 않는다.
  function matchVerdict(scores, thresholds) {
    const [best = -1, second = -1] = scores;
    const margin = best - second;
    if (best >= thresholds.same_work_median && margin >= 0.03) return "same";
    // 1등이 아주 높은데 2등도 붙어 있으면(연작) "비슷함"으로 — 같은 작품이라 단정하지는 않는다
    if (best >= Math.max(0.65, thresholds.different_work_median + 0.05) &&
        (margin >= 0.05 || best >= thresholds.same_work_median)) return "similar";
    return "none";
  }

  // ---------- 2. 작가 판정 ----------
  function artistVerdict(probs, claimed) {
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
    if (claimed === null || claimed === undefined) return { kind: "none", order };
    const rank = order.indexOf(claimed) + 1;
    const p = probs[claimed];
    if (rank === 1 && p >= 0.5) return { kind: "good", order, rank, p };
    if (rank <= 3) return { kind: "maybe", order, rank, p };
    return { kind: "bad", order, rank, p };
  }

  // ---------- 학습 범위 밖 ("모름") ----------
  // scope.threshold는 학습 때 15명의 시험 그림과 15명 밖 그림을 가장 잘 가르는 최고 확률 값(meta.artist_scope).
  // 소장품과 같거나 비슷한 작품을 찾았으면 범위 밖이라 하지 않는다 — 그 그림은 모델이 알고 있다.
  function scopeVerdict(probs, scope, matchKind) {
    const maxProb = Math.max(...probs);
    const threshold = scope && typeof scope.threshold === "number" ? scope.threshold : 0.5;
    const outside = maxProb < threshold && matchKind !== "same" && matchKind !== "similar";
    return { outside, maxProb, threshold };
  }

  // ---------- 입력 사진 확인 ----------
  const MAX_FILE_MB = 30;
  function checkFile(file) {
    if (!file) return { ok: false, ko: "사진을 고르지 않았어요.", en: "No photo selected." };
    if (file.type && !file.type.startsWith("image/")) {
      return { ok: false, ko: "그림 사진(JPG·PNG 등)을 골라 주세요.", en: "Please choose a photo (JPG, PNG, …)." };
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      return { ok: false, ko: `사진이 너무 커요 (${MAX_FILE_MB}MB 이하로 골라 주세요).`, en: `The photo is too large (max ${MAX_FILE_MB} MB).` };
    }
    return { ok: true };
  }

  // ---------- 불러온 모델·데이터 점검 (한 번에 모든 문제를 모아 알린다) ----------
  function validateAssets({ meta, refs, refBytes, headsOutputs, decoderOutputs }) {
    const problems = [];
    if (!meta || !Array.isArray(meta.artists) || !Array.isArray(meta.vocab)) problems.push("meta.json 형식");
    else {
      if (headsOutputs && headsOutputs[1] !== meta.artists.length) problems.push(`작가 수 ${meta.artists.length} ≠ 모델 출력 ${headsOutputs[1]}`);
      if (decoderOutputs && decoderOutputs[0] !== meta.vocab.length) problems.push(`단어 수 ${meta.vocab.length} ≠ 디코더 출력 ${decoderOutputs[0]}`);
      if (Array.isArray(refs) && refBytes !== refs.length * meta.emb_dim) problems.push(`소장품 ${refs.length}점 ≠ refs.bin 크기`);
    }
    if (!Array.isArray(refs)) problems.push("refs.json 형식");
    return problems;
  }

  const api = { captionSentence, objectParticle, matchVerdict, artistVerdict, scopeVerdict, checkFile, validateAssets, toEn, MAX_FILE_MB };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Logic = api;
})(typeof self !== "undefined" ? self : globalThis);
