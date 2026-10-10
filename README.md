# cnn-tensorflow-imagecaptioning — 명화 진위 참고 검사 · Art Authenticity Check

**https://cnn-tensorflow-imagecaptioning.vercel.app**

> On-device art authenticity checker: CNN style attribution, museum-collection matching and image captioning run entirely in the phone's browser with TensorFlow.js — no API key, no server, works offline. 화면의 모든 안내는 한국어와 영어를 함께 보여 준다.

그림 사진을 넣으면 **스마트폰 안에서** CNN이 화풍과 미술관 소장품을 대조하고, 이미지 캡셔닝이 그림을 한국어로 설명하는 웹앱.
**API 키·서버·인터넷이 필요 없다** — 처음 한 번 열 때 앱과 모델(약 12MB)을 폰에 저장하고, 그다음부터는 비행기 모드에서도 동작한다.

> 결과는 **참고용**이며 감정서가 아니다. 실제 거래에서는 전문 감정기관에 맡긴다.

## 무엇을 하나

| 기능 | 기술 | 무엇을 알려 주나 |
|---|---|---|
| 1. 미술관 소장품 대조 | CNN 임베딩(MobileNetV2 특징 → PCA 256차원) + 코사인 유사도 | 사진이 미술관 소장품(3,505점)과 같은 작품인지. 원작이 미술관에 있는 그림이 "진품"으로 팔린다면 복제품일 가능성이 크다 |
| 2. 화풍으로 본 작가 | CNN 전이학습 분류 (작가 15명) | 판매자가 말한 작가의 화풍과 얼마나 맞는지 (확률·순위) |
| 3. 그림 설명 | 이미지 캡셔닝 (CNN 특징 → GRU 디코더) | 시대·문화권·기법·주제 (예: "19세기 유럽 유화 작품으로, 풍경·나무를 그린 것으로 보입니다") |

## 구조

```
사진 ─▶ MobileNetV2 (ImageNet 사전학습, 고정) ─▶ 특징 1280 ─┬─▶ PCA 256 → 단위 벡터 → 소장품 3,505점과 유사도
        public/models/backbone                               ├─▶ Dense(15, softmax) → 작가 확률
                                                             └─▶ cap_h0, cap_ctx ─▶ GRU 디코더 (한 단어씩) → 캡션
                                                                 public/models/heads      public/models/decoder
```

- **기기 안 추론**: TensorFlow.js(`public/vendor/tf.min.js`, 앱 안에 포함). 서비스 워커(`public/sw.js`)가 처음 열 때
  `public/models/files.json`에 적힌 모델 파일을 모두 저장한다. 홈 화면에 추가하면 앱처럼 쓸 수 있다(PWA).
- **사진은 어디로도 보내지 않는다** — 분석은 전부 브라우저 안에서 한다.

## 정확도 (학습에 쓰지 않은 그림으로 시험)

`public/models/meta.json`의 `metrics`에 있다. 앱 화면 아래에도 표시된다.

| 항목 | 결과 |
|---|---|
| 작가 판별 1순위 정답 (15명, 시험 그림 94점) | 61.7% |
| 작가 판별 3순위 안 정답 | 84.0% |
| 캡션 시대·문화권·기법 칸 정답 | 72.0% |
| 원본 대조: 같은 작품을 좌우 반전한 사진이 원본을 1등으로 찾음 (`verify.mjs`) | 8장 중 8장 |

한계: 학습한 15명 밖의 작가 그림도 15명 중 하나로 답한다. 사람이 정교하게 그린 위작을 가려내는 모델은 아니다.

## 학습 데이터

[AITOOLLEARN-7-1](https://github.com/KANGSIK-SEO/AITOOLLEARN-7-1)의 작품 DB(`data/art.db`)에서 **퍼블릭 도메인(CC0) 회화**만 썼다
(메트로폴리탄·시카고·클리블랜드 미술관). 작가 15명 506점 + 캡션용 회화 3,000점 = 3,505점.
캡션 정답은 미술관 메타데이터(연도·문화권·기법·주제)를 한국어 단어로 바꿔 만든 토큰열이다 (`training/prepare_data.py`).

## 다시 학습하기

```bash
# 1) 그림 내려받기 + 학습용 보기(전체·좌우 반전·잘라 보기 2개) + 썸네일 — Python, Pillow·requests 필요
python training/prepare_data.py --art-db ../AITOOLLEARN-7-1/data/art.db --captions 3000

# 2) 학습 + 내보내기 — Node.js, TensorFlow.js (WebAssembly 백엔드라 네이티브 DLL이 필요 없다)
cd training && npm install && npm run train

# 3) 내보낸 모델을 앱과 같은 방식으로 다시 불러와 확인
node verify.mjs
```

모델이나 앱을 바꾸면 `public/sw.js`의 `CACHE`와 `public/app.js`의 `CACHE_NAME`을 함께 올려야 폰에 저장된 예전 파일이 바뀐다.

```bash
# 판정·문장 만들기 시험 (public/logic.js)
node --test tests/*.test.js
```

## 팀 PR에서 가져온 것 (명화 추천 챗봇 AITOOLLEARN-7-1)

| PR | 가져온 것 |
|---|---|
| 유영민 #31 | 스크린리더 접근성: `role`·`aria-live`·`aria-label`, 결과 영역 알림 |
| 유영민 #32·#107 | 시스템 테마를 따르고 버튼으로 바꾸면 기억 (`theme-init.js`, `light-dark()` 토큰, theme-color 두 개) |
| 유영민 #33·#107 | 대기 표시(점 세 개)·분석 중 뼈대(shimmer)·움직임 줄이기 설정 존중 |
| 유영민 #34 | 360px 이하 작은 화면 레이아웃, 44px 이상 누르기 영역 |
| 유영민 #36·#160 | 서비스 워커: 페이지는 네트워크 먼저→저장본→`offline.html`, 모델·앱 파일은 저장본 먼저. manifest `id`·`orientation`·`categories` |
| 유영민 #38 | 결과에 분석 시각 `<time datetime>` |
| 유영민 #39 | 상태 메시지 공통 컴포넌트 (`setStatus(kind, ko, en)`: info·busy·success·warning·error) |
| 유영민 #40 | SEO·공유 미리보기 메타(Open Graph·Twitter, canonical, PNG 아이콘) |
| 유영민 #107 | 갤러리 톤 디자인 토큰(따뜻한 종이색 + 테라코타 강조), 썸네일 lazy·async·페이드 |
| 오철호 #21 | 시작할 때 모델·데이터가 서로 맞는지 한 번에 점검 (`Logic.validateAssets`) |
| 오철호 #23 | 내려받기: 일시 장애만 재시도, 전체 시간 예산(90초) |
| 오철호 #26 | 입력 사진 점검 (형식·30MB 상한), 이상한 입력에도 멈추지 않기 |
| 오철호 #27 | 단계별 소요 시간 로그 `stage=cnn run=3 latency_ms=91 ok=true` (사진 내용은 남기지 않음) |
| 오철호 #29 | 오류 안내에 "다음에 할 일"과 다시 시도 버튼 |
| 오철호 #20·#24 | 막아야 할 것과 통과시켜야 할 것을 함께 시험 (`tests/logic.test.js`, GitHub Actions `test`) |

## 장애 기록

- **2026-10-11 첫 배포 직후 사진에 반응 없음**: tf.min.js 안의 regenerator-runtime이 `Function(...)`(eval)을 써서 CSP(`script-src 'self'`)에 막혔고,
  TensorFlow.js가 아예 뜨지 않았다. CSP를 풀지 않고 `public/vendor/tf-shim.js`로 전역 자리를 미리 만들어 해결.
  모델 준비 전에 사진을 넣으면 준비 뒤 바로 분석하도록, 실패하면 이유와 다시 시도 버튼을 보이도록 고쳤다.
- **캡션이 7~9초 걸림**: 단어마다 GPU 결과를 비동기로 기다렸는데, 화면이 가려지면 타이머가 1초씩 밀렸다. 숫자 하나만 동기로 읽어 0.1초대로.

## 배포

정적 사이트라 빌드가 없다. Vercel에서 이 저장소를 가져오면 `vercel.json`대로 `public/`을 그대로 올린다.

## 라이선스

- 그림 이미지·메타데이터: 각 미술관의 CC0 퍼블릭 도메인
- MobileNetV2 가중치: Google, Apache 2.0 (TensorFlow Hub `imagenet/mobilenet_v2_100_224/feature_vector/3`)
- TensorFlow.js: Apache 2.0
