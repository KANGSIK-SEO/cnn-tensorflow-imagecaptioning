// 저장해 둔 테마(밝게/어둡게)를 화면이 그려지기 전에 적용한다 — 깜빡임 방지.
// CSP가 인라인 스크립트를 막으므로 별도 파일로 두고 <head>에서 defer 없이 부른다 (명화 추천 챗봇 PR #107과 같은 방식).
try {
  const t = localStorage.getItem("av-theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch (_) { /* 비공개 창 등 저장소를 못 쓰면 시스템 설정을 따른다 */ }
