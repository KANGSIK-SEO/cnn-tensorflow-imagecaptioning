// 인터넷 없이 쓰기 — 명화 추천 챗봇 PR #36의 나눔을 따른다.
//   페이지(navigate): 네트워크 먼저 → 실패하면 저장본 → 그것도 없으면 offline.html
//   그 밖의 같은 사이트 파일(앱·모델 가중치·썸네일): 저장본 먼저 → 없으면 내려받아 저장
// 처음 설치할 때 앱과 모델 파일 전부(models/files.json)를 미리 저장한다. 모델·앱을 바꾸면 CACHE 이름을 올린다
// (app.js의 CACHE_NAME도 같이).
const CACHE = "art-verify-v3";
const OFFLINE_URL = "offline.html";
const SHELL = ["./", "index.html", OFFLINE_URL, "style.css", "theme-init.js", "logic.js", "app.js", "manifest.json",
  "icon.svg", "icon-192.png", "icon-512.png", "vendor/tf-shim.js", "vendor/tf.min.js", "models/files.json"];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // cache: "reload" — 브라우저 HTTP 저장본을 건너뛰고 서버에서 새로 받는다. 안 그러면 새 앱 코드에 예전 모델 파일이
    // 섞여 들어갈 수 있다 (2026-10-11 시험 중 실제로 섞였다)
    const fresh = (urls) => urls.map((u) => new Request(u, { cache: "reload" }));
    await cache.addAll(fresh(SHELL));
    const files = await (await cache.match("models/files.json")).json();
    await cache.addAll(fresh(files));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const { request } = e;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;

  if (request.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const res = await fetch(request);
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(request, copy));
        }
        return res;
      } catch {
        return (await caches.match(request, { ignoreSearch: true })) || (await caches.match("index.html")) ||
          (await caches.match(OFFLINE_URL));
      }
    })());
    return;
  }

  e.respondWith((async () => {
    const hit = await caches.match(request, { ignoreSearch: true });
    if (hit) return hit;
    const res = await fetch(request);
    if (res.ok) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(request, copy));
    }
    return res;
  })());
});
