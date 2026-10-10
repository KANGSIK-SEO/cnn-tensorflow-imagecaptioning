// 인터넷 없이 쓰기: 처음 열 때 앱과 모델 파일 전부를 기기에 저장하고, 그다음부터는 저장본으로 돈다.
// 모델 파일 목록은 학습 스크립트가 만드는 models/files.json에서 읽는다. 모델을 바꾸면 CACHE 이름을 올린다.
const CACHE = "art-verify-v1";
const SHELL = ["./", "index.html", "style.css", "app.js", "manifest.json", "icon.svg", "vendor/tf.min.js",
  "models/files.json"];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(SHELL);
    const files = await (await cache.match("models/files.json")).json();
    await cache.addAll(files);
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// 저장본을 먼저 쓰고, 없으면 내려받아 저장한다 (썸네일은 볼 때 저장)
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const hit = await caches.match(e.request, { ignoreSearch: true });
    if (hit) return hit;
    const res = await fetch(e.request);
    if (res.ok) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
    }
    return res;
  })());
});
