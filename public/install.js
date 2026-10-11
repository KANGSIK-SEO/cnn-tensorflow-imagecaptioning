// "앱으로 설치하기" 버튼 — 기종·브라우저마다 다른 설치 방법을 한 버튼으로.
//   안드로이드 Chrome·삼성 인터넷·Edge: 브라우저의 설치 창(beforeinstallprompt)을 바로 띄운다
//   아이폰·아이패드: 애플이 자동 설치를 막아 두어 "공유 → 홈 화면에 추가"를 안내한다
//   중국폰 기본 브라우저(화웨이·샤오미·오포·비보): 메뉴 이름(添加到桌面)과 바로가기 권한까지 안내한다
//   카카오톡·인스타그램·위챗 등 앱 안 브라우저: 설치가 안 되므로 바깥 브라우저로 여는 길부터 보여 준다
// 기종 판단은 logic.js의 installPlatform (tests/logic.test.js에 실제 폰 User-Agent로 시험).
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const HERE = location.origin + location.pathname;
  const bar = $("install-bar");
  const btn = $("install-btn");
  const dialog = $("install-dialog");
  const body = $("install-body");
  let deferredPrompt = null;

  const standalone = () => matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const platform = Logic.installPlatform(navigator.userAgent, { maxTouchPoints: navigator.maxTouchPoints || 0 });

  // 작은 DOM 도우미 — 글자는 늘 textContent로 넣는다
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    Object.assign(node, props);
    for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return node;
  }
  const en = (text) => el("span", { className: "en", lang: "en", textContent: text });
  const key = (text) => el("span", { className: "key", textContent: text });
  // 단계: [한국어 조각들, 영어 문장]
  const steps = (...items) => el("ol", {}, ...items.map(([ko, english]) => el("li", {}, ...[].concat(ko), en(english))));

  function copyRow() {
    const input = el("input", { type: "text", value: HERE, readOnly: true, ariaLabel: "앱 주소 · App address" });
    const copy = el("button", { type: "button", className: "btn", textContent: "주소 복사 · Copy" });
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(HERE); } catch { input.select(); document.execCommand("copy"); }
      copy.textContent = "복사됨 ✓ · Copied";
    });
    return el("div", { className: "copy-row" }, input, copy);
  }

  function markInstalled() {
    btn.disabled = true;
    btn.replaceChildren(el("span", { ariaHidden: "true", textContent: "✓" }),
      el("span", {}, "설치됐어요 — 홈 화면에서 여세요", en("Installed — open it from your home screen")));
  }

  // ---------- 기종별 안내 ----------
  function guide() {
    const p = platform;
    if (p.kind === "inapp") {
      const parts = [el("p", {}, "지금은 앱 안의 브라우저라 설치할 수 없어요. 먼저 Chrome이나 Safari로 열어 주세요.",
        en("In-app browsers can't install apps. Open this page in Chrome or Safari first."))];
      if (p.opener === "kakao") {
        parts.push(el("a", { className: "btn primary", href: `kakaotalk://web/openExternal?url=${encodeURIComponent(HERE)}` },
          "기본 브라우저로 열기 · Open in browser"));
      } else if (p.opener === "android-intent") {
        parts.push(el("a", { className: "btn primary",
          href: `intent://${location.host}${location.pathname}#Intent;scheme=https;package=com.android.chrome;end` },
        "Chrome으로 열기 · Open in Chrome"));
      }
      parts.push(steps(
        [["오른쪽 위(또는 아래) ", key("⋯"), " 메뉴를 누르세요."], "Tap the ⋯ menu (top or bottom right)."],
        [["\"다른 브라우저로 열기\", \"Safari로 열기\", \"외부 브라우저\" 중 보이는 것을 누르세요."], "Choose \"Open in browser\" / \"Open in Safari\"."],
        [["열린 화면에서 위쪽 ", key("📲 앱으로 설치하기"), "를 다시 누르세요."], "Then tap \"Install the app\" again."]));
      parts.push(el("p", { className: "hint" }, "메뉴가 없으면 주소를 복사해 Chrome·Safari에 붙여 넣으세요. · Or copy the address into your browser."), copyRow());
      return parts;
    }
    if (p.kind === "wechat") {
      return [el("p", {}, "위챗 안에서는 설치할 수 없어요. 브라우저로 열어 주세요.", en("WeChat can't install apps. Open it in a browser.")),
        steps([["오른쪽 위 ", key("⋯"), "를 누르세요."], "Tap ⋯ at the top right."],
          [["\"在浏览器打开\" (브라우저로 열기)를 누르세요."], "Choose \"在浏览器打开\" (Open in browser)."],
          [["열린 화면에서 ", key("📲 앱으로 설치하기"), "를 다시 누르세요."], "Then tap \"Install the app\" again."]),
        copyRow()];
    }
    if (p.kind === "ios") {
      const parts = [el("p", {}, "아이폰·아이패드는 버튼 한 번으로 설치할 수 없게 되어 있어요. 아래 3단계면 돼요.",
        en("iPhone and iPad don't allow one-tap installs — it takes 3 quick steps."))];
      if (p.browser !== "safari") {
        parts.push(el("p", { className: "hint" }, "Chrome 등에서 안 보이면 Safari로 열어 주세요 (주소 복사 → Safari에 붙여 넣기).",
          en("If you don't see the option, open this page in Safari.")), copyRow());
      }
      parts.push(steps(
        [["화면 아래(아이패드는 위)의 공유 버튼 ", key("□↑"), "을 누르세요."], "Tap the Share button (square with an arrow)."],
        [["목록을 위로 올려 ", key("홈 화면에 추가"), "를 누르세요."], "Scroll down and tap \"Add to Home Screen\"."],
        [["오른쪽 위 ", key("추가"), "를 누르면 끝이에요."], "Tap \"Add\". Done."]));
      return parts;
    }
    if (p.kind === "android") {
      const menu = {
        chrome: [["오른쪽 위 ", key("⋮"), " → ", key("앱 설치"), " 또는 ", key("홈 화면에 추가")], "Tap ⋮ → \"Install app\" or \"Add to Home screen\"."],
        samsung: [["아래 ", key("≡"), " → ", key("현재 페이지 추가"), " → ", key("홈 화면")], "Tap ≡ → \"Add page to\" → \"Home screen\"."],
        huawei: [["메뉴 ", key("≡"), " 또는 ", key("⋮"), " → ", key("添加到桌面"), " (바탕화면에 추가)"], "Menu → \"添加到桌面\" (Add to desktop)."],
        xiaomi: [["아래 메뉴 ", key("≡"), " → ", key("添加到桌面"), " (바탕화면에 추가)"], "Menu → \"添加到桌面\" (Add to desktop)."],
        oppo: [["메뉴 ", key("≡"), " → ", key("添加到桌面"), " (바탕화면에 추가)"], "Menu → \"添加到桌面\" (Add to desktop)."],
        vivo: [["메뉴 ", key("≡"), " → ", key("添加到桌面"), " (바탕화면에 추가)"], "Menu → \"添加到桌面\" (Add to desktop)."],
        "china-other": [["메뉴 ", key("≡"), " → ", key("添加到桌面"), " 또는 ", key("发送到桌面")], "Menu → \"添加到桌面\" (Add to desktop)."],
      }[p.vendor] || [["브라우저 메뉴 → ", key("홈 화면에 추가")], "Browser menu → \"Add to Home screen\"."];
      const parts = [steps(menu, [["확인 창에서 ", key("추가"), " 또는 ", key("설치"), "를 누르세요."], "Confirm with \"Add\" or \"Install\"."])];
      if (["huawei", "xiaomi", "oppo", "vivo", "china-other"].includes(p.vendor)) {
        parts.push(el("p", { className: "hint" },
          "아이콘이 안 생기면: 설정 → 앱 → 이 브라우저 → 권한 → \"바탕화면 바로가기(桌面快捷方式)\"를 허용한 뒤 다시 해 주세요. 그래도 안 되면 Chrome에서 열어 설치하세요.",
          en("No icon? Settings → Apps → this browser → Permissions → allow \"Home screen shortcuts (桌面快捷方式)\", then try again. Or install from Chrome.")), copyRow());
      }
      return parts;
    }
    return [steps(
      [["주소창 오른쪽의 설치 아이콘 ", key("⊕"), "을 누르거나, 메뉴 ", key("⋮"), " → ", key("앱 설치"), "를 누르세요."],
        "Click the install icon in the address bar, or menu ⋮ → \"Install app\"."],
      [["휴대폰에 설치하려면 폰에서 이 주소를 여세요."], "To install on your phone, open this address there."]), copyRow()];
  }

  function openGuide() {
    body.replaceChildren(...guide());
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");   // 아주 오래된 브라우저
  }

  // ---------- 동작 ----------
  if (standalone()) { bar.hidden = true; return; }   // 이미 설치한 앱으로 열었으면 버튼이 필요 없다

  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();   // 브라우저가 멋대로 띄우지 않게 잡아 두었다가 버튼을 누를 때 띄운다
    deferredPrompt = e;
  });
  window.addEventListener("appinstalled", () => { deferredPrompt = null; markInstalled(); });

  btn.addEventListener("click", async () => {
    if (deferredPrompt) {
      const prompt = deferredPrompt;
      deferredPrompt = null;   // 한 번만 쓸 수 있다
      prompt.prompt();
      const { outcome } = await prompt.userChoice;
      if (outcome === "accepted") markInstalled();
      return;
    }
    openGuide();
  });
})();
