// tf.min.js 안의 regenerator-runtime은 전역 regeneratorRuntime이 없으면 Function("r", "...")로 만들려 한다.
// 이 사이트의 CSP는 문자열 코드 실행(eval)을 막으므로 그 순간 TensorFlow.js 전체가 멈춘다 (2026-10-11 실제 장애).
// 전역 자리를 미리 만들어 두면 tf.min.js가 평범한 대입으로 끝내서 eval이 필요 없다 — CSP를 약하게 풀지 않는 방법.
// tf.min.js보다 먼저 불러야 한다 (index.html).
window.regeneratorRuntime = window.regeneratorRuntime || undefined;
