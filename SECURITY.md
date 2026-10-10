# 보안 정책 (Security Policy)

## 취약점 신고
보안 문제를 발견하면 **공개 이슈로 올리지 말고** GitHub의 비공개 신고 기능을 써 주세요.
저장소 → **Security** 탭 → **Report a vulnerability** (Private vulnerability reporting).
3일 안에 받았다는 답을 드리고, 14일 안에 처리 계획을 알려 드립니다.

Please report vulnerabilities privately via GitHub **Security → Report a vulnerability**. We acknowledge within 3 days.

## 구조상 지켜지는 것
- **서버·로그인·DB·API 키가 없다**: 정적 파일만 배포하고, 분석은 전부 사용자 기기 안(TensorFlow.js)에서 한다.
  가로챌 비밀 값도, 공격받을 서버 API도, 비용을 노린 호출 공격 대상도 없다.
- **사진은 기기 밖으로 나가지 않는다**: 고른 사진은 브라우저 메모리에서만 쓰고 어디로도 보내지 않는다.
- **외부 스크립트 없음**: TensorFlow.js도 저장소 안 파일(`public/vendor/tf.min.js`)을 쓴다.

## 우리가 하는 것 (명화 추천 챗봇 [AITOOLLEARN-7-1](https://github.com/KANGSIK-SEO/AITOOLLEARN-7-1)과 같은 방식)
- 모든 응답에 보안 헤더 (`vercel.json`): HSTS, CSP(우리 사이트의 스크립트만 실행, 외부 전송 금지), X-Frame-Options DENY,
  nosniff, Referrer-Policy, Permissions-Policy, Cross-Origin-Opener-Policy
- 의존성: Dependabot 매주 점검, 코드 정적 분석: CodeQL, 공개 보안 점수: OpenSSF Scorecard
