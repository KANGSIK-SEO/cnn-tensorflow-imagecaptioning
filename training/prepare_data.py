"""학습 데이터 만들기 — 미술관 퍼블릭 도메인(CC0) 그림을 내려받고, 작가 라벨과 한국어 캡션(정답 문장)을 붙인다.

원본: AITOOLLEARN-7-1 저장소의 data/art.db (MET·시카고 미술관·클리블랜드 미술관 CC0 작품 메타데이터)
결과:
  training/data/images/<id>.jpg   — 긴 변 256px로 줄인 그림
  training/data/views/<id>_<k>.jpg — 224×224 학습용 보기 4개 (전체·좌우 반전·잘라 보기 2개)
  training/data/metadata.json     — id, 작가 라벨(15명 안이면), 캡션 토큰, 원본 링크
  public/thumbs/<id>.webp          — 앱 결과 화면용 160px 썸네일 (앱이 직접 갖고 있어 인터넷 없이도 보인다)

캡션은 사람이 쓴 문장이 아니라 미술관 메타데이터(시대·문화권·기법·주제)에서 규칙으로 만든 토큰열이다.
모델은 그림만 보고 이 토큰열을 맞히도록 배운다 (image captioning).

사용: python training/prepare_data.py --art-db ../AITOOLLEARN-7-1/data/art.db --captions 3000
"""
import argparse
import io
import json
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
IMAGES = OUT / "images"
VIEWS = OUT / "views"
THUMBS = ROOT.parent / "public" / "thumbs"
MAX_SIDE = 256
VIEW_SIZE = 224   # MobileNetV2 입력 크기
UA = "cnn-tensorflow-imagecaptioning/1.0 (research; public-domain images)"
# 시카고 미술관 IIIF는 AIC-User-Agent 헤더가 없으면 403을 준다 (누가 쓰는지 밝히라는 규칙)
HEADERS = {"User-Agent": UA,
           "AIC-User-Agent": "cnn-tensorflow-imagecaptioning (https://github.com/KANGSIK-SEO/cnn-tensorflow-imagecaptioning)"}

# CNN 작가 판별에 쓰는 15명 — art.db에서 유화가 25점 이상인 작가 중 화풍이 다양하게 골랐다
ARTISTS = [
    "Claude Monet", "Vincent van Gogh", "Paul Gauguin", "Camille Pissarro", "Edgar Degas",
    "Gustave Courbet", "Camille Corot", "John Singer Sargent", "Winslow Homer", "George Inness",
    "Gilbert Stuart", "Peter Paul Rubens", "Giovanni Battista Tiepolo", "Francesco Guardi", "Chen Hongshou",
]
ARTIST_KO = {
    "Claude Monet": "클로드 모네", "Vincent van Gogh": "빈센트 반 고흐", "Paul Gauguin": "폴 고갱",
    "Camille Pissarro": "카미유 피사로", "Edgar Degas": "에드가 드가", "Gustave Courbet": "귀스타브 쿠르베",
    "Camille Corot": "카미유 코로", "John Singer Sargent": "존 싱어 사전트", "Winslow Homer": "윈슬로 호머",
    "George Inness": "조지 이네스", "Gilbert Stuart": "길버트 스튜어트", "Peter Paul Rubens": "페테르 파울 루벤스",
    "Giovanni Battista Tiepolo": "조반니 바티스타 티에폴로", "Francesco Guardi": "프란체스코 과르디",
    "Chen Hongshou": "천훙서우(陳洪綬)",
}

# 영어 주제어 → 한국어 캡션 토큰 (자주 나오는 것만, 나머지는 버린다)
SUBJECT_KO = {
    "landscapes": "풍경", "landscape": "풍경", "portraits": "초상", "portrait": "초상",
    "portraits: male subject": "초상", "portraits: female subject": "초상",
    "men": "남성", "man": "남성", "women": "여성", "woman": "여성", "children": "아이", "boys": "아이",
    "girls": "아이", "trees": "나무", "foliage": "나무", "horses": "말", "boats": "배", "ships": "배",
    "dogs": "개", "cows": "소", "birds": "새", "animals": "동물", "flowers": "꽃", "fruit": "과일",
    "still life": "정물", "mountains": "산", "clouds": "구름", "sky": "하늘", "water": "물", "rivers": "강",
    "virgin mary": "성모", "madonna and child": "성모", "christ": "예수", "jesus": "예수", "angels": "천사",
    "saints": "성인", "buddhism": "불교", "buildings": "건물", "building": "건물", "architecture": "건물",
    "interiors": "실내", "interior": "실내", "soldiers": "군인", "female nudes": "누드",
    "musical instruments": "음악", "music": "음악", "musicians": "음악", "death": "죽음",
}
MAX_SUBJECTS = 3


def tradition(origin: str, department: str, subjects: str) -> str:
    text = f"{origin} {department} {subjects}".lower()
    for key, ko in (("korea", "한국"), ("japan", "일본"), ("china", "중국"), ("chinese", "중국"),
                    ("india", "인도"), ("nepal", "인도"), ("persia", "페르시아"), ("iran", "페르시아"),
                    ("america", "미국"), ("united states", "미국")):
        if key in text:
            return ko
    # 유럽은 나라까지 나눈다 — "르네상스 이탈리아 그림을 18세기 유럽이라 부르는" 식의 뭉뚱그림을 줄인다 (2026-10-11 피드백)
    for keys, ko in ((("italian", "italy", "florence", "florentine", "venice", "venetian", "siena", "sienese", "rome", "bologna",
                       "milan", "naples", "genoa", "ferrara"), "이탈리아"),
                     (("flemish", "flanders", "antwerp", "brussels", "bruges"), "플랑드르"),
                     (("dutch", "netherland", "holland", "haarlem", "amsterdam", "utrecht", "leiden", "delft"), "네덜란드"),
                     (("spanish", "spain", "seville", "madrid"), "스페인"),
                     (("french", "france", "paris"), "프랑스"),
                     (("british", "english", "england", "scottish", "london"), "영국"),
                     (("german", "germany", "austrian", "austria", "swiss"), "독일어권")):
        if any(k in text for k in keys):
            return ko
    if "europe" in text:
        return "유럽"
    return "기타"


def technique(medium: str) -> str:
    m = (medium or "").lower()
    if "oil" in m:
        return "유화"
    if "fresco" in m:
        return "프레스코"
    if "tempera" in m and "gum" not in m:
        return "템페라"
    if "gum tempera" in m or "opaque watercolor" in m or "gouache" in m:
        return "과슈"
    if "watercolor" in m:
        return "수채"
    if "pastel" in m:
        return "파스텔"
    if "ink" in m:
        return "수묵채색" if "color" in m else "수묵"
    return "회화"


def century(year) -> str | None:
    if year is None:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    if y < 1000 or y > 2000:
        return None
    return f"{y // 100 + 1}세기"


def caption_tokens(row: dict) -> list[str]:
    tokens = []
    c = century(row["year_start"])
    if c:
        tokens.append(c)
    tokens.append(tradition(row["origin"] or "", row["department"] or "", row["subjects"] or ""))
    tokens.append(technique(row["medium"]))
    seen = []
    for word in re.split(r"[,;]\s*", row["subjects"] or ""):
        ko = SUBJECT_KO.get(word.strip().lower())
        if ko and ko not in seen:
            seen.append(ko)
        if len(seen) == MAX_SUBJECTS:
            break
    if seen:
        tokens.append("<sep>")   # 앞은 '어떤 그림', 뒤는 '무엇을 그렸나'
        tokens.extend(seen)
    return tokens


def download(row: dict) -> bool:
    path = IMAGES / f"{row['id']}.jpg"
    if path.exists():
        return True
    url = row["thumbnail_url"] or row["image_url"]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(path, "JPEG", quality=88)
        return True
    except Exception as e:  # noqa: BLE001 — 하나 실패해도 나머지는 계속
        print(f"skip {row['id']}: {type(e).__name__}")
        return False


def make_views(item_id: int, seed: int) -> None:
    """학습용 보기 4개: 0=전체(정사각형으로 늘림, 앱과 같은 방식), 1=좌우 반전, 2·3=무작위로 60~85% 잘라 보기."""
    if (VIEWS / f"{item_id}_3.jpg").exists() and (THUMBS / f"{item_id}.webp").exists():
        return
    rnd = random.Random(seed * 100003 + item_id)
    img = Image.open(IMAGES / f"{item_id}.jpg").convert("RGB")
    full = img.resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
    views = [full, full.transpose(Image.FLIP_LEFT_RIGHT)]
    w, h = img.size
    for _ in range(2):
        s = rnd.uniform(0.6, 0.85)
        cw, ch = int(w * s), int(h * s)
        x, y = rnd.randint(0, w - cw), rnd.randint(0, h - ch)
        views.append(img.crop((x, y, x + cw, y + ch)).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR))
    for k, v in enumerate(views):
        v.save(VIEWS / f"{item_id}_{k}.jpg", "JPEG", quality=92)
    thumb = img.copy()
    thumb.thumbnail((160, 160))
    thumb.save(THUMBS / f"{item_id}.webp", "WEBP", quality=70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-db", required=True)
    ap.add_argument("--captions", type=int, default=3000, help="캡션 학습에 더 쓸 그림 수 (작가 그림 외)")
    ap.add_argument("--old-masters", action="store_true",
                    help="1300~1699년 유럽 회화를 모두 더한다 (르네상스·매너리즘·바로크 — 캡션 시대 판정과 소장품 대조 보강)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    conn = sqlite3.connect(args.art_db)
    conn.row_factory = sqlite3.Row
    painting = ("image_url != '' AND is_public_domain = 1 "
                "AND (classification LIKE '%paint%' OR medium LIKE '%oil%')")
    artist_rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM artworks WHERE {painting} AND artist IN ({','.join('?' * len(ARTISTS))})", ARTISTS)]
    others = [dict(r) for r in conn.execute(
        f"SELECT * FROM artworks WHERE {painting} AND artist NOT IN ({','.join('?' * len(ARTISTS))})", ARTISTS)]
    random.Random(args.seed).shuffle(others)
    rows = artist_rows + others[:args.captions]
    if args.old_masters:
        have = {r["id"] for r in rows}
        old = [r for r in others[args.captions:] if r["id"] not in have and r["year_start"] and 1300 <= int(r["year_start"]) <= 1699
               and tradition(r["origin"] or "", r["department"] or "", r["subjects"] or "") not in
               ("한국", "일본", "중국", "인도", "페르시아", "미국", "기타")]
        print(f"옛 거장(1300~1699 유럽) 회화 {len(old)}점 추가")
        rows += old

    IMAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(download, rows))

    meta = []
    for row, good in zip(rows, ok):
        if not good:
            continue
        meta.append({
            "id": row["id"], "source": row["source"], "title": row["title"], "artist": row["artist"],
            "label": row["artist"] if row["artist"] in ARTISTS else None,
            "caption": caption_tokens(row), "date": row["date_display"],
            "source_url": row["source_url"], "thumbnail_url": row["thumbnail_url"] or row["image_url"],
        })
    VIEWS.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    for m in meta:
        make_views(m["id"], args.seed)
    (OUT / "metadata.json").write_text(json.dumps({"artists": ARTISTS, "artist_ko": ARTIST_KO, "items": meta},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    labeled = sum(1 for m in meta if m["label"])
    print(f"그림 {len(meta)}점 (작가 라벨 {labeled}점) → {OUT}")


if __name__ == "__main__":
    main()
