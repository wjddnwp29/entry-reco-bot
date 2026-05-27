# -*- coding: utf-8 -*-
"""
디스커버리(3/4/5) + 강의 CSV + 교과 CSV를 머지해
dataset/built/contents_meta.json 을 생성합니다.
"""

import csv, json, re, html
from pathlib import Path
from collections import OrderedDict

# ---- 경로 설정 (프로젝트 루트에서 실행 가정) ----
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "dataset"
BUILT = DATA / "built"
BUILT.mkdir(parents=True, exist_ok=True)

SRC_DISCOVERY_3 = DATA / "discovery_3.csv"        # rows=38 (CSV)
SRC_DISCOVERY_4 = DATA / "discovery_4.json"       # list(JSON)
SRC_DISCOVERY_5 = DATA / "discovery_5.json"       # list(JSON)

SRC_LECTURES    = ROOT / "entry_lectures_title_utf8.csv"  # rows=392 (CSV)
SRC_TEXTBOOK    = ROOT / "entry_textbook.csv"             # rows=32  (CSV)

OUT_META        = BUILT / "contents_meta.json"

# ---------------- 공통 유틸 ----------------
def clean_text(x: str) -> str:
    if x is None: return ""
    s = html.unescape(str(x))
    s = re.sub(r"<.*?>", " ", s)          # HTML 태그 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s

def difficulty_to_int(x):
    if x is None: return 1
    s = str(x).strip().lower()
    if s.isdigit():
        n = int(s)
        if n in (1, 2, 3): return n
    table = {
        "1":1, "2":2, "3":3,
        "초급":1, "입문":1, "기초":1, "beginner":1,
        "중급":2, "보통":2, "intermediate":2,
        "고급":3, "심화":3, "advanced":3,
    }
    return table.get(s, 1)

def make_discovery_url(any_id) -> str | None:
    if not any_id: return None
    sid = str(any_id)
    m = re.match(r"ObjectId\(([\w\d]+)\)", sid)
    oid = m.group(1) if m else sid
    return f"https://playentry.org/discovery/{oid}"

def pick(d, *keys):
    for k in keys:
        if k in d and str(d[k]).strip():
            return d[k]
    return None

def synth_description(*candidates, title=""):
    text = " ".join([clean_text(x) for x in candidates if x]).strip()
    if not text: text = clean_text(title)
    return (text[:100] + "…") if len(text) > 100 else text

def synth_goal(text_like="", title=""):
    t = clean_text(text_like) or clean_text(title)
    m = re.search(r"(.+?)(만들어요|만들기|만듭니다|구현)", t)
    if m:
        return m.group(1).strip(" .,!~") + " 만들기"
    if title:
        return f"{clean_text(title)} 완성하기"
    return "직접 만들어 보기"

def build_item(id_, title, category, difficulty, url, description, goal, search_text):
    return {
        "id": str(id_) if id_ else title,
        "title": clean_text(title),
        "category": (category or "general").strip(),
        "difficulty": difficulty_to_int(difficulty),
        "url": url,
        "description": clean_text(description),
        "goal": clean_text(goal),
        "search_text": clean_text(search_text),
    }

# ---------------- 로더들 ----------------
def load_discovery_3_csv(path: Path):
    items = []
    if not path.exists(): return items
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            _id   = pick(r, "_id", "id")
            title = pick(r, "title") or ""
            if not str(title).strip(): continue
            url   = make_discovery_url(_id)
            cat   = pick(r, "categoryCode") or "discovery"
            diff  = pick(r, "difficulty")
            # 설명 생성: summary/description/content/goal 조합
            desc  = synth_description(pick(r, "summary"), pick(r, "description"),
                                      pick(r, "content"), pick(r, "goal"), title=title)
            goal  = synth_goal(pick(r, "goal") or pick(r, "description"), title=title)
            stext = " ".join([str(title), pick(r, "summary") or "", pick(r, "goal") or ""])
            items.append(build_item(_id, title, cat, diff, url, desc, goal, stext))
    return items

def load_discovery_json(path: Path):
    items = []
    if not path.exists(): return items
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    for r in data:
        _id   = pick(r, "_id", "id")
        title = pick(r, "title") or ""
        if not str(title).strip(): continue
        url   = make_discovery_url(_id)
        cat   = pick(r, "categoryCode") or "discovery"
        diff  = pick(r, "difficulty")
        desc  = synth_description(pick(r, "summary"), pick(r, "description"),
                                  pick(r, "content"), pick(r, "goal"), title=title)
        goal  = synth_goal(pick(r, "goal") or pick(r, "description"), title=title)
        stext = " ".join([str(title), pick(r, "summary") or "", pick(r, "goal") or ""])
        items.append(build_item(_id, title, cat, diff, url, desc, goal, stext))
    return items

def load_csv_meta(path: Path, default_cat="lecture"):
    """entry_lectures_title_utf8.csv / entry_textbook.csv 공통 로더"""
    items = []
    if not path.exists(): return items
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            _id   = pick(r, "id", "_id", "uuid")
            title = pick(r, "title", "name") or ""
            if not str(title).strip(): continue
            url   = pick(r, "url", "link")
            cat   = pick(r, "category") or default_cat
            diff  = pick(r, "difficulty", "level")
            desc  = pick(r, "description", "summary", "intro") or ""
            goal  = pick(r, "goal", "objectives") or ""
            if not desc:
                # 없는 경우 보강
                desc = synth_description(goal, title=title)
            if not goal:
                goal = synth_goal(desc, title)
            stext = " ".join([str(title), desc, goal])
            items.append(build_item(_id, title, cat, diff, url, desc, goal, stext))
    return items

# ---------------- 머지/정리 ----------------
def dedupe(items):
    """title+url 기준 중복 제거"""
    seen, out = set(), []
    for it in items:
        key = (it["title"], it["url"])
        if key in seen: 
            continue
        seen.add(key)
        # URL이 없다면 제거(카드 클릭을 위해 URL은 필수로 두자)
        if not it["url"]:
            continue
        out.append(it)
    return out

def main():
    items = []
    # Discovery
    items += load_discovery_3_csv(SRC_DISCOVERY_3)
    items += load_discovery_json(SRC_DISCOVERY_4)
    items += load_discovery_json(SRC_DISCOVERY_5)
    # Lectures/Textbook
    items += load_csv_meta(SRC_LECTURES, default_cat="lecture")
    items += load_csv_meta(SRC_TEXTBOOK, default_cat="textbook")

    items = dedupe(items)
    OUT_META.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # 통계 출력
    by_cat = OrderedDict()
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
    print(f"생성: {OUT_META} (총 {len(items)}개)")
    for k,v in by_cat.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
