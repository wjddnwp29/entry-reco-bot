import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

ROOT = Path("./dataset")   # 현재 chatbot/dataset 폴더
OUT  = ROOT / "built"
OUT.mkdir(parents=True, exist_ok=True)

def to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if x is None: return False
    return str(x).strip().lower() in ("true","1","yes","y","t")

def is_empty(s: Optional[str]) -> bool:
    return not s or len(str(s).strip()) == 0

def clean_text(s: Optional[str]) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def difficulty_to_int(x) -> int:
    if x is None: return 1
    s = str(x).strip().lower()
    if s in ("1","beginner","초급"): return 1
    if s in ("2","intermediate","중급"): return 2
    if s in ("3","advanced","고급"): return 3
    try:
        v = int(float(s)); return v if v in (1,2,3) else 1
    except: return 1

def make_discovery_url(id_or_objid: Optional[str]) -> Optional[str]:
    if not id_or_objid: return None
    s = str(id_or_objid)
    m = re.search(r'([0-9a-fA-F]{24})', s)
    if m:
        return f"https://playentry.org/discovery/{m.group(1)}"
    if s.startswith("http"): return s
    return f"https://playentry.org/discovery/{s}"

def normalize_row(row: Dict[str, Any], dataset_kind: str) -> Optional[Dict[str, Any]]:
    if not to_bool(row.get("isopen")): return None
    goal = row.get("goal") or row.get("objectives")
    if is_empty(goal): return None

    _id = row.get("_id") or row.get("id") or row.get("objectId")
    title = row.get("title") or row.get("name") or ""
    description = row.get("description") or row.get("summary") or ""
    category = row.get("category") or "general"
    difficulty = difficulty_to_int(row.get("difficulty"))

    title, description, goal = map(clean_text, [title, description, goal])

    if dataset_kind == "discovery":
        url = make_discovery_url(_id)
    else:
        url = row.get("url") or None

    return {
        "id": str(_id) if _id else "",
        "title": title,
        "category": str(category).strip(),
        "difficulty": difficulty,
        "url": url,
        "search_text": " ".join([title, description, goal]).strip()
    }

def process_csv(path: Path, kind: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype="object").fillna("")
    return [rec for _, r in df.iterrows() if (rec:=normalize_row(r.to_dict(), kind))]

def process_json(path: Path, kind: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [rec for r in data if (rec:=normalize_row(r, kind))]

def save_json(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    files = list(ROOT.glob("*.csv")) + list(ROOT.glob("*.json"))
    merged = {"discovery": [], "newlectures": []}

    for f in files:
        name = f.stem  # 예: discovery_3
        if name.startswith("discovery"):
            kind = "discovery"
        elif name.startswith("newlectures"):
            kind = "newlectures"
        else:
            continue

        if f.suffix == ".csv":
            rows = process_csv(f, kind)
        else:
            rows = process_json(f, kind)

        save_json(OUT / f"{name}_filtered.json", rows)
        merged[kind].extend(rows)

    for kind in merged:
        dedup = {r["id"]: r for r in merged[kind]}
        save_json(OUT / f"{kind}_merged.json", list(dedup.values()))

    print("완료. 결과는 dataset/built/ 에 저장됨")

if __name__ == "__main__":
    main()
