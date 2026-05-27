# app.py
import os, csv, json, random
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, send_from_directory
)

# ----- 추천 모듈: similarity baseline + MAML meta-ranker -----
from recommender import (
    compare_recommendations,
    get_recommender_config,
    recommend_by_maml_meta,
    recommend_by_similarity,
    recommend_selected,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ===== 경로/파일 =====
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
ENTRY_TEST_DIR = DATASET_DIR                                  # entry_test_set_*.json 이 여기 있음
ENTRY_TEST_PATTERN = "entry_test_set_"
ENTRY_IMAGE_DIR = (DATASET_DIR / "entry_test_image").resolve()
USER_DB = ROOT / "users.csv"
USER_LOGS_DIR = ROOT / "logs" / "user_logs"
USER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 유틸: 사용자 DB =====
def load_users():
    if not USER_DB.exists():
        return {}
    users = {}
    with open(USER_DB, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" in row and "password" in row:
                users[row["id"]] = row
    return users

def save_user(user):
    file_exists = USER_DB.exists()
    with open(USER_DB, mode="a", newline="", encoding="utf-8") as f:
        # birthdate 사용(나이 X)
        writer = csv.DictWriter(f, fieldnames=["id", "password", "name", "birthdate", "gender", "grade", "level"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(user)

# ===== 엔트리 테스트 =====
def load_random_entry_test():
    files = [p for p in ENTRY_TEST_DIR.glob(f"{ENTRY_TEST_PATTERN}*.json")]
    if not files:
        raise FileNotFoundError("dataset 폴더에 entry_test_set_*.json 이 없습니다.")
    selected = random.choice(files)
    with open(selected, encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    return selected.name, items

def grade_entry_test(answers, items):
    # answers: ["A","B","C","D","A"]
    def norm_to_idx(a):
        if isinstance(a, str):
            a = a.strip().upper()
            if a in "ABCD":
                return "ABCD".index(a)
            try:
                return int(a)
            except:
                return -1
        return int(a)

    correct = 0
    for i, a in enumerate(answers):
        gt = items[i].get("answer")
        gt_idx = norm_to_idx(gt)
        if gt_idx < 0 and isinstance(gt, str) and gt in "ABCD":
            gt_idx = "ABCD".index(gt)
        if norm_to_idx(a) == gt_idx:
            correct += 1

    score = correct * 20  # 5문항 * 20점 = 100점
    if score >= 90:
        level = "Advanced"
    elif score >= 50:
        level = "Intermediate"
    else:
        level = "Beginner"
    return score, correct, level

# ===== 로그 저장 =====
def save_user_log(user_id, user_level, payload):
    log_path = USER_LOGS_DIR / f"{user_id}.json"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log_data = json.load(f)
    else:
        log_data = {"user_id": user_id, "created_at": datetime.now().isoformat(), "sessions": []}

    log_data["sessions"].append({
        "timestamp": datetime.now().isoformat(),
        "user_level": user_level,
        **payload
    })
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# ================== 라우팅 ==================
@app.route("/")
def main():
    return render_template("main.html")

# 설명서 화면
@app.route("/manual")
def manual():
    return render_template("manual.html")

# 엔트리 이미지 서빙
@app.route("/entry_img/<path:filename>")
def entry_img(filename):
    return send_from_directory(ENTRY_IMAGE_DIR, filename)

# ---- 로그인 / 회원가입 ----
@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        user_id = request.form.get("id","").strip()
        pw = request.form.get("pw","").strip()
        users = load_users()
        if user_id in users and users[user_id]["password"] == pw:
            session["user_id"] = user_id
            user_level = users[user_id].get("level", "")
            save_user_log(user_id, user_level or "", {"event": "login"})
            if not user_level or user_level.strip().lower() in ["", "new"]:
                return redirect(url_for("entry_test"))
            session["user_level"] = user_level
            return redirect(url_for("chat"))
        else:
            msg = "로그인 실패! 다시 시도해주세요."
    return render_template("login.html", msg=msg)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    msg = ""
    if request.method == "POST":
        user_id   = request.form.get("id","").strip()
        pw        = request.form.get("pw","").strip()
        pw2       = request.form.get("pw2","").strip()
        name      = request.form.get("name","").strip()
        birthdate = request.form.get("birthdate","").strip()  # 달력 입력
        gender    = request.form.get("gender","").strip()      # 여/남
        grade     = request.form.get("grade","").strip()       # 초1~초6

        if pw != pw2:
            msg = "비밀번호 확인이 일치하지 않습니다."
            return render_template("signup.html", msg=msg)

        users = load_users()
        if user_id in users:
            msg = "이미 존재하는 아이디입니다."
            return render_template("signup.html", msg=msg)

        save_user({
            "id": user_id, "password": pw,
            "name": name, "birthdate": birthdate, "gender": gender,
            "grade": grade, "level": "new"
        })
        flash("회원가입 완료! 로그인 해주세요.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html", msg=msg)

@app.post("/logout")
def logout():
    save_user_log(session.get("user_id","guest"), session.get("user_level",""), {"event": "logout"})
    session.clear()
    return redirect(url_for("main"))

# ---- 엔트리 테스트 화면/제출 ----
@app.route("/entry_test", methods=["GET", "POST"])
def entry_test():
    if request.method == "GET":
        set_name, items = load_random_entry_test()
        session["entry_set_name"] = set_name
        session["entry_items"] = items
        return render_template("entry_test.html", set_name=set_name, items=items)

    # POST
    items = session.get("entry_items", [])
    if not items:
        flash("세션이 만료되었습니다. 다시 시작해주세요.", "error")
        return redirect(url_for("entry_test"))

    answers = [request.form.get(f"q{i}", "").strip().upper() for i in range(len(items))]
    score, correct, level = grade_entry_test(answers, items)

    uid = session.get("user_id","guest")
    users = load_users()
    if uid in users:
        users[uid]["level"] = level
        with open(USER_DB, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id","password","name","birthdate","gender","grade","level"])
            w.writeheader()
            for k,v in users.items():
                w.writerow(v)

    save_user_log(uid, level, {
        "entry_test": {
            "set": session.get("entry_set_name"),
            "answers": answers,
            "score": score,
            "correct": correct
        }
    })
    session["user_level"] = level
    return render_template(
        "entry_test.html",
        set_name=session.get("entry_set_name"),
        items=items, score=score, correct=correct, level=level,
        answers=answers, finished=True
    )

# ---- 추천 챗 화면 ----
@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_level = session.get("user_level", "Beginner")
    rec_config = get_recommender_config()
    default_recommender = rec_config["recommender"].get("type", "maml_meta")
    default_debug = bool(rec_config["recommender"].get("debug_compare", True))

    # GET: 빈 화면
    if request.method == "GET":
        return render_template("chat.html",
                               level=user_level,
                               recommender_type=default_recommender,
                               debug_compare=default_debug,
                               user_question=None,
                               chatbot_response=None,
                               question_results=[],
                               maml_results=[])

    # POST: JSON도 지원
    if request.is_json:
        data = request.get_json() or {}
        query = (data.get("question") or "").strip()
        level = data.get("user_level") or user_level
        topk  = int(data.get("topk", 6))
        recommender_type = data.get("recommender_type") or default_recommender
        debug_compare = to_bool(data.get("debug_compare"), default=default_debug)

        uid = session.get("user_id","guest")
        if debug_compare:
            compared = compare_recommendations(query, topk=topk, user_level=level, user_id=uid)
            sim = compared["similarity"]
            maml = compared["maml_meta"]
            selected = maml if recommender_type == "maml_meta" else sim
        else:
            selected = recommend_selected(
                query,
                recommender_type=recommender_type,
                topk=topk,
                user_level=level,
                user_id=uid,
            )
            sim = selected if recommender_type == "similarity" else []
            maml = selected if recommender_type == "maml_meta" else []

        save_user_log(uid, level, {
            "query": query,
            "recommender_type": recommender_type,
            "debug_compare": debug_compare,
            "recommend": {"similarity": sim, "maml_meta": maml, "selected": selected}
        })
        return jsonify({
            "results": selected,
            "similarity": sim,
            "title": maml,
            "maml_meta": maml,
            "level": level,
            "selected_type": recommender_type,
            "debug_compare": debug_compare,
        })

    # 폼 POST (chat.html)
    user_question = (request.form.get("user_question") or "").strip()
    recommender_type = request.form.get("recommender_type") or default_recommender
    debug_compare = to_bool(request.form.get("debug_compare"), default=default_debug)
    uid = session.get("user_id","guest")

    if debug_compare:
        compared = compare_recommendations(user_question, topk=6, user_level=user_level, user_id=uid)
        sim = compared["similarity"]
        maml = compared["maml_meta"]
        selected = maml if recommender_type == "maml_meta" else sim
    else:
        selected = recommend_selected(
            user_question,
            recommender_type=recommender_type,
            topk=6,
            user_level=user_level,
            user_id=uid,
        )
        sim = selected if recommender_type == "similarity" else []
        maml = selected if recommender_type == "maml_meta" else []

    save_user_log(uid, user_level, {
        "query": user_question,
        "recommender_type": recommender_type,
        "debug_compare": debug_compare,
        "recommend": {"similarity": sim, "maml_meta": maml, "selected": selected}
    })

    return render_template(
        "chat.html",
        level=user_level,
        recommender_type=recommender_type,
        debug_compare=debug_compare,
        user_question=user_question,
        chatbot_response="추천 결과를 확인해보세요!",
        question_results=sim,
        maml_results=maml
    )

# ---- 카드 선택/피드백 로깅 ----
@app.post("/select")
def select_content():
    content_id = request.form.get("content_id")
    title      = request.form.get("title")
    source     = request.form.get("source")
    from_model = request.form.get("from_model")
    uid   = session.get("user_id", "guest")
    level = session.get("user_level", "")
    save_user_log(uid, level, {
        "event": "select",
        "content": {"id": content_id, "title": title, "source": source, "from_model": from_model}
    })
    return redirect(url_for("chat"))

@app.post("/feedback")
def feedback_content():
    content_id = request.form.get("content_id")
    title      = request.form.get("title")
    source     = request.form.get("source")
    from_model = request.form.get("from_model")
    feedback   = request.form.get("feedback")    # like / dislike
    uid   = session.get("user_id", "guest")
    level = session.get("user_level", "")
    save_user_log(uid, level, {
        "event": "feedback",
        "feedback": feedback,
        "content": {"id": content_id, "title": title, "source": source, "from_model": from_model}
    })
    return redirect(url_for("chat"))

# ---- 자유형 로깅(JS용) ----
@app.post("/log_action")
def log_action():
    data  = request.get_json() or {}
    uid   = session.get("user_id","guest")
    level = session.get("user_level","")
    save_user_log(uid, level, {"action": data})
    return jsonify({"ok": True})

# ============================================
if __name__ == "__main__":
    app.run(debug=True)
