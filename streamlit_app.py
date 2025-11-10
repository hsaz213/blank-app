import io
from datetime import datetime, timedelta
from uuid import uuid4
import random

import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from icalendar import Calendar, Event, Alarm
from pytz import timezone

# =========================================================
# Config
# =========================================================
EVENT_DURATION_MINUTES = 120  # exam length used for overlap checks & ICS

# =========================================================
# i18n
# =========================================================
STRINGS = {
    "en": {
        "title": "🗓️ Exam Scheduler — Maximize Minimum Gap",
        "read_csv": "read csv",
        "randomize": "randomize",
        "add_date": "add date",
        "optimize": "optimize schedule",
        "export": "export",

        "input_header": "Input",
        "input_caption": "Left table is read-only. Use the buttons above to load or add rows.",

        "toast_loaded": "Loaded {n} subjects.",
        "toast_randomized": "Randomized {n} subjects.",
        "toast_row_added": "Row added.",
        "toast_export_ready": "Export ready.",
        "toast_optimization_complete": "Optimization complete.",

        "error_read_csv": "Failed to read CSV: {err}",
        "error_rows_invalid": "Some rows are invalid. See left table (added 'errors' column).",

        "clash_alert": "Clash detected: {n} exam(s) overlap and were excluded from the schedule.",
        "min_gap_banner": "Minimum gap (between starts): {gap}.",
        "one_exam_banner": "Only one exam scheduled; no gap to report.",

        "optimized_header": "Optimized schedule",
        "exp_excluded": "Why were some exams excluded?",

        "col_subject": "Subject",
        "col_exam1": "Exam 1",
        "col_exam2": "Exam 2",
        "col_sched_start": "Scheduled start",
        "col_slot": "Slot",
        "col_overlaps_with": "Overlaps with",

        "dlg_add_title": "Add subject",
        "dlg_subject": "Subject",
        "dlg_exam1_date": "Exam 1 date",
        "dlg_exam1_time": "Exam 1 time",
        "dlg_exam2_date": "Exam 2 date",
        "dlg_exam2_time": "Exam 2 time",
        "dlg_save": "Save",
        "dlg_cancel": "Cancel",
        "dlg_warn_subject": "Please enter a subject.",

        "dlg_rand_title": "Randomize demo data",
        "dlg_rand_how_many": "How many subjects?",
        "dlg_generate": "Generate",

        "calendar_name": "Exam Schedule",
        "ics_desc": "Exam for {subject}",
        "ics_alarm": "Upcoming exam: {subject}",

        "csv_meta": "# Minimum gap (between starts): {gap}\n",

        "lang_label": "Language / Sprache",
        "lang_en": "English",
        "lang_de": "Deutsch",
    },
    "de": {
        "title": "🗓️ Prüfungsplaner — Maximierung des Mindestabstands",
        "read_csv": "CSV laden",
        "randomize": "zufällig",
        "add_date": "Termin hinzufügen",
        "optimize": "Zeitplan optimieren",
        "export": "exportieren",

        "input_header": "Eingabe",
        "input_caption": "Die linke Tabelle ist schreibgeschützt. Obige Buttons zum Laden oder Hinzufügen verwenden.",

        "toast_loaded": "{n} Fächer geladen.",
        "toast_randomized": "{n} Fächer zufällig erstellt.",
        "toast_row_added": "Zeile hinzugefügt.",
        "toast_export_ready": "Export bereit.",
        "toast_optimization_complete": "Optimierung abgeschlossen.",

        "error_read_csv": "CSV konnte nicht gelesen werden: {err}",
        "error_rows_invalid": "Einige Zeilen sind ungültig. Siehe linke Tabelle (Spalte 'errors' hinzugefügt).",

        "clash_alert": "Konflikt erkannt: {n} Prüfung(en) überlappen sich und wurden ausgeschlossen.",
        "min_gap_banner": "Mindestabstand (zwischen Starts): {gap}.",
        "one_exam_banner": "Nur eine Prüfung geplant; kein Abstand zu berichten.",

        "optimized_header": "Optimierter Zeitplan",
        "exp_excluded": "Warum wurden einige Prüfungen ausgeschlossen?",

        "col_subject": "Fach",
        "col_exam1": "Ersttermin",
        "col_exam2": "Zweittermin",
        "col_sched_start": "Geplanter Beginn",
        "col_slot": "Anlass",
        "col_overlaps_with": "Überlappt mit",

        "dlg_add_title": "Fach hinzufügen",
        "dlg_subject": "Fach",
        "dlg_exam1_date": "Ersttermin Datum",
        "dlg_exam1_time": "Ersttermin Uhrzeit",
        "dlg_exam2_date": "Zweittermin Datum",
        "dlg_exam2_time": "Zweittermin Uhrzeit",
        "dlg_save": "Speichern",
        "dlg_cancel": "Abbrechen",
        "dlg_warn_subject": "Bitte ein Fach eingeben.",

        "dlg_rand_title": "Demo-Daten zufällig erstellen",
        "dlg_rand_how_many": "Wie viele Fächer?",
        "dlg_generate": "Erzeugen",

        "calendar_name": "Prüfungsplan",
        "ics_desc": "Prüfung für {subject}",
        "ics_alarm": "Bevorstehende Prüfung: {subject}",

        "csv_meta": "# Mindestabstand (Stunden zwischen Starts): {gap}\n",

        "lang_label": "Language / Sprache",
        "lang_en": "English",
        "lang_de": "Deutsch",
    },
}

def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "en")
    txt = STRINGS.get(lang, STRINGS["en"]).get(key, key)
    return txt.format(**kwargs) if kwargs else txt

# --- Gap formatter ----------------------------------------------------------

def format_gap_text(td: timedelta, lang: str = "en") -> str:
    total_minutes = int(td.total_seconds() // 60)
    h, m = divmod(total_minutes, 60)
    if lang == "de":
        parts = []
        if h:
            parts.append(f"{h} Stunde" + ("" if h == 1 else "n"))
        if m:
            parts.append(f"{m} Minute" + ("" if m == 1 else "n"))
        if not parts:
            parts.append("0 Minuten")
        return " und ".join(parts)
    else:
        parts = []
        if h:
            parts.append(f"{h} hour" + ("" if h == 1 else "s"))
        if m:
            parts.append(f"{m} minute" + ("" if m == 1 else "s"))
        if not parts:
            parts.append("0 minutes")
        return " and ".join(parts)

# -----------------------------
# Helpers: data & state
# -----------------------------

LEFT_COLUMNS = ["subject", "exam1", "exam2"]   # left table, read-only
RIGHT_COLUMNS = ["subject", "optimal_date", "slot"]  # right table, read-only

def init_state():
    if "left_df" not in st.session_state:
        st.session_state.left_df = pd.DataFrame(columns=LEFT_COLUMNS)
    if "right_df" not in st.session_state:
        st.session_state.right_df = pd.DataFrame(columns=RIGHT_COLUMNS)
    if "min_gap_td" not in st.session_state:
        st.session_state.min_gap_td = None
    if "last_export" not in st.session_state:
        st.session_state.last_export = None  # (csv_bytes, ics_bytes)
    if "show_add_dialog" not in st.session_state:
        st.session_state.show_add_dialog = False
    if "show_random_dialog" not in st.session_state:
        st.session_state.show_random_dialog = False
    if "dropped_df" not in st.session_state:
        st.session_state.dropped_df = pd.DataFrame()
    if "drop_explanations" not in st.session_state:
        st.session_state.drop_explanations = pd.DataFrame()
    if "clash_count" not in st.session_state:
        st.session_state.clash_count = 0
    if "lang" not in st.session_state:
        st.session_state.lang = "en"  # default language

init_state()

# --- Add-form state helpers (fix time pickers) ------------------------------

def _rounded_now():
    return datetime.now().replace(minute=0, second=0, microsecond=0)

def init_add_form_defaults():
    if "add_subj" not in st.session_state:
        st.session_state.add_subj = ""
    if "add_d1" not in st.session_state:
        st.session_state.add_d1 = datetime.now().date()
    if "add_t1" not in st.session_state:
        st.session_state.add_t1 = _rounded_now().time()
    if "add_d2" not in st.session_state:
        st.session_state.add_d2 = datetime.now().date()
    if "add_t2" not in st.session_state:
        st.session_state.add_t2 = (_rounded_now() + timedelta(hours=2)).time()

def clear_add_form_defaults():
    for k in ("add_subj", "add_d1", "add_t1", "add_d2", "add_t2"):
        st.session_state.pop(k, None)

# -----------------------------
# Validation
# -----------------------------

REQ_COLS = {
    "MODULNAME": str,
    "ERSTTERMIN_DATUM": str,
    "ERSTTERMIN_UHRZEIT": str,
    "ZWEITTERMIN_DATUM": str,
    "ZWEITTERMIN_UHRZEIT": str,
}

def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    cols = [c.upper().strip() for c in df.columns]
    missing = [c for c in REQ_COLS.keys() if c not in cols]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "ok"

def df_from_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    df.columns = [c.upper().strip() for c in df.columns]
    for col in ["ERSTTERMIN_DATUM", "ERSTTERMIN_UHRZEIT", "ZWEITTERMIN_DATUM", "ZWEITTERMIN_UHRZEIT", "MODULNAME"]:
        df[col] = df[col].astype(str).str.strip()
    first_str  = df["ERSTTERMIN_DATUM"] + " " + df["ERSTTERMIN_UHRZEIT"]
    second_str = df["ZWEITTERMIN_DATUM"] + " " + df["ZWEITTERMIN_UHRZEIT"]
    fmt = "%Y-%m-%d %H:%M:%S"
    first_dt  = pd.to_datetime(first_str,  format=fmt, errors="coerce", utc=False)
    second_dt = pd.to_datetime(second_str, format=fmt, errors="coerce", utc=False)
    left = pd.DataFrame({
        "subject": df["MODULNAME"],
        "exam1": first_dt,
        "exam2": second_dt,
    })
    return left

def row_errors(df: pd.DataFrame) -> pd.Series:
    errs = []
    for _, r in df.iterrows():
        e = []
        if pd.isna(r["subject"]) or str(r["subject"]).strip() == "":
            e.append("subject missing")
        if pd.isna(r["exam1"]):
            e.append("exam1 invalid")
        if pd.isna(r["exam2"]):
            e.append("exam2 invalid")
        errs.append("; ".join(e))
    return pd.Series(errs)

# -----------------------------
# Random data for demo
# -----------------------------

SUBJECT_POOL = [
    "Mathematik I","Mathematik II","Statistik","Informatik I","Informatik II","Recht","BWL","VWL",
    "Marketing","Finanzierung","Investition","Controlling","Logistik","Operations Research",
    "Datenbanken","Softwaretechnik","Maschinelles Lernen","KI Grundlagen","Projektmanagement",
    "Kommunikation","Wirtschaftsinformatik","Elektrotechnik","Physik","Chemie","Nachhaltigkeit"
]

def random_demo_df(n=12, start_date=None, days=40):
    if start_date is None:
        start_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    base = SUBJECT_POOL[:]
    if n <= len(base):
        subjects = random.sample(base, k=n)
    else:
        subjects = base[:]
        extra = [f"Subject {i:03d}" for i in range(len(base)+1, n+1)]
        subjects.extend(extra)
    rows = []
    for s in subjects:
        d1 = start_date + timedelta(
            days=random.randint(0, days),
            hours=random.choice([0,2,4,6,8,10,12,14,16])
        )
        d2 = d1 + timedelta(
            days=random.randint(3, 21),
            hours=random.choice([2,4,6,8,10,12])
        )
        rows.append({"subject": s, "exam1": d1, "exam2": d2})
    return pd.DataFrame(rows, columns=LEFT_COLUMNS)

# -----------------------------
# Optimizer (max-min gap with deterministic tie-break)
# -----------------------------

def optimize(left_df: pd.DataFrame):
    # validate rows
    errs = row_errors(left_df)
    if any(errs):
        # return errors back
        return None, errs

    exams = []
    for _, r in left_df.iterrows():
        exams.append({
            "module": str(r["subject"]),
            "slots": [pd.to_datetime(r["exam1"]).to_pydatetime(),
                      pd.to_datetime(r["exam2"]).to_pydatetime()]
        })

    n = len(exams)
    model = cp_model.CpModel()
    x = [model.NewIntVar(0, 1, f"x_{i}") for i in range(n)]
    ts_mat = [[int(s.timestamp()) for s in ex["slots"]] for ex in exams]

    chosen = []
    for i in range(n):
        tvar = model.NewIntVar(0, int(1e12), f"time_{i}")
        model.AddElement(x[i], ts_mat[i], tvar)
        chosen.append(tvar)

    min_gap = model.NewIntVar(0, int(1e12), "min_gap")
    for i in range(n):
        for j in range(i+1, n):
            diff = model.NewIntVar(0, int(1e12), f"diff_{i}_{j}")
            model.AddAbsEquality(diff, chosen[i] - chosen[j])
            model.Add(diff >= min_gap)

    model.Maximize(min_gap)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, pd.Series(["no solution"]*n)

    rows = []
    for i in range(n):
        idx = solver.Value(x[i])
        dt = exams[i]["slots"][idx]
        rows.append({"subject": exams[i]["module"], "optimal_date": dt, "slot": idx + 1})
    right_df = pd.DataFrame(rows, columns=RIGHT_COLUMNS)
    return right_df, pd.Series([""]*n)

# -----------------------------
# Feasible subset (remove overlaps)
# -----------------------------

def extract_feasible_subset(right_df: pd.DataFrame, duration_minutes: int = EVENT_DURATION_MINUTES):
    if right_df.empty:
        return right_df.copy(), right_df.copy()
    dur = timedelta(minutes=duration_minutes)
    intervals = []
    for i, r in right_df.iterrows():
        s = pd.to_datetime(r["optimal_date"]).to_pydatetime()
        e = s + dur
        intervals.append((e, s, i))
    intervals.sort()
    selected_idx = []
    last_end = None
    for e, s, i in intervals:
        if last_end is None or s >= last_end:
            selected_idx.append(i)
            last_end = e
    selected = right_df.loc[selected_idx].reset_index(drop=True)
    dropped = right_df.drop(index=selected_idx).reset_index(drop=True)
    return selected, dropped

def explain_drops(selected_df: pd.DataFrame, dropped_df: pd.DataFrame,
                  duration_minutes: int = EVENT_DURATION_MINUTES) -> pd.DataFrame:
    if dropped_df.empty:
        return pd.DataFrame(columns=["subject", "overlaps_with"])
    dur = timedelta(minutes=duration_minutes)
    sel = []
    for _, r in selected_df.iterrows():
        s = pd.to_datetime(r["optimal_date"]).to_pydatetime()
        sel.append((r["subject"], s, s + dur))
    rows = []
    for _, r in dropped_df.iterrows():
        s = pd.to_datetime(r["optimal_date"]).to_pydatetime()
        e = s + dur
        overlaps_with = [name for (name, ss, ee) in sel if max(s, ss) < min(e, ee)]
        rows.append({"subject": r["subject"], "overlaps_with": ", ".join(overlaps_with) if overlaps_with else "-"})
    return pd.DataFrame(rows, columns=["subject", "overlaps_with"])

def compute_min_start_gap_td(df: pd.DataFrame) -> timedelta | None:
    if df is None or df.empty or len(df) < 2:
        return None
    starts = sorted(pd.to_datetime(df["optimal_date"]).tolist())
    diffs = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
    return min(diffs) if diffs else None

# -----------------------------
# Exporters (localized metadata)
# -----------------------------

def build_results_csv(right_df: pd.DataFrame, min_gap_td: timedelta | None) -> bytes:
    buf = io.StringIO()
    gap_txt = format_gap_text(min_gap_td or timedelta(0), st.session_state.get("lang", "en"))
    meta = t("csv_meta", gap=gap_txt)
    buf.write(meta)
    right_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def build_ics(right_df: pd.DataFrame, *,
              calendar_name=None,
              tz_name="Europe/Berlin",
              duration_minutes: int = EVENT_DURATION_MINUTES,
              alarms=(60, 15)) -> bytes:
    tz = timezone(tz_name)
    cal = Calendar()
    cal.add("prodid", "-//Exam Scheduler//")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", calendar_name or t("calendar_name"))
    cal.add("x-wr-timezone", tz_name)
    for _, r in right_df.iterrows():
        start_dt = r["optimal_date"]
        if start_dt.tzinfo is None:
            start_dt = tz.localize(start_dt)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        ev = Event()
        ev.add("uid", f"{uuid4()}@exam-scheduler")
        ev.add("summary", r["subject"])
        ev.add("dtstart", start_dt)
        ev.add("dtend", end_dt)
        ev.add("description", t("ics_desc", subject=r["subject"]))
        for m in alarms:
            al = Alarm()
            al.add("action", "DISPLAY")
            al.add("description", t("ics_alarm", subject=r["subject"]))
            al.add("trigger", timedelta(minutes=-int(m)))
            ev.add_component(al)
        cal.add_component(ev)
    return cal.to_ical()

# -----------------------------
# Dialogs (titles must be static at decorate-time → two variants)
# -----------------------------

@st.dialog("Add subject")
def add_subject_dialog_en():
    _add_subject_body()

@st.dialog("Fach hinzufügen")
def add_subject_dialog_de():
    _add_subject_body()

def _add_subject_body():
    def _init_defaults():
        if "add_subj" not in st.session_state:
            st.session_state.add_subj = ""
        if "add_d1" not in st.session_state:
            st.session_state.add_d1 = datetime.now().date()
        if "add_t1" not in st.session_state:
            st.session_state.add_t1 = _rounded_now().time()
        if "add_d2" not in st.session_state:
            st.session_state.add_d2 = datetime.now().date()
        if "add_t2" not in st.session_state:
            st.session_state.add_t2 = (_rounded_now() + timedelta(hours=2)).time()
    _init_defaults()

    subj = st.text_input(t("dlg_subject"), key="add_subj")
    col_a, col_b = st.columns(2)
    with col_a:
        st.date_input(t("dlg_exam1_date"), key="add_d1")
        st.time_input(t("dlg_exam1_time"), key="add_t1", step=timedelta(minutes=15))
    with col_b:
        st.date_input(t("dlg_exam2_date"), key="add_d2")
        st.time_input(t("dlg_exam2_time"), key="add_t2", step=timedelta(minutes=15))

    csave, ccancel = st.columns(2)
    with csave:
        if st.button(t("dlg_save")):
            subj_val = st.session_state.get("add_subj", "").strip()
            if not subj_val:
                st.warning(t("dlg_warn_subject"))
                return
            dt1 = datetime.combine(st.session_state.add_d1, st.session_state.add_t1)
            dt2 = datetime.combine(st.session_state.add_d2, st.session_state.add_t2)
            new_row = pd.DataFrame([{"subject": subj_val, "exam1": dt1, "exam2": dt2}])
            st.session_state.left_df = pd.concat([st.session_state.left_df, new_row], ignore_index=True)
            st.session_state.right_df = pd.DataFrame(columns=RIGHT_COLUMNS)
            st.session_state.min_gap_td = None
            st.session_state.last_export = None
            st.session_state.dropped_df = pd.DataFrame()
            st.session_state.drop_explanations = pd.DataFrame()
            st.session_state.clash_count = 0
            st.session_state.show_add_dialog = False
            for k in ("add_subj", "add_d1", "add_t1", "add_d2", "add_t2"):
                st.session_state.pop(k, None)
            st.toast(t("toast_row_added"), icon="➕")
            st.rerun()
    with ccancel:
        if st.button(t("dlg_cancel")):
            st.session_state.show_add_dialog = False
            st.rerun()

@st.dialog("Randomize demo data")
def randomize_dialog_en():
    _randomize_body()

@st.dialog("Demo-Daten zufällig erstellen")
def randomize_dialog_de():
    _randomize_body()

def _randomize_body():
    n = st.slider(t("dlg_rand_how_many"), min_value=2, max_value=200, value=25, step=1)
    cgen, ccancel = st.columns(2)
    with cgen:
        if st.button(t("dlg_generate"), type="primary"):
            st.session_state.left_df = random_demo_df(n=n)
            st.session_state.right_df = pd.DataFrame(columns=RIGHT_COLUMNS)
            st.session_state.min_gap_td = None
            st.session_state.dropped_df = pd.DataFrame()
            st.session_state.drop_explanations = pd.DataFrame()
            st.session_state.last_export = None
            st.session_state.clash_count = 0
            st.session_state.show_random_dialog = False
            st.toast(t("toast_randomized", n=n), icon="🎲")
            st.rerun()
    with ccancel:
        if st.button(t("dlg_cancel")):
            st.session_state.show_random_dialog = False
            st.rerun()

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Exam Scheduler", page_icon="🗓️", layout="wide")

# Header row: title (left) + compact language switch (right)
head_left, head_right = st.columns([10, 2], vertical_alignment="center")
with head_left:
    st.markdown(f"# {t('title')}")

with head_right:
    # small spacer to align vertically with the title baseline
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Non-editable toggle (no typing), compact, inline with title
    lang_choice = st.radio(
        label="",                       # no label
        options=["EN", "DE"],
        horizontal=True,
        index=0 if st.session_state.lang == "en" else 1,
        label_visibility="collapsed",
    )

    new_lang = "en" if lang_choice == "EN" else "de"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

left_col, right_col = st.columns([1, 1])

# Top button rows
with left_col:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        uploaded = st.file_uploader(t("read_csv"), type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                uploaded.seek(0)
                tmp = pd.read_csv(uploaded, sep=None, engine="python")
                ok, msg = validate_csv(tmp)
                if not ok:
                    st.error(msg)
                else:
                    uploaded.seek(0)
                    st.session_state.left_df = df_from_csv(uploaded)
                    st.toast(t("toast_loaded", n=len(st.session_state.left_df)), icon="✅")
                    st.session_state.right_df = pd.DataFrame(columns=RIGHT_COLUMNS)
                    st.session_state.min_gap_td = None
                    st.session_state.dropped_df = pd.DataFrame()
                    st.session_state.drop_explanations = pd.DataFrame()
                    st.session_state.last_export = None
                    st.session_state.clash_count = 0
            except Exception as e:
                st.error(t("error_read_csv", err=e))
    with c2:
        if st.button(t("randomize"), use_container_width=True):
            st.session_state.show_random_dialog = True
    with c3:
        if st.button(t("add_date"), use_container_width=True):
            st.session_state.show_add_dialog = True

    if st.session_state.show_add_dialog:
        (add_subject_dialog_de if st.session_state.lang == "de" else add_subject_dialog_en)()
    if st.session_state.show_random_dialog:
        (randomize_dialog_de if st.session_state.lang == "de" else randomize_dialog_en)()

    st.subheader(t("input_header"))
    st.caption(t("input_caption"))
    left_display = st.session_state.left_df.copy()
    # Format datetime columns to hide seconds
    if not left_display.empty:
        left_display["exam1"] = pd.to_datetime(left_display["exam1"]).dt.strftime('%Y-%m-%d %H:%M')
        left_display["exam2"] = pd.to_datetime(left_display["exam2"]).dt.strftime('%Y-%m-%d %H:%M')
    left_display = left_display.rename(
        columns={"subject": t("col_subject"), "exam1": t("col_exam1"), "exam2": t("col_exam2")}
    )
    st.dataframe(left_display, use_container_width=True)

with right_col:
    c4, c5 = st.columns([1, 1])
    with c4:
        if st.button(t("optimize"), use_container_width=True):
            res, errs = optimize(st.session_state.left_df)
            if res is None:
                left = st.session_state.left_df.copy()
                left["errors"] = errs
                st.session_state.left_df = left
                st.error(t("error_rows_invalid"))
            else:
                feasible, dropped = extract_feasible_subset(res, duration_minutes=EVENT_DURATION_MINUTES)
                mg_td = compute_min_start_gap_td(feasible)
                st.session_state.right_df = feasible
                st.session_state.min_gap_td = mg_td
                st.session_state.dropped_df = dropped
                st.session_state.drop_explanations = explain_drops(
                    feasible, dropped, duration_minutes=EVENT_DURATION_MINUTES
                )
                st.session_state.clash_count = int(len(dropped) or 0)
                if dropped.empty:
                    st.toast(t("toast_optimization_complete"), icon="🧠")
    with c5:
        if st.button(t("export"), use_container_width=True, disabled=st.session_state.right_df.empty):
            csv_bytes = build_results_csv(st.session_state.right_df, st.session_state.min_gap_td)
            ics_bytes = build_ics(st.session_state.right_df)
            st.session_state.last_export = (csv_bytes, ics_bytes)
            st.toast(t("toast_export_ready"), icon="📦")

    if st.session_state.clash_count > 0:
        st.error(t("clash_alert", n=st.session_state.clash_count))
    if st.session_state.min_gap_td is not None:
        gap_txt = format_gap_text(st.session_state.min_gap_td, st.session_state.lang)
        st.success(t("min_gap_banner", gap=gap_txt))
    elif not st.session_state.right_df.empty:
        st.info(t("one_exam_banner"))

    st.subheader(t("optimized_header"))
    right_display = st.session_state.right_df.copy()
    # Format datetime column to hide seconds
    if not right_display.empty:
        right_display["optimal_date"] = pd.to_datetime(right_display["optimal_date"]).dt.strftime('%Y-%m-%d %H:%M')
    right_display = right_display.rename(
        columns={"subject": t("col_subject"), "optimal_date": t("col_sched_start"), "slot": t("col_slot")}
    )
    st.dataframe(right_display, use_container_width=True)

    if not st.session_state.drop_explanations.empty:
        with st.expander(t("exp_excluded")):
            drops_display = st.session_state.drop_explanations.rename(
                columns={"subject": t("col_subject"), "overlaps_with": t("col_overlaps_with")}
            )
            st.dataframe(drops_display, use_container_width=True)

    if st.session_state.last_export:
        csv_bytes, ics_bytes = st.session_state.last_export
        st.download_button("⬇️ Download results.csv", data=csv_bytes, file_name="results.csv", mime="text/csv")
        st.download_button("⬇️ Download exams_schedule.ics", data=ics_bytes, file_name="exams_schedule.ics", mime="text/calendar")
