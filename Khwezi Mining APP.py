
import difflib
import hashlib
import io
import json
import re
import secrets
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Prototype thresholds
TEMP_WARN, TEMP_CRIT = 80, 100          # degrees C
VIB_WARN, VIB_CRIT = 5, 8               # mm/s
FATIGUE_WARN, FATIGUE_CRIT = 7, 9       # 1-10 scale
PPE_SITE_TARGET = 90.0                  # % of workers compliant
DOWNTIME_LIMIT_PCT = 20.0               # % of scheduled time
LOW_AVAIL_PCT = 80.0                    # %
REPEATED_ALERTS = 3                     # alerts in the period
DUE_SOON_DAYS = 14                      # days before a service is "due"

LEVEL_ORDER = {"Normal": 0, "Warning": 1, "Critical": 2}

# Equipment condition monitoring (requirement e)
def classify_parameter(value, warn, crit):
    """Return Normal / Warning / Critical for one monitored parameter."""
    if value > crit:
        return "Critical"
    if value >= warn:
        return "Warning"
    return "Normal"


def equipment_condition(temp, vib):
    """Return (overall_status, message) for a temperature and vibration pair."""
    t = classify_parameter(temp, TEMP_WARN, TEMP_CRIT)
    v = classify_parameter(vib, VIB_WARN, VIB_CRIT)
    overall = max(t, v, key=LEVEL_ORDER.get)

    if overall == "Critical":
        parts = []
        if v == "Critical":
            parts.append("vibration")
        if t == "Critical":
            parts.append("temperature")
        msg = ("CRITICAL: Equipment " + " and ".join(parts) +
               " exceeds predefined threshold. Inspection required")
    elif overall == "Warning":
        parts = []
        if t == "Warning":
            parts.append("temperature")
        if v == "Warning":
            parts.append("vibration")
        msg = ("WARNING: Equipment " + " and ".join(parts) +
               " approaching predefined threshold.")
    else:
        msg = "NORMAL: Equipment operating within defined monitoring limits."
    return overall, msg

# Equipment maintenance monitoring (requirement f)
def availability(operating_time, downtime):
    """Availability = Operating / (Downtime + Operating) x 100."""
    total = operating_time + downtime
    return 0.0 if total <= 0 else operating_time / total * 100.0


def add_equipment_metrics(df, today=None):
    """Add condition, availability and maintenance columns to an equipment table."""
    df = df.copy()
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    if df.empty:
        for c in ["Availability %", "Downtime %", "Temp Status", "Vib Status",
                  "Condition", "Condition Message", "Days To Maintenance",
                  "Maintenance Category", "Excess Downtime", "Repeated Alerts",
                  "Low Availability", "Maintenance Flags"]:
            df[c] = []
        return df

    df["Availability %"] = [availability(o, d) for o, d in
                            zip(df["Operating Time (h)"], df["Downtime (h)"])]
    df["Availability %"] = df["Availability %"].round(1)
    df["Downtime %"] = (100 - df["Availability %"]).round(1)

    df["Temp Status"] = df["Temperature (C)"].apply(
        lambda x: classify_parameter(x, TEMP_WARN, TEMP_CRIT))
    df["Vib Status"] = df["Vibration (mm/s)"].apply(
        lambda x: classify_parameter(x, VIB_WARN, VIB_CRIT))
    cond = [equipment_condition(t, v) for t, v in
            zip(df["Temperature (C)"], df["Vibration (mm/s)"])]
    df["Condition"] = [c[0] for c in cond]
    df["Condition Message"] = [c[1] for c in cond]

    df["Days To Maintenance"] = (pd.to_datetime(df["Next Maintenance Due"]) - today).dt.days

    def category(row):
        if row["Maintenance Status"] == "Under Maintenance":
            return "Under Maintenance"
        if row["Days To Maintenance"] < 0:
            return "Overdue"
        if row["Days To Maintenance"] <= DUE_SOON_DAYS:
            return "Due Soon"
        return "OK"

    df["Maintenance Category"] = df.apply(category, axis=1)
    df["Excess Downtime"] = df["Downtime %"] > DOWNTIME_LIMIT_PCT
    df["Repeated Alerts"] = df["Alert Count"] >= REPEATED_ALERTS
    df["Low Availability"] = df["Availability %"] < LOW_AVAIL_PCT

    def flags(row):
        f = []
        if row["Maintenance Category"] in ("Due Soon", "Overdue", "Under Maintenance"):
            f.append(row["Maintenance Category"])
        if row["Excess Downtime"]:
            f.append("Excessive downtime")
        if row["Repeated Alerts"]:
            f.append("Repeated alerts")
        if row["Low Availability"]:
            f.append("Low availability")
        return "; ".join(f)

    df["Maintenance Flags"] = df.apply(flags, axis=1)
    return df


def maintenance_priority(df):
    """Rank equipment for maintenance (higher score = more urgent). Expects metrics columns."""
    d = df.copy()
    d["Priority Score"] = (
        (d["Condition"] == "Critical") * 3
        + (d["Condition"] == "Warning") * 1
        + (d["Maintenance Category"] == "Overdue") * 3
        + (d["Maintenance Category"] == "Due Soon") * 1
        + d["Low Availability"] * 2
        + d["Excess Downtime"] * 1
        + d["Repeated Alerts"] * 1
    )
    return d.sort_values("Priority Score", ascending=False)

# Risk assessment (requirement g)
def risk_score(likelihood, consequence):
    """RiskScore = Likelihood x Consequence (both 1-5)."""
    return int(likelihood) * int(consequence)


def classify_risk(score):
    if score <= 4:
        return "Low"
    if score <= 9:
        return "Medium"
    if score <= 16:
        return "High"
    return "Critical"

# Worker safety (requirement b)
def add_worker_metrics(df):
    """Add a risk level and a list of warnings to each worker record."""
    df = df.copy()
    if df.empty:
        df["Risk Points"], df["Risk Level"], df["Warnings"] = [], [], []
        return df
    points, levels, warns = [], [], []
    for _, r in df.iterrows():
        p, w = 0, []
        if r["PPE Compliance"] != "Compliant":
            p += 3
            w.append("PPE non-compliant")
        if r["Training Status"] == "Expired":
            p += 3
            w.append("Safety training expired")
        elif r["Training Status"] == "Pending":
            p += 1
            w.append("Safety training pending")
        if r["Fatigue Level"] >= FATIGUE_CRIT:
            p += 3
            w.append("Severe fatigue")
        elif r["Fatigue Level"] >= FATIGUE_WARN:
            p += 2
            w.append("High fatigue")
        if r["Near Misses"] >= 3:
            p += 2
            w.append("Repeated near misses")
        elif r["Near Misses"] >= 1:
            p += 1
        if r["Previous Incidents"] >= 2:
            p += 2
            w.append("Repeat incident history")
        elif r["Previous Incidents"] == 1:
            p += 1
        if r["Safety Observations"] >= 5:
            p += 1
            w.append("Many safety observations")
        points.append(p)
        levels.append("High" if p >= 7 else "Medium" if p >= 4 else "Low")
        warns.append("; ".join(w))
    df["Risk Points"], df["Risk Level"], df["Warnings"] = points, levels, warns
    return df


def ppe_compliance_pct(workers):
    if workers is None or workers.empty:
        return 0.0
    return round((workers["PPE Compliance"] == "Compliant").mean() * 100, 1)

# Alerts and notifications (requirement h)
SEV_ORDER = {"CRITICAL": 0, "WARNING": 1}
ACT_EQUIP = "Inspect equipment and initiate appropriate maintenance/safety procedures."
ACT_WORKER = "Supervisor to review worker status and take corrective action before next shift."
ACT_INCIDENT = "Complete investigation, confirm corrective action and notify management."
ACT_RISK = "Review controls and implement risk reduction measures."


def build_alerts(workers=None, incidents=None, equipment=None, risks=None):
    """
    Build a table of alerts from whichever datasets are supplied.
    Pass None for datasets the current user is not allowed to see.
    """
    rows = []

    def add(sev, cat, id_, param, value, status, action):
        rows.append({"Severity": sev, "Category": cat, "ID": id_, "Parameter": param,
                     "Value": value, "Status": status, "Recommended Action": action})

    if equipment is not None and not equipment.empty:
        eq = add_equipment_metrics(equipment)
        for _, r in eq.iterrows():
            i = r["Equipment ID"]
            if r["Temp Status"] == "Critical":
                add("CRITICAL", "Equipment", i, "Temperature", f"{r['Temperature (C)']:.1f} C",
                    "Above Critical Threshold", ACT_EQUIP)
            elif r["Temp Status"] == "Warning":
                add("WARNING", "Equipment", i, "Temperature", f"{r['Temperature (C)']:.1f} C",
                    "Approaching Critical Threshold", ACT_EQUIP)
            if r["Vib Status"] == "Critical":
                add("CRITICAL", "Equipment", i, "Vibration", f"{r['Vibration (mm/s)']:.1f} mm/s",
                    "Above Critical Threshold", ACT_EQUIP)
            elif r["Vib Status"] == "Warning":
                add("WARNING", "Equipment", i, "Vibration", f"{r['Vibration (mm/s)']:.1f} mm/s",
                    "Approaching Critical Threshold", ACT_EQUIP)
            if r["Excess Downtime"]:
                add("WARNING", "Equipment", i, "Downtime", f"{r['Downtime %']:.1f} %",
                    f"Above {DOWNTIME_LIMIT_PCT:.0f}% limit", "Investigate causes of downtime.")
            if r["Maintenance Category"] == "Overdue":
                add("WARNING", "Equipment", i, "Maintenance",
                    f"{abs(int(r['Days To Maintenance']))} days overdue", "Overdue maintenance",
                    "Schedule maintenance immediately.")
            if r["Repeated Alerts"]:
                add("WARNING", "Equipment", i, "Alert history", f"{int(r['Alert Count'])} alerts",
                    "Repeated alerts", "Carry out root-cause inspection.")
            if r["Low Availability"]:
                add("WARNING", "Equipment", i, "Availability", f"{r['Availability %']:.1f} %",
                    f"Below {LOW_AVAIL_PCT:.0f}% target", "Review reliability and maintenance plan.")

    if workers is not None and not workers.empty:
        wk = add_worker_metrics(workers)
        for _, r in wk.iterrows():
            i = r["Worker ID"]
            if r["PPE Compliance"] != "Compliant":
                add("WARNING", "Worker", i, "PPE compliance", "Non-compliant", "Low PPE compliance", ACT_WORKER)
            if r["Fatigue Level"] >= FATIGUE_CRIT:
                add("CRITICAL", "Worker", i, "Fatigue level", f"{int(r['Fatigue Level'])}/10",
                    "Above Critical Threshold", ACT_WORKER)
            elif r["Fatigue Level"] >= FATIGUE_WARN:
                add("WARNING", "Worker", i, "Fatigue level", f"{int(r['Fatigue Level'])}/10",
                    "High fatigue level", ACT_WORKER)
            if r["Risk Level"] == "High":
                add("WARNING", "Worker", i, "Safety risk level", f"{int(r['Risk Points'])} points",
                    "High-risk safety observation", ACT_WORKER)
        pct = ppe_compliance_pct(wk)
        if pct < PPE_SITE_TARGET:
            add("WARNING", "Site", "SITE", "PPE compliance", f"{pct:.1f} %",
                f"Below {PPE_SITE_TARGET:.0f}% target", "Launch PPE compliance campaign.")

    if incidents is not None and not incidents.empty:
        crit = incidents[(incidents["Severity"] == "Critical") & (incidents["Incident Status"] != "Closed")]
        for _, r in crit.iterrows():
            add("CRITICAL", "Incident", r["Incident ID"], "Incident severity", "Critical",
                f"Critical safety incident ({r['Incident Status']})", ACT_INCIDENT)

    if risks is not None and not risks.empty:
        for _, r in risks.iterrows():
            if r["Classification"] == "Critical":
                add("CRITICAL", "Risk", r["Risk ID"], "Risk score", f"{int(r['Risk Score'])}/25",
                    "Critical risk", ACT_RISK)
            elif r["Classification"] == "High":
                add("WARNING", "Risk", r["Risk ID"], "Risk score", f"{int(r['Risk Score'])}/25",
                    "High risk", ACT_RISK)

    cols = ["Severity", "Category", "ID", "Parameter", "Value", "Status", "Recommended Action"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["_o"] = out["Severity"].map(SEV_ORDER)
        out = out.sort_values(["_o", "Category", "ID"]).drop(columns="_o").reset_index(drop=True)
    return out


def format_alert(a):
    """Format one alert row like Figure 1 in the project brief."""
    bar = "=" * 53
    return (f"{bar}\nMINING MONITORING ALERT\n{bar}\n[{a['Severity']}]\n"
            f"{a['Category']} ID: {a['ID']}\nParameter: {a['Parameter']}\n"
            f"Value: {a['Value']}\nStatus: {a['Status']}\n"
            f"Recommended Action:\n {a['Recommended Action']}\n{bar}")


# 2. LOGIN AND ROLE-BASED ACCESS CONTROL

DATA_DIR = Path(__file__).parent / "app_data_3"
USERS_FILE = DATA_DIR / "users.json"

ROLES = ["Administrator", "Safety Officer", "Mining Engineer", "Maintenance Engineer", "Manager"]

# Permission matrix - mirrors Table 1 in the project brief.
PERMISSIONS = {
    "dashboard":    {"Administrator", "Safety Officer", "Mining Engineer", "Maintenance Engineer", "Manager"},
    "worker_safety": {"Administrator", "Safety Officer", "Mining Engineer", "Manager"},
    "incidents":    {"Administrator", "Safety Officer", "Mining Engineer", "Manager"},
    "equipment":    {"Administrator", "Mining Engineer", "Maintenance Engineer", "Manager"},
    "maintenance":  {"Administrator", "Mining Engineer", "Maintenance Engineer", "Manager"},
    "risk":         {"Administrator", "Safety Officer", "Mining Engineer", "Manager"},
    "reports":      {"Administrator", "Safety Officer", "Mining Engineer", "Maintenance Engineer", "Manager"},
    "manage_users": {"Administrator"},
}

DEMO_USERS = [
    ("admin",   "Admin@123",   "Administrator",        "Siphesihle Gumbi"),
    ("safety",  "Safety@123",  "Safety Officer",       "Walter Mabitsela"),
    ("mining",  "Mining@123",  "Mining Engineer",      "Minenhle Sayiya"),
    ("maint",   "Maint@123",   "Maintenance Engineer", "Thabisile Ntuli"),
    ("manager", "Manager@123", "Manager",              "Cat Matlala"),
]


def role_can(role, permission):
    """True if the role holds the permission."""
    return role in PERMISSIONS.get(permission, set())


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()


def load_users():
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    DATA_DIR.mkdir(exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _make_record(password, role, full_name):
    salt = secrets.token_hex(16)
    return {"salt": salt, "hash": _hash(password, salt), "role": role, "full_name": full_name}


def ensure_users():
    """Create the demo accounts on first run."""
    if USERS_FILE.exists():
        return
    users = {u: _make_record(p, r, n) for u, p, r, n in DEMO_USERS}
    save_users(users)


def authenticate(username, password):
    """Return a user dict if the credentials are valid, otherwise None."""
    users = load_users()
    rec = users.get(username.strip().lower())
    if rec and secrets.compare_digest(_hash(password, rec["salt"]), rec["hash"]):
        return {"username": username.strip().lower(), "role": rec["role"], "full_name": rec["full_name"]}
    return None


def add_user(username, password, role, full_name):
    """Return (ok, message)."""
    username = username.strip().lower()
    users = load_users()
    if not username.isalnum() or len(username) < 3:
        return False, "Username must be at least 3 letters/digits with no spaces or symbols."
    if username in users:
        return False, f"User '{username}' already exists."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if role not in ROLES:
        return False, "Invalid role."
    if not full_name.strip():
        return False, "Full name is required."
    users[username] = _make_record(password, role, full_name.strip())
    save_users(users)
    return True, f"User '{username}' created."


def delete_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        return True
    return False


def set_password(username, password):
    users = load_users()
    if username not in users:
        return False, "User not found."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    users[username].update(_make_record(password, users[username]["role"], users[username]["full_name"]))
    save_users(users)
    return True, "Password updated."


def set_role(username, role):
    users = load_users()
    if username in users and role in ROLES:
        users[username]["role"] = role
        save_users(users)
        return True
    return False

# 3. DATA STORAGE

DATA_DIR = Path(__file__).parent / "app_data_3"
FILES = {
    "workers": DATA_DIR / "workers.csv",
    "incidents": DATA_DIR / "incidents.csv",
    "equipment": DATA_DIR / "equipment.csv",
    "risks": DATA_DIR / "risks.csv",
}
DATE_COLS = {"incidents": ["Date"], "equipment": ["Last Maintenance Date", "Next Maintenance Due"]}

DEPARTMENTS = ["Underground Mining", "Open Pit", "Processing Plant",
               "Maintenance Workshop", "Engineering", "Logistics"]
SHIFTS = ["Day", "Afternoon", "Night"]


INCIDENT_TYPES = ["Near Miss", "Slip/Trip/Fall", "Fall of Ground", "Vehicle Collision",
                  "Equipment Damage", "Hand/Finger Injury", "Fire/Explosion", "Chemical Exposure"]
EQUIPMENT_TYPES = ["Haul Truck", "Loader", "Excavator", "Drilling Machine", "Bulldozer",
                   "Scraper Winch", "Crusher", "Conveyor"]

COLUMNS = {
    "workers": ["Worker ID", "Department", "Job Role", "Shift", "PPE Compliance", "Training Status",
                "Fatigue Level", "Safety Observations", "Near Misses", "Previous Incidents"],
    "incidents": ["Incident ID", "Date", "Time", "Shift", "Location", "Department", "Incident Type", "Severity",
                  "Injury Status", "Lost-Time Injury", "Cause", "Corrective Action", "Incident Status"],
    "equipment": ["Equipment ID", "Equipment Type", "Manufacturer", "Department", "Operating Hours",
                  "Operating Time (h)", "Downtime (h)", "Temperature (C)", "Vibration (mm/s)",
                  "Fuel Consumption (L/h)", "Brake Status", "Tyre Status", "Engine Status", "Maintenance Status",
                  "Last Maintenance Date", "Next Maintenance Due", "Alert Count"],
    "risks": ["Risk ID", "Department", "Hazard", "Likelihood", "Consequence", "Risk Score", "Classification"],
}
NUMERIC = {
    "workers": ["Fatigue Level", "Safety Observations", "Near Misses", "Previous Incidents"],
    "incidents": [],
    "equipment": ["Operating Hours", "Operating Time (h)", "Downtime (h)", "Temperature (C)", "Vibration (mm/s)",
                  "Fuel Consumption (L/h)", "Alert Count"],
    "risks": ["Likelihood", "Consequence", "Risk Score"],
}


def ensure_data():
    """First run: create the tables pre-filled with SAMPLE data."""
    DATA_DIR.mkdir(exist_ok=True)
    if not all(p.exists() for p in FILES.values()):
        generate_all()      # first run: fill the app with made-up sample records


def load(name):
    df = pd.read_csv(FILES[name])
    for c in DATE_COLS.get(name, []):
        df[c] = pd.to_datetime(df[c])
    for c in NUMERIC[name]:
        df[c] = pd.to_numeric(df[c])
    return df


def save(name, df):
    out = df.copy()
    for c in DATE_COLS.get(name, []):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out.to_csv(FILES[name], index=False)

# Sample data generation

def generate_all(seed=2026):
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(exist_ok=True)
    save("workers", _workers(rng))
    save("incidents", _incidents(rng))
    save("equipment", _equipment(rng))
    save("risks", _risks(rng))


def _workers(rng, n=250):
    roles = {
        "Underground Mining": ["Miner", "Drill Operator", "Shift Boss", "Rock Engineer"],
        "Open Pit": ["Truck Driver", "Excavator Operator", "Blaster", "Surveyor"],
        "Processing Plant": ["Plant Operator", "Metallurgist", "Technician"],
        "Maintenance Workshop": ["Fitter", "Electrician", "Boilermaker"],
        "Engineering": ["Mining Engineer", "Planner", "Geologist"],
        "Logistics": ["Dispatcher", "Store Controller", "Driver"],
    }
    rows = []
    for i in range(1, n + 1):
        dept = rng.choice(DEPARTMENTS, p=[.18, .30, .20, .14, .06, .12])
        shift = rng.choice(SHIFTS, p=[.38, .27, .35])
        fat_mean = {"Day": 3.5, "Afternoon": 4.5, "Night": 6.0}[shift]
        rows.append({
            "Worker ID": f"W{i:04d}",
            "Department": dept,
            "Job Role": rng.choice(roles[dept]),
            "Shift": shift,
            "PPE Compliance": "Compliant" if rng.random() < 0.93 else "Non-Compliant",
            "Training Status": rng.choice(["Valid", "Expired", "Pending"], p=[.82, .08, .10]),
            "Fatigue Level": int(np.clip(round(rng.normal(fat_mean, 1.9)), 1, 10)),
            "Safety Observations": int(rng.poisson(2.0)),
            "Near Misses": int(rng.poisson(0.9)),
            "Previous Incidents": int(rng.choice([0, 1, 2, 3], p=[.62, .26, .09, .03])),
        })
    return pd.DataFrame(rows)


def _incidents(rng, n=200):
    today = date.today()
    locs = {
        "Underground Mining": ["Level 12 Stope", "Level 8 Haulage", "Shaft 2 Station", "Decline Ramp"],
        "Open Pit": ["Pit North Bench", "Pit South Ramp", "Blast Area B", "Waste Dump"],
        "Processing Plant": ["Crusher House", "Mill Building", "Conveyor Gallery", "Thickener Area"],
        "Maintenance Workshop": ["Main Workshop", "Tyre Bay", "Wash Bay"],
        "Engineering": ["Engineering Office", "Survey Site"],
        "Logistics": ["Stores Yard", "Haul Road", "Weighbridge"],
    }
    types = ["Near Miss", "Slip/Trip/Fall", "Fall of Ground", "Vehicle Collision",
             "Equipment Damage", "Hand/Finger Injury", "Fire/Explosion", "Chemical Exposure"]
    causes = ["Poor housekeeping", "Fatigue", "PPE not worn", "Equipment failure", "Inadequate supervision",
              "Procedure not followed", "Poor ground conditions", "Poor visibility", "Training gap"]
    actions = ["Retraining of crew", "Equipment inspected and repaired", "Area barricaded and cleaned",
               "Procedure revised", "Additional supervision allocated", "PPE refresher issued",
               "Ground support installed", "Lighting improved"]
    rows = []
    for k in range(1, n + 1):
        d = today - timedelta(days=int(rng.integers(0, 180)))
        dept = rng.choice(DEPARTMENTS, p=[.30, .26, .16, .12, .04, .12])
        shift = rng.choice(SHIFTS, p=[.32, .28, .40])
        itype = rng.choice(types, p=[.28, .14, .08, .08, .12, .14, .05, .11])
        sev = rng.choice(["Low", "Medium", "High", "Critical"], p=[.42, .32, .19, .07])
        if itype in ("Near Miss", "Equipment Damage"):
            injury = "No Injury"
        else:
            injury = rng.choice(["First Aid", "Medical Treatment", "Lost Time Injury"], p=[.45, .35, .20])
        status = rng.choice(["Closed", "Under Investigation", "Open"], p=[.65, .22, .13])
        hr = {"Day": rng.integers(6, 14), "Afternoon": rng.integers(14, 22),
              "Night": rng.choice([22, 23, 0, 1, 2, 3, 4, 5])}[shift]
        rows.append({
            "Incident ID": f"INC-{k:04d}", "Date": d, "Time": f"{int(hr):02d}:{int(rng.integers(0, 60)):02d}",
            "Shift": shift, "Location": rng.choice(locs[dept]), "Department": dept,
            "Incident Type": itype, "Severity": sev, "Injury Status": injury,
            "Lost-Time Injury": "Yes" if injury == "Lost Time Injury" else "No",
            "Cause": rng.choice(causes), "Corrective Action": rng.choice(actions),
            "Incident Status": status,
        })
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def _equipment(rng):
    today = date.today()
    fleet = [
        ("Haul Truck", "TRK", 14, ["Caterpillar", "Komatsu", "Hitachi"], "Open Pit", True),
        ("Loader", "LDR", 6, ["Caterpillar", "Volvo", "Komatsu"], "Open Pit", True),
        ("Excavator", "EXC", 5, ["Hitachi", "Liebherr"], "Open Pit", True),
        ("Drilling Machine", "DRL", 5, ["Sandvik", "Epiroc"], "Underground Mining", True),
        ("Bulldozer", "DOZ", 4, ["Caterpillar", "Komatsu"], "Open Pit", True),
        ("Scraper Winch", "WCH", 3, ["Sandvik", "Joy Global"], "Underground Mining", False),
        ("Crusher", "CRS", 3, ["Metso", "FLSmidth"], "Processing Plant", False),
        ("Conveyor", "CNV", 5, ["Bateman", "Continental"], "Processing Plant", False),
    ]
    rows = []
    for etype, prefix, count, makers, dept, mobile in fleet:
        for i in range(1, count + 1):
            op = float(rng.uniform(470, 690))
            down = float(rng.uniform(8, 60) if rng.random() > 0.18 else rng.uniform(100, 200))
            last = today - timedelta(days=int(rng.integers(10, 110)))
            due = today + timedelta(days=int(rng.choice([-25, -12, -4, 3, 9, 20, 35, 50, 70],
                                                         p=[.04, .06, .06, .10, .10, .20, .20, .14, .10])))
            rows.append({
                "Equipment ID": f"{prefix}-{i:03d}", "Equipment Type": etype,
                "Manufacturer": rng.choice(makers), "Department": dept,
                "Operating Hours": int(rng.integers(2000, 32000)),
                "Operating Time (h)": round(op, 1), "Downtime (h)": round(down, 1),
                "Temperature (C)": round(float(np.clip(rng.normal(72, 13), 40, 115)), 1),
                "Vibration (mm/s)": round(float(np.clip(abs(rng.normal(3.6, 1.9)), 0.5, 10)), 1),
                "Fuel Consumption (L/h)": round(float(rng.uniform(25, 95)), 1) if mobile else 0.0,
                "Brake Status": rng.choice(["Good", "Worn", "Fault"], p=[.82, .14, .04]) if mobile else "N/A",
                "Tyre Status": rng.choice(["Good", "Worn", "Damaged"], p=[.80, .15, .05]) if mobile else "N/A",
                "Engine Status": rng.choice(["Good", "Warning", "Fault"], p=[.84, .12, .04]),
                "Maintenance Status": "Under Maintenance" if rng.random() < 0.09 else "Operational",
                "Last Maintenance Date": last, "Next Maintenance Due": due,
                "Alert Count": int(rng.poisson(1.2)),
            })
    df = pd.DataFrame(rows)
    # Match the example alert in Figure 1 of the project brief
    m = df["Equipment ID"] == "TRK-010"
    df.loc[m, ["Vibration (mm/s)", "Temperature (C)", "Alert Count"]] = [8.7, 96.5, 4]
    return df


def _risks(rng):
    hazards = [
        ("Underground Mining", "Fall of ground in stope"), ("Underground Mining", "Poor ventilation"),
        ("Underground Mining", "Seismic event"), ("Open Pit", "Highwall instability"),
        ("Open Pit", "Haul truck collision"), ("Open Pit", "Flyrock during blasting"),
        ("Processing Plant", "Conveyor entanglement"), ("Processing Plant", "Chemical exposure"),
        ("Processing Plant", "Dust inhalation"), ("Maintenance Workshop", "Unexpected energy release"),
        ("Maintenance Workshop", "Working at height"), ("Engineering", "Survey crew exposure to blasting"),
        ("Logistics", "Vehicle-pedestrian interaction"), ("Logistics", "Fuel spill at depot"),
        ("Open Pit", "Haul road washout"), ("Underground Mining", "Fire underground"),
    ]
    rows = []
    for i, (dept, hz) in enumerate(hazards, 1):
        l, c = int(rng.integers(1, 6)), int(rng.integers(2, 6))
        s = l * c
        cls = "Low" if s <= 4 else "Medium" if s <= 9 else "High" if s <= 16 else "Critical"
        rows.append({"Risk ID": f"RSK-{i:03d}", "Department": dept, "Hazard": hz,
                     "Likelihood": l, "Consequence": c, "Risk Score": s, "Classification": cls})
    return pd.DataFrame(rows)

# 4. INTERACTIVE DATA ANALYSIS

def _bar(df, x, y, title, **kw):
    fig = px.bar(df, x=x, y=y, title=title, text_auto=True, **kw)
    fig.update_layout(margin=dict(t=50, b=20))
    return fig


def _top(df, col, label):
    """Count rows per category in `col` and report the largest."""
    counts = df[col].value_counts().rename_axis(col).reset_index(name="Count")
    if counts.empty:
        return "No records in the selected period.", counts, None
    top = counts.iloc[0]
    ties = counts[counts["Count"] == top["Count"]][col].tolist()
    who = " and ".join(f"**{t}**" for t in ties)
    return (f"{who} {'have' if len(ties) > 1 else 'has'} the highest {label} "
            f"with **{int(top['Count'])}** records."), counts, _bar(counts, col, "Count", label.title())


def _top_equipment(eq, col, label, unit="", ascending=False):
    d = eq.sort_values(col, ascending=ascending).head(10)[["Equipment ID", "Equipment Type", col]]
    r = d.iloc[0]
    return (f"**{r['Equipment ID']}** ({r['Equipment Type']}) has the {label}: "
            f"**{r[col]}{unit}**."), d, _bar(d, "Equipment ID", col, f"Top 10 - {label}")

# Health and safety

def q_total_incidents(d):
    inc = d["incidents"]
    nm = (inc["Incident Type"] == "Near Miss").sum()
    t = inc.assign(Month=inc["Date"].dt.to_period("M").astype(str)).groupby("Month").size().reset_index(name="Incidents")
    return (f"**{len(inc)}** incidents were recorded in the selected period, of which **{nm}** were near misses "
            f"and **{(inc['Lost-Time Injury'] == 'Yes').sum()}** were lost-time injuries."), t, \
        px.line(t, x="Month", y="Incidents", markers=True, title="Incidents per month")


def q_dept_incidents(d):
    return _top(d["incidents"], "Department", "number of incidents")


def q_shift_incidents(d):
    return _top(d["incidents"], "Shift", "number of incidents")


def q_ppe(d):
    wk = d["workers"]
    pct = ppe_compliance_pct(wk)
    by = (wk.assign(c=wk["PPE Compliance"] == "Compliant").groupby("Department")["c"].mean() * 100).round(1)
    t = by.reset_index(name="PPE Compliance %")
    status = "meets" if pct >= PPE_SITE_TARGET else "is below"
    return (f"**{pct}%** of workers comply with PPE requirements, which {status} the prototype target of "
            f"{PPE_SITE_TARGET:.0f}%."), t, _bar(t, "Department", "PPE Compliance %", "PPE compliance by department")


def q_near_miss(d):
    inc = d["incidents"]
    nm = inc[inc["Incident Type"] == "Near Miss"]
    t = nm["Department"].value_counts().rename_axis("Department").reset_index(name="Near Misses")
    return (f"**{len(nm)}** near-miss incidents were recorded in the selected period "
            f"({len(nm) / max(len(inc), 1) * 100:.1f}% of all incidents)."), t, \
        _bar(t, "Department", "Near Misses", "Near misses by department")


def q_type(d):
    return _top(d["incidents"], "Incident Type", "frequency of incident type")


def q_high_crit(d):
    inc = d["incidents"]
    t = inc["Severity"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).astype(int) \
        .rename_axis("Severity").reset_index(name="Count")
    hc = int(t[t["Severity"].isin(["High", "Critical"])]["Count"].sum())
    return (f"**{hc}** incidents ({hc / max(len(inc), 1) * 100:.1f}%) are classified as High or Critical."), t, \
        _bar(t, "Severity", "Count", "Incidents by severity")


def q_dept_risk(d):
    rk = d["risks"]
    t = rk.groupby("Department").agg(**{"Average Risk Score": ("Risk Score", "mean"),
                                        "Highest Risk Score": ("Risk Score", "max"),
                                        "Risks Logged": ("Risk ID", "count")}).round(1) \
        .sort_values("Average Risk Score", ascending=False).reset_index()
    return (f"**{t.iloc[0]['Department']}** has the highest average risk score "
            f"(**{t.iloc[0]['Average Risk Score']}**) in the risk register."), t, \
        _bar(t, "Department", "Average Risk Score", "Average risk score by department")


def q_day_night(d):
    inc = d["incidents"]
    t = inc.groupby("Shift").agg(Incidents=("Incident ID", "count"),
                                 **{"High/Critical": ("Severity", lambda s: s.isin(["High", "Critical"]).sum())}) \
        .reindex(SHIFTS := ["Day", "Afternoon", "Night"]).fillna(0).astype(int).reset_index()
    day, night = int(t.loc[t["Shift"] == "Day", "Incidents"].iloc[0]), int(t.loc[t["Shift"] == "Night", "Incidents"].iloc[0])
    if day == 0:
        txt = f"Night shift recorded {night} incidents versus {day} on day shift."
    else:
        diff = (night - day) / day * 100
        txt = (f"Night shift recorded **{night}** incidents versus **{day}** on day shift - "
               f"**{abs(diff):.0f}% {'more' if diff > 0 else 'fewer'}** at night. "
               "This is an observed difference in the sample, not a statistical test.")
    return txt, t, px.bar(t.melt("Shift", var_name="Measure", value_name="Count"), x="Shift", y="Count",
                          color="Measure", barmode="group", title="Incidents by shift")


def q_attention_safety(d):
    al = d["alerts"]
    t = al[(al["Category"].isin(["Worker", "Incident", "Risk", "Site"])) & (al["Severity"] == "CRITICAL")]
    return (f"**{len(t)}** critical safety conditions need immediate attention."), t, None

# Equipment

def q_downtime(d):
    return _top_equipment(d["equipment"], "Downtime (h)", "highest downtime", " h")


def q_lowavail(d):
    return _top_equipment(d["equipment"], "Availability %", "lowest availability", " %", ascending=True)


def q_vib(d):
    return _top_equipment(d["equipment"], "Vibration (mm/s)", "highest vibration", " mm/s")


def q_temp(d):
    return _top_equipment(d["equipment"], "Temperature (C)", "highest operating temperature", " C")


def q_alerts(d):
    return _top_equipment(d["equipment"], "Alert Count", "most recorded alerts")


def q_needs_maint(d):
    eq = d["equipment"]
    t = eq[eq["Maintenance Category"].isin(["Due Soon", "Overdue"])][
        ["Equipment ID", "Equipment Type", "Maintenance Category", "Days To Maintenance", "Maintenance Flags"]]
    return f"**{len(t)}** items of equipment are due or overdue for maintenance.", t, None


def q_overdue(d):
    eq = d["equipment"]
    t = eq[eq["Maintenance Category"] == "Overdue"][["Equipment ID", "Equipment Type", "Days To Maintenance"]]
    return f"**{len(t)}** items of equipment are overdue for maintenance.", t, None


def q_avg_avail(d):
    eq = d["equipment"]
    t = eq.groupby("Equipment Type")["Availability %"].mean().round(1).reset_index()
    return (f"Average fleet availability is **{eq['Availability %'].mean():.1f}%**."), t, \
        _bar(t, "Equipment Type", "Availability %", "Average availability by equipment type")


def q_type_downtime(d):
    eq = d["equipment"]
    t = eq.groupby("Equipment Type")["Downtime (h)"].mean().round(1).sort_values(ascending=False).reset_index()
    return (f"**{t.iloc[0]['Equipment Type']}** has the highest average downtime "
            f"(**{t.iloc[0]['Downtime (h)']} h** per unit)."), t, \
        _bar(t, "Equipment Type", "Downtime (h)", "Average downtime by equipment type")


def q_inspect(d):
    eq = d["equipment"]
    t = eq[eq["Condition"] == "Critical"][["Equipment ID", "Equipment Type", "Temperature (C)",
                                           "Vibration (mm/s)", "Condition Message"]]
    return f"**{len(t)}** items of equipment require immediate inspection.", t, None

# Integrated

def q_shift_assoc(d):
    inc = d["incidents"]
    t = inc["Shift"].value_counts(normalize=True).mul(100).round(1).rename_axis("Shift").reset_index(name="Share of incidents %")
    top = t.iloc[0]
    return (f"**{top['Shift']}** shift accounts for **{top['Share of incidents %']}%** of incidents. "
            "If incidents were spread evenly across three shifts, each would be about 33%."), t, \
        px.pie(t, names="Shift", values="Share of incidents %", title="Share of incidents by shift")


def q_dept_both(d):
    eq, rk = d["equipment"], d["risks"]
    a = rk.groupby("Department")["Risk Score"].mean().round(1)
    b = eq.groupby("Department")["Downtime %"].mean().round(1)
    t = pd.concat([a, b], axis=1, keys=["Avg Risk Score", "Avg Downtime %"]).dropna().reset_index()
    t["Both High"] = (t["Avg Risk Score"] >= t["Avg Risk Score"].median()) & (t["Avg Downtime %"] >= t["Avg Downtime %"].median())
    names = ", ".join(t[t["Both High"]]["Department"]) or "none"
    fig = px.scatter(t, x="Avg Risk Score", y="Avg Downtime %", text="Department", color="Both High",
                     title="Risk vs equipment downtime by department")
    fig.update_traces(textposition="top center")
    return f"Departments above the median on both risk and downtime: **{names}**.", t, fig


def q_equip_alerts(d):
    al = d["alerts"]
    t = al[al["Category"] == "Equipment"].groupby("ID").size().sort_values(ascending=False).head(10) \
        .reset_index(name="Alerts")
    if t.empty:
        return "No active equipment alerts.", t, None
    return f"**{t.iloc[0]['ID']}** contributes the most active alerts (**{t.iloc[0]['Alerts']}**).", t, \
        _bar(t, "ID", "Alerts", "Active alerts per equipment")


def q_priority(d):
    p = maintenance_priority(d["equipment"]).head(10)[
        ["Equipment ID", "Equipment Type", "Condition", "Maintenance Category", "Availability %", "Priority Score"]]
    return (f"**{p.iloc[0]['Equipment ID']}** should receive maintenance priority "
            f"(score {p.iloc[0]['Priority Score']}). Score = weighted sum of condition, overdue/due status, "
            "availability, downtime and repeated alerts."), p, _bar(p, "Equipment ID", "Priority Score", "Maintenance priority")


def q_causes(d):
    return _top(d["incidents"], "Cause", "frequency of incident cause")


def q_indicators(d):
    wk, inc = d["workers"], d["incidents"]
    ppe = ppe_compliance_pct(wk)
    hc = inc["Severity"].isin(["High", "Critical"]).mean() * 100
    fat = (wk["Fatigue Level"] >= FATIGUE_WARN).mean() * 100
    exp = (wk["Training Status"] == "Expired").mean() * 100
    open_crit = ((inc["Severity"] == "Critical") & (inc["Incident Status"] != "Closed")).sum()
    t = pd.DataFrame([
        ["PPE compliance %", round(ppe, 1), f">= {PPE_SITE_TARGET:.0f}", ppe < PPE_SITE_TARGET],
        ["High/Critical incidents %", round(hc, 1), "<= 20", hc > 20],
        ["Workers with high fatigue %", round(fat, 1), "<= 10", fat > 10],
        ["Workers with expired training %", round(exp, 1), "<= 5", exp > 5],
        ["Open critical incidents", int(open_crit), "0", open_crit > 0],
    ], columns=["Indicator", "Value", "Target", "Needs intervention"])
    n = int(t["Needs intervention"].sum())
    return f"**{n}** of {len(t)} safety indicators require management intervention (targets are prototype values).", t, None


def q_normal_pct(d):
    eq = d["equipment"]
    t = eq["Condition"].value_counts().reindex(["Normal", "Warning", "Critical"]).fillna(0).astype(int) \
        .rename_axis("Condition").reset_index(name="Count")
    return (f"**{(eq['Condition'] == 'Normal').mean() * 100:.1f}%** of monitored equipment is operating normally."), t, \
        px.pie(t, names="Condition", values="Count", title="Equipment condition", color="Condition",
               color_discrete_map={"Normal": "#2e7d32", "Warning": "#f9a825", "Critical": "#c62828"})


def q_crit_alerts(d):
    al = d["alerts"]
    c = al[al["Severity"] == "CRITICAL"]
    t = c["Category"].value_counts().rename_axis("Category").reset_index(name="Critical Alerts")
    return f"**{len(c)}** critical alerts are currently active.", t, \
        _bar(t, "Category", "Critical Alerts", "Active critical alerts by category") if not t.empty else None


def q_attention_areas(d):
    inc, eq = d["incidents"], d["equipment"]
    a = inc.groupby("Department").agg(Incidents=("Incident ID", "count"),
                                      **{"High/Critical": ("Severity", lambda s: s.isin(["High", "Critical"]).sum())})
    b = eq.groupby("Department").agg(**{"Equipment Alerts": ("Alert Count", "sum"),
                                        "Avg Downtime %": ("Downtime %", "mean"),
                                        "Critical Equipment": ("Condition", lambda s: (s == "Critical").sum())}).round(1)
    t = a.join(b, how="outer").fillna(0)
    t["Attention Rank Score"] = t.rank(ascending=False).sum(axis=1).rsub(t.rank(ascending=False).sum(axis=1).max() + 1).round(1)
    t = t.sort_values("Attention Rank Score", ascending=False).reset_index()
    return (f"**{t.iloc[0]['Department']}** requires the greatest attention, ranking highest on a combined score of "
            "incidents, equipment alerts, downtime and critical equipment."), t, \
        _bar(t, "Department", "Attention Rank Score", "Combined attention score by department")


def q_recommend(d):
    recs = []
    eq, inc, wk = d["equipment"], d["incidents"], d["workers"]
    crit = eq[eq["Condition"] == "Critical"]["Equipment ID"].tolist()
    if crit:
        recs.append(f"Inspect critical-condition equipment immediately: {', '.join(crit)}.")
    over = eq[eq["Maintenance Category"] == "Overdue"]["Equipment ID"].tolist()
    if over:
        recs.append(f"Schedule overdue maintenance: {', '.join(over)}.")
    ppe = ppe_compliance_pct(wk)
    if ppe < PPE_SITE_TARGET:
        recs.append(f"PPE compliance is {ppe}% - run a PPE campaign and supervisor spot checks.")
    top_shift = inc["Shift"].value_counts()
    if not top_shift.empty:
        recs.append(f"The {top_shift.index[0]} shift records the most incidents - review supervision and fatigue management.")
    top_cause = inc["Cause"].value_counts()
    if not top_cause.empty:
        recs.append(f"Most common incident cause is '{top_cause.index[0]}' - target it with a corrective programme.")
    fat = int((wk["Fatigue Level"] >= FATIGUE_CRIT).sum())
    if fat:
        recs.append(f"{fat} workers report severe fatigue - consider rest-period or roster changes.")
    if not recs:
        recs.append("No major issues detected. Continue routine monitoring.")
    t = pd.DataFrame({"#": range(1, len(recs) + 1), "Recommendation": recs})
    return "Recommendations generated from the current monitoring results:\n\n" + "\n".join(
        f"{i}. {r}" for i, r in enumerate(recs, 1)), t, None


# data needed: keys of datasets; "alerts" is always available
QUESTIONS = {
    "Health and Safety": {
        "How many safety incidents occurred during the selected period?": ({"incidents"}, q_total_incidents),
        "Which department has the highest number of incidents?": ({"incidents"}, q_dept_incidents),
        "Which shift records the most incidents?": ({"incidents"}, q_shift_incidents),
        "What percentage of workers comply with PPE requirements?": ({"workers"}, q_ppe),
        "How many near misses have been recorded?": ({"incidents"}, q_near_miss),
        "Which incident types occur most frequently?": ({"incidents"}, q_type),
        "How many incidents are classified as high or critical?": ({"incidents"}, q_high_crit),
        "Which departments have the highest safety risk?": ({"risks"}, q_dept_risk),
        "Is there an observable difference in incidents between day and night shifts?": ({"incidents"}, q_day_night),
        "Which safety conditions require immediate attention?": ({"alerts"}, q_attention_safety),
    },
    "Equipment": {
        "Which equipment has the highest downtime?": ({"equipment"}, q_downtime),
        "Which equipment has the lowest availability?": ({"equipment"}, q_lowavail),
        "Which equipment has the highest vibration?": ({"equipment"}, q_vib),
        "Which equipment has the highest operating temperature?": ({"equipment"}, q_temp),
        "Which equipment has generated the most alerts?": ({"equipment"}, q_alerts),
        "Which equipment requires maintenance?": ({"equipment"}, q_needs_maint),
        "Which equipment is overdue for maintenance?": ({"equipment"}, q_overdue),
        "What is the average equipment availability?": ({"equipment"}, q_avg_avail),
        "Which equipment type has the highest downtime?": ({"equipment"}, q_type_downtime),
        "Which equipment requires immediate inspection?": ({"equipment"}, q_inspect),
    },
    "Integrated Safety and Equipment": {
        "Are safety incidents associated with particular shifts?": ({"incidents"}, q_shift_assoc),
        "Which departments have both high safety risk and high equipment downtime?": ({"equipment", "risks"}, q_dept_both),
        "Which equipment contributes to the greatest number of alerts?": ({"equipment"}, q_equip_alerts),
        "Which equipment should receive maintenance priority?": ({"equipment"}, q_priority),
        "What are the most common causes of safety incidents?": ({"incidents"}, q_causes),
        "Which safety indicators require management intervention?": ({"workers", "incidents"}, q_indicators),
        "What percentage of monitored equipment is operating normally?": ({"equipment"}, q_normal_pct),
        "How many critical alerts are currently active?": ({"alerts"}, q_crit_alerts),
        "Which areas of the operation require the greatest attention?": ({"incidents", "equipment"}, q_attention_areas),
        "What recommendations can be generated from the monitoring results?": ({"incidents", "equipment", "workers"}, q_recommend),
    },
}


def available_questions(accessible):
    """Filter QUESTIONS down to those the user's data access can answer."""
    out = {}
    for cat, qs in QUESTIONS.items():
        ok = {q: v for q, v in qs.items() if (v[0] - {"alerts"}) <= accessible}
        if ok:
            out[cat] = ok
    return out

# 5. STREAMLIT APPLICATION

st.set_page_config(page_title="Shaft Sense", page_icon="👷👷‍♀️", layout="wide")
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

SEV_COLOURS = {"Normal": "#2e7d32", "Warning": "#f9a825", "Critical": "#c62828",
               "Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00"}

# page name
PAGES = {
    "Dashboard": "dashboard",
    "Worker Safety": "worker_safety",
    "Safety Incidents": "incidents",
    "Equipment Monitoring": "equipment",
    "Maintenance": "maintenance",
    "Risk Assessment": "risk",
    "Data Analysis": "dashboard",
    "Reports": "reports",
    "User Management": "manage_users",
}

st.markdown("""
<style>
 .block-container {padding-top: 1.5rem;}
 div[data-testid="stMetric"] {background:#f4f6f8; border-left:5px solid #b8860b; padding:10px 14px; border-radius:6px;}
 div[data-testid="stMetric"] * {color:#1f2933 !important;}
</style>
""", unsafe_allow_html=True)

# Small helpers

def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_fig(fig):
    if fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def can(user, key):
    return role_can(user["role"], key)


def accessible_data(user):
    """Which datasets this role is allowed to see (drives dashboards, alerts, analysis)."""
    keys = set()
    if can(user, "worker_safety"):
        keys.add("workers")
    if can(user, "incidents"):
        keys.add("incidents")
    if can(user, "equipment") or can(user, "maintenance"):
        keys.add("equipment")
    if can(user, "risk"):
        keys.add("risks")
    return keys


def load_for(user):
    """Load permitted datasets (with computed columns) into a dict."""
    keys = accessible_data(user)
    d = {}
    if "workers" in keys:
        d["workers"] = add_worker_metrics(load("workers"))
    if "incidents" in keys:
        d["incidents"] = load("incidents")
    if "equipment" in keys:
        d["equipment"] = add_equipment_metrics(load("equipment"))
    if "risks" in keys:
        d["risks"] = load("risks")
    d["alerts"] = build_alerts(d.get("workers"), d.get("incidents"), d.get("equipment"), d.get("risks"))
    return d


def next_id(series, prefix, width):
    nums = [int(m.group(1)) for v in series if (m := re.search(r"(\d+)$", str(v)))]
    return f"{prefix}{(max(nums) + 1 if nums else 1):0{width}d}"


def _mode(series):
    """Most common value of a column, or 'n/a' when there is no data."""
    vc = series.value_counts()
    return vc.index[0] if len(vc) else "n/a"


def status_badge(level, text):
    """Show a message with BOTH colour and a text label (accessibility)."""
    if level == "Critical":
        st.error(text)
    elif level == "Warning":
        st.warning(text)
    else:
        st.success(text)

# Theme: black background, white writing, tint + accent colour depend on the role

LOGIN_THEME = {"bg": "#000000", "side": "#0d0d0d", "accent": "#d4af37"}
ROLE_THEMES = {
    "Administrator":        {"bg": "#000000", "side": "#17130a", "accent": "#d4af37"},   # black + gold
    "Safety Officer":       {"bg": "#03100a", "side": "#08241a", "accent": "#2ecc71"},   # black-green
    "Mining Engineer":      {"bg": "#040a18", "side": "#0a1a36", "accent": "#4da3ff"},   # black-blue
    "Maintenance Engineer": {"bg": "#140a02", "side": "#2b1607", "accent": "#ff8c1a"},   # black-orange
    "Manager":              {"bg": "#10041b", "side": "#240a3a", "accent": "#b46bff"},   # black-purple
}
THEME_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:__BG__ !important; color:#ffffff !important;}
[data-testid="stHeader"] {background:__BG__ !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {background:__SIDE__ !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp li, .stApp label,
.stApp span, .stApp small, .stApp .katex, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {color:#ffffff !important;}
.stApp h1, .stApp h2 {border-bottom:2px solid __ACCENT__; padding-bottom:4px;}
div[data-testid="stMetric"] {background:#14141a !important; border-left:5px solid __ACCENT__ !important; padding:10px 14px; border-radius:6px;}
div[data-testid="stMetric"] * {color:#ffffff !important;}
input, textarea, [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {background:#16161a !important; color:#ffffff !important; border-color:#3a3a44 !important;}
[data-baseweb="popover"] *, [data-baseweb="menu"] * {background:#16161a; color:#ffffff !important;}
.stTabs [data-baseweb="tab"] {color:#ffffff !important;}
.stTabs [aria-selected="true"] {border-bottom:3px solid __ACCENT__ !important;}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {background:__ACCENT__ !important; color:#000000 !important; border:none !important; font-weight:600;}
.stButton > button *, .stFormSubmitButton > button *, .stDownloadButton > button * {color:#000000 !important;}
[data-testid="stExpander"] {border:1px solid #3a3a44 !important; background:#101015 !important;}
[data-testid="stAlert"] {color:#ffffff !important; background:#1c1c22 !important;}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {background:#4a1418 !important;}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {background:#4a3a0c !important;}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {background:#0f3a1c !important;}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {background:#10304f !important;}
</style>
"""


def apply_theme(theme):
    css = THEME_CSS.replace("__BG__", theme["bg"]).replace("__SIDE__", theme["side"]).replace("__ACCENT__", theme["accent"])
    st.markdown(css, unsafe_allow_html=True)
    BACKGROUND_URL = "https://tse2.mm.bing.net/th/id/OIP.Ec-9KvCd1iC9vWCcHu8BZwHaEJ?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"
    st.markdown('<style>[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {background:transparent !important;} '
                '.stApp {background-image: linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)), '
                'url("' + BACKGROUND_URL + '") !important; background-size: cover !important; '
                'background-position: center !important; background-attachment: fixed !important;}</style>',
                unsafe_allow_html=True)


def _write_theme_config():
    """Make Streamlit's own widgets (tables etc.) dark too, if the project has no theme yet."""
    try:
        cfg = Path(__file__).resolve().parent / ".streamlit" / "config.toml"
        if not cfg.exists():
            cfg.parent.mkdir(exist_ok=True)
            cfg.write_text('[theme]\nbase = "dark"\nbackgroundColor = "#000000"\n'
                           'secondaryBackgroundColor = "#16161a"\ntextColor = "#ffffff"\n')
    except Exception:
        pass

# Login

def login_page():
    st.title("Shaft Sense🤔💡")
    st.caption("Smart Safety For Stronger Mining")
    left, mid, _ = st.columns([1, 1, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Login ID")
            password = st.text_input("Password", type="password")
            go_btn = st.form_submit_button("Log in", type="primary")
        if go_btn:
            if not username or not password:
                st.error("Please enter both a username and a password.")
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Incorrect Login ID or password.")
        with st.expander("Demo accounts"):
            st.markdown("\n".join(f"- `{u}` / `{p}` - {r}" for u, p, r, _ in DEMO_USERS))



# Dashboard

def page_dashboard(user):
    st.header("Khwezi Mine Operation Dashboard")
    d = load_for(user)
    al = d["alerts"]

    if all(len(d[k]) == 0 for k in d if k != "alerts"):
        st.info("No records yet. Use the pages in the sidebar to add workers, incidents, equipment and risks - "
                "this dashboard fills in automatically.")
        return

    if "workers" in d:
        wk = d["workers"]
        c = st.columns(4)
        c[0].metric("Total workers", len(wk))
        c[1].metric("PPE compliance", f"{ppe_compliance_pct(wk)}%" if len(wk) else "n/a")
        c[2].metric("High-risk observations", int((wk["Risk Level"] == "High").sum()))
        c[3].metric("High fatigue workers", int((wk["Fatigue Level"] >= FATIGUE_WARN).sum()))
    if "incidents" in d:
        inc = d["incidents"]
        c = st.columns(4)
        c[0].metric("Total safety incidents", len(inc))
        c[1].metric("Near misses", int((inc["Incident Type"] == "Near Miss").sum()))
        c[2].metric("High/Critical incidents", int(inc["Severity"].isin(["High", "Critical"]).sum()))
        c[3].metric("Open critical incidents", int(((inc["Severity"] == "Critical") & (inc["Incident Status"] != "Closed")).sum()))
    if "equipment" in d:
        eq = d["equipment"]
        c = st.columns(5)
        c[0].metric("Total equipment", len(eq))
        c[1].metric("Equipment available", int((eq["Maintenance Status"] == "Operational").sum()))
        c[2].metric("Under maintenance", int((eq["Maintenance Status"] == "Under Maintenance").sum()))
        c[3].metric("Avg availability", f"{eq['Availability %'].mean():.1f}%" if len(eq) else "n/a")
        c[4].metric("Critical equipment alerts", int((eq["Condition"] == "Critical").sum()))

    tab1, tab2 = st.tabs(["Overview charts", f"Active alerts ({len(al)})"])
    with tab1:
        a, b = st.columns(2)
        if "incidents" in d and len(d["incidents"]):
            inc = d["incidents"].assign(Month=lambda x: x["Date"].dt.to_period("M").astype(str))
            m = inc.groupby("Month").size().reset_index(name="Incidents")
            with a:
                show_fig(px.line(m, x="Month", y="Incidents", markers=True, title="Incidents per month"))
        if "equipment" in d and len(d["equipment"]):
            cnt = d["equipment"]["Condition"].value_counts().rename_axis("Condition").reset_index(name="Count")
            with b:
                show_fig(px.pie(cnt, names="Condition", values="Count", title="Equipment condition",
                                color="Condition", color_discrete_map=SEV_COLOURS))
        if "workers" in d and len(d["workers"]):
            wk = d["workers"]
            r = wk["Risk Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
            r.columns = ["Risk Level", "Workers"]
            with a:
                show_fig(px.bar(r, x="Risk Level", y="Workers", color="Risk Level", color_discrete_map=SEV_COLOURS,
                                title="Workers by risk level", text_auto=True))
        if not al.empty:
            s = al.groupby(["Category", "Severity"]).size().reset_index(name="Alerts")
            with b:
                show_fig(px.bar(s, x="Category", y="Alerts", color="Severity", title="Active alerts",
                                color_discrete_map={"CRITICAL": "#c62828", "WARNING": "#f9a825"}, text_auto=True))
    with tab2:
        if al.empty:
            st.success("No active alerts.")
        else:
            c1, c2 = st.columns(2)
            sev = c1.multiselect("Severity", ["CRITICAL", "WARNING"], default=["CRITICAL", "WARNING"])
            cat = c2.multiselect("Category", sorted(al["Category"].unique()))
            view = al[al["Severity"].isin(sev)]
            if cat:
                view = view[view["Category"].isin(cat)]
            show_df(view)
            with st.expander("Show formatted alert notices (first 10)"):
                for _, r in view.head(10).iterrows():
                    st.code(format_alert(r), language=None)

# Worker safety

def page_workers(user):
    st.header("Worker Health and Safety Monitoring")
    wk = add_worker_metrics(load("workers"))
    t1, t2, t3 = st.tabs(["Records", "Add worker record", "Warnings"])

    with t1:
        c = st.columns(4)
        dep = c[0].multiselect("Department", sorted(wk["Department"].unique()))
        sh = c[1].multiselect("Shift", SHIFTS)
        rl = c[2].multiselect("Risk level", ["Low", "Medium", "High"])
        q = c[3].text_input("Search ID or job role")
        v = wk.copy()
        if dep:
            v = v[v["Department"].isin(dep)]
        if sh:
            v = v[v["Shift"].isin(sh)]
        if rl:
            v = v[v["Risk Level"].isin(rl)]
        if q:
            v = v[v["Worker ID"].str.contains(q, case=False) | v["Job Role"].str.contains(q, case=False)]
        st.caption(f"{len(v)} of {len(wk)} workers shown")
        if wk.empty:
            st.info("No worker records yet. Use the 'Add worker record' tab.")
        else:
            show_df(v.drop(columns=["Warnings"]))
            g = (wk.assign(c=wk["PPE Compliance"] == "Compliant").groupby("Department")["c"].mean() * 100).round(1).reset_index(name="PPE %")
            show_fig(px.bar(g, x="Department", y="PPE %", title="PPE compliance by department", text_auto=True))

    with t2:
        with st.form("add_worker", clear_on_submit=True):
            c = st.columns(3)
            wid = c[0].text_input("Worker ID (format W0001)")
            dept = c[1].selectbox("Department", DEPARTMENTS)
            role = c[2].text_input("Job role")
            c = st.columns(3)
            shift = c[0].selectbox("Shift", SHIFTS)
            ppe = c[1].selectbox("PPE compliance", ["Compliant", "Non-Compliant"])
            train = c[2].selectbox("Safety training status", ["Valid", "Expired", "Pending"])
            c = st.columns(4)
            fat = c[0].slider("Fatigue level (1-10)", 1, 10, 3)
            obs = c[1].number_input("Safety observations", 0, 100, 0)
            nm = c[2].number_input("Near misses", 0, 100, 0)
            prev = c[3].number_input("Previous incidents", 0, 100, 0)
            ok = st.form_submit_button("Save record", type="primary")
        if ok:
            wid = wid.strip().upper()
            if not re.fullmatch(r"W\d{4}", wid):
                st.error("Worker ID must look like W0001 (letter W then four digits).")
            elif wid in set(wk["Worker ID"]):
                st.error(f"Worker ID {wid} already exists.")
            elif not role.strip():
                st.error("Job role is required.")
            else:
                new = pd.DataFrame([{"Worker ID": wid, "Department": dept, "Job Role": role.strip(), "Shift": shift,
                                     "PPE Compliance": ppe, "Training Status": train, "Fatigue Level": fat,
                                     "Safety Observations": obs, "Near Misses": nm, "Previous Incidents": prev}])
                base = load("workers")
                save("workers", pd.concat([base, new], ignore_index=True))
                w = add_worker_metrics(new).iloc[0]
                st.success(f"Worker {wid} saved. Calculated risk level: {w['Risk Level']}.")
                if w["Warnings"]:
                    st.warning("Warnings: " + w["Warnings"])

    with t3:
        w = wk[wk["Warnings"] != ""].sort_values("Risk Points", ascending=False)
        st.caption(f"{len(w)} workers have at least one warning")
        show_df(w[["Worker ID", "Department", "Shift", "Risk Level", "Risk Points", "Warnings"]])

# Safety incidents

def page_incidents(user):
    st.header("Safety Incident Database")
    inc = load("incidents")
    t1, t2, t3 = st.tabs(["View / search / filter", "Add incident", "Trends and counts"])

    with t1:
        if inc.empty:
            st.info("No incidents recorded yet. Use the 'Add incident' tab.")
        else:
            c = st.columns(4)
            lo, hi = inc["Date"].min().date(), max(inc["Date"].max().date(), date.today())
            rng = c[0].date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
            dep = c[1].multiselect("Department", sorted(inc["Department"].unique()))
            sh = c[2].multiselect("Shift", SHIFTS)
            ty = c[3].multiselect("Incident type", sorted(inc["Incident Type"].unique()))
            c = st.columns(4)
            sv = c[0].multiselect("Severity", ["Low", "Medium", "High", "Critical"])
            stt = c[1].multiselect("Status", ["Open", "Under Investigation", "Closed"])
            lti = c[2].selectbox("Lost-time injury", ["Any", "Yes", "No"])
            q = c[3].text_input("Search (ID, location, cause)")
            v = inc.copy()
            if isinstance(rng, tuple) and len(rng) == 2:
                v = v[(v["Date"].dt.date >= rng[0]) & (v["Date"].dt.date <= rng[1])]
            for col, sel in [("Department", dep), ("Shift", sh), ("Incident Type", ty), ("Severity", sv), ("Incident Status", stt)]:
                if sel:
                    v = v[v[col].isin(sel)]
            if lti != "Any":
                v = v[v["Lost-Time Injury"] == lti]
            if q:
                m = v["Incident ID"].str.contains(q, case=False) | v["Location"].str.contains(q, case=False) | v["Cause"].str.contains(q, case=False)
                v = v[m]
            st.caption(f"{len(v)} of {len(inc)} incidents shown")
            show_df(v.assign(Date=v["Date"].dt.strftime("%Y-%m-%d")))
            st.download_button("Download filtered incidents (CSV)", v.to_csv(index=False), "incidents_filtered.csv", "text/csv")

    with t2:
        with st.form("add_incident", clear_on_submit=True):
            c = st.columns(4)
            d_ = c[0].date_input("Date", date.today(), max_value=date.today())
            t_ = c[1].time_input("Time", datetime.now().time().replace(second=0, microsecond=0))
            shift = c[2].selectbox("Shift", SHIFTS)
            dept = c[3].selectbox("Department", DEPARTMENTS)
            c = st.columns(4)
            loc = c[0].text_input("Location")
            itype = c[1].selectbox("Incident type", INCIDENT_TYPES)
            sev = c[2].selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            injury = c[3].selectbox("Injury status", ["No Injury", "First Aid", "Medical Treatment", "Lost Time Injury"])
            c = st.columns(3)
            cause = c[0].text_input("Cause")
            action = c[1].text_input("Corrective action")
            status = c[2].selectbox("Incident status", ["Open", "Under Investigation", "Closed"])
            ok = st.form_submit_button("Add incident", type="primary")
        if ok:
            if not loc.strip() or not cause.strip() or len(action.strip()) < 5:
                st.error("Location, cause and a corrective action (at least 5 characters) are required.")
            elif itype == "Near Miss" and injury != "No Injury":
                st.error("A near miss cannot have an injury. Choose 'No Injury' or change the incident type.")
            else:
                iid = next_id(inc["Incident ID"], "INC-", 4)
                new = pd.DataFrame([{"Incident ID": iid, "Date": pd.Timestamp(d_), "Time": t_.strftime("%H:%M"),
                                     "Shift": shift, "Location": loc.strip(), "Department": dept, "Incident Type": itype,
                                     "Severity": sev, "Injury Status": injury,
                                     "Lost-Time Injury": "Yes" if injury == "Lost Time Injury" else "No",
                                     "Cause": cause.strip(), "Corrective Action": action.strip(), "Incident Status": status}])
                save("incidents", pd.concat([inc, new], ignore_index=True))
                st.success(f"Incident {iid} recorded.")
                if sev in ("High", "Critical"):
                    st.warning(f"{sev.upper()} severity incident - notify management and start investigation.")

    with t3:
        if inc.empty:
            st.info("No incidents recorded yet. Use the 'Add incident' tab.")
        else:
            dim = st.selectbox("Count incidents by", ["Department", "Shift", "Incident Type", "Severity", "Cause",
                                                      "Location", "Injury Status", "Incident Status"])
            cnt = inc[dim].value_counts().rename_axis(dim).reset_index(name="Incidents")
            a, b = st.columns([1, 2])
            with a:
                show_df(cnt)
            with b:
                show_fig(px.bar(cnt, x=dim, y="Incidents", text_auto=True, title=f"Incidents by {dim.lower()}"))
            m = inc.assign(Month=inc["Date"].dt.to_period("M").astype(str)).groupby(["Month", "Severity"]).size().reset_index(name="Incidents")
            show_fig(px.line(m, x="Month", y="Incidents", color="Severity", markers=True, title="Monthly incident trend by severity"))

# Equipment monitoring

def page_equipment(user):
    st.header("Equipment Database and Condition Monitoring")
    raw = load("equipment")
    eq = add_equipment_metrics(raw)
    t1, t2, t3 = st.tabs(["Equipment database", "Condition monitor", "Add equipment"])

    with t1:
        if eq.empty:
            st.info("No equipment recorded yet. Use the 'Add equipment' tab.")
        else:
            c = st.columns(3)
            ty = c[0].multiselect("Equipment type", sorted(eq["Equipment Type"].unique()))
            co = c[1].multiselect("Condition", ["Normal", "Warning", "Critical"])
            q = c[2].text_input("Search ID or manufacturer")
            v = eq
            if ty:
                v = v[v["Equipment Type"].isin(ty)]
            if co:
                v = v[v["Condition"].isin(co)]
            if q:
                v = v[v["Equipment ID"].str.contains(q, case=False) | v["Manufacturer"].str.contains(q, case=False)]
            st.caption(f"{len(v)} of {len(eq)} units shown")
            cols = ["Equipment ID", "Equipment Type", "Manufacturer", "Department", "Operating Hours", "Temperature (C)",
                    "Vibration (mm/s)", "Fuel Consumption (L/h)", "Brake Status", "Tyre Status", "Engine Status",
                    "Maintenance Status", "Downtime (h)", "Availability %", "Condition"]
            show_df(v[cols])

    with t2:
        if eq.empty:
            st.info("No equipment recorded yet. Use the 'Add equipment' tab.")
        else:
            st.info("Thresholds are educational prototype values only (not manufacturer or mine safety limits). "
                    f"Temperature: warning {TEMP_WARN}-{TEMP_CRIT} C, critical > {TEMP_CRIT} C. "
                    f"Vibration: warning {VIB_WARN}-{VIB_CRIT} mm/s, critical > {VIB_CRIT} mm/s.")
            c = st.columns(3)
            c[0].metric("Normal", int((eq["Condition"] == "Normal").sum()))
            c[1].metric("Warning", int((eq["Condition"] == "Warning").sum()))
            c[2].metric("Critical", int((eq["Condition"] == "Critical").sum()))
            fig = px.scatter(eq, x="Temperature (C)", y="Vibration (mm/s)", color="Condition", hover_name="Equipment ID",
                             color_discrete_map=SEV_COLOURS, title="Temperature vs vibration")
            fig.add_vline(x=TEMP_WARN, line_dash="dot", line_color="#f9a825")
            fig.add_vline(x=TEMP_CRIT, line_dash="dash", line_color="#c62828")
            fig.add_hline(y=VIB_WARN, line_dash="dot", line_color="#f9a825")
            fig.add_hline(y=VIB_CRIT, line_dash="dash", line_color="#c62828")
            show_fig(fig)
            show_critical_first = eq.assign(_o=eq["Condition"].map(LEVEL_ORDER)).sort_values("_o", ascending=False)
            pick = st.radio("Show", ["Warning and Critical only", "All equipment"], horizontal=True)
            for _, r in show_critical_first.iterrows():
                if pick.startswith("Warning") and r["Condition"] == "Normal":
                    continue
                status_badge(r["Condition"], f"**{r['Equipment ID']}** ({r['Equipment Type']}) - {r['Condition Message']} "
                                             f"[{r['Temperature (C)']} C, {r['Vibration (mm/s)']} mm/s]")

    with t3:
        if not (can(user, "equipment") and user["role"] in ("Administrator", "Mining Engineer", "Maintenance Engineer")):
            st.info("Your role can view equipment data but not add new equipment.")
        else:
            with st.form("add_equipment", clear_on_submit=True):
                c = st.columns(4)
                eid = c[0].text_input("Equipment ID (e.g. TRK-013)")
                etype = c[1].selectbox("Equipment type", EQUIPMENT_TYPES)
                maker = c[2].text_input("Manufacturer")
                dept = c[3].selectbox("Department", DEPARTMENTS)
                c = st.columns(4)
                hrs = c[0].number_input("Operating hours (lifetime)", 0, 200000, 0)
                op = c[1].number_input("Operating time this period (h)", 0.0, 744.0, 500.0)
                down = c[2].number_input("Downtime this period (h)", 0.0, 744.0, 20.0)
                fuel = c[3].number_input("Fuel consumption (L/h)", 0.0, 500.0, 0.0)
                c = st.columns(4)
                temp = c[0].number_input("Temperature (C)", 0.0, 300.0, 70.0)
                vib = c[1].number_input("Vibration (mm/s)", 0.0, 50.0, 3.0)
                brake = c[2].selectbox("Brake status", ["Good", "Worn", "Fault", "N/A"])
                tyre = c[3].selectbox("Tyre status", ["Good", "Worn", "Damaged", "N/A"])
                c = st.columns(3)
                eng = c[0].selectbox("Engine status", ["Good", "Warning", "Fault"])
                last = c[1].date_input("Last maintenance", date.today(), max_value=date.today())
                due = c[2].date_input("Next maintenance due", date.today() + timedelta(days=90))
                ok = st.form_submit_button("Add equipment", type="primary")
            if ok:
                eid = eid.strip().upper()
                if not re.fullmatch(r"[A-Z]{3}-\d{3}", eid):
                    st.error("Equipment ID must look like TRK-013 (three letters, dash, three digits).")
                elif eid in set(raw["Equipment ID"]):
                    st.error(f"Equipment ID {eid} already exists.")
                elif not maker.strip():
                    st.error("Manufacturer is required.")
                elif op + down <= 0:
                    st.error("Operating time and downtime cannot both be zero.")
                elif due <= last:
                    st.error("Next maintenance due must be after the last maintenance date.")
                else:
                    new = pd.DataFrame([{
                        "Equipment ID": eid, "Equipment Type": etype, "Manufacturer": maker.strip(), "Department": dept,
                        "Operating Hours": hrs, "Operating Time (h)": op, "Downtime (h)": down,
                        "Temperature (C)": temp, "Vibration (mm/s)": vib, "Fuel Consumption (L/h)": fuel,
                        "Brake Status": brake, "Tyre Status": tyre, "Engine Status": eng,
                        "Maintenance Status": "Operational", "Last Maintenance Date": pd.Timestamp(last),
                        "Next Maintenance Due": pd.Timestamp(due), "Alert Count": 0}])
                    save("equipment", pd.concat([raw, new], ignore_index=True))
                    cond, msg = equipment_condition(temp, vib)
                    st.success(f"Equipment {eid} added.")
                    status_badge(cond, msg)

# Maintenance

def page_maintenance(user):
    st.header("Equipment Maintenance Monitoring")
    raw = load("equipment")
    eq = add_equipment_metrics(raw)
    st.latex(r"\text{Availability} = \frac{\text{Operating Time}}{\text{Downtime} + \text{Operating Time}} \times 100")

    if eq.empty:
        st.info("No equipment recorded yet. Add equipment on the Equipment Monitoring page first.")
        return

    c = st.columns(6)
    c[0].metric("Due soon", int((eq["Maintenance Category"] == "Due Soon").sum()))
    c[1].metric("Overdue", int((eq["Maintenance Category"] == "Overdue").sum()))
    c[2].metric("Under maintenance", int((eq["Maintenance Category"] == "Under Maintenance").sum()))
    c[3].metric("Excessive downtime", int(eq["Excess Downtime"].sum()))
    c[4].metric("Repeated alerts", int(eq["Repeated Alerts"].sum()))
    c[5].metric("Low availability", int(eq["Low Availability"].sum()))
    st.caption(f"Rules: due = within {DUE_SOON_DAYS} days; excessive downtime > {DOWNTIME_LIMIT_PCT:.0f}%; "
               f"repeated alerts >= {REPEATED_ALERTS}; low availability < {LOW_AVAIL_PCT:.0f}% (prototype values).")

    tabs = st.tabs(["Due soon", "Overdue", "Under maintenance", "Excessive downtime", "Repeated alerts",
                    "Low availability", "Availability chart", "Update status"])
    base = ["Equipment ID", "Equipment Type", "Department", "Next Maintenance Due", "Days To Maintenance",
            "Availability %", "Downtime %", "Alert Count"]
    filters = [eq["Maintenance Category"] == "Due Soon", eq["Maintenance Category"] == "Overdue",
               eq["Maintenance Category"] == "Under Maintenance", eq["Excess Downtime"],
               eq["Repeated Alerts"], eq["Low Availability"]]
    for tab, f in zip(tabs[:6], filters):
        with tab:
            v = eq[f][base].assign(**{"Next Maintenance Due": lambda x: x["Next Maintenance Due"].dt.strftime("%Y-%m-%d")})
            if v.empty:
                st.success("No equipment in this category.")
            else:
                show_df(v)
    with tabs[6]:
        s = eq.sort_values("Availability %")
        fig = px.bar(s, x="Equipment ID", y="Availability %", color="Equipment Type", title="Availability by equipment")
        fig.add_hline(y=LOW_AVAIL_PCT, line_dash="dash", line_color="#c62828", annotation_text="Low availability limit")
        show_fig(fig)
    with tabs[7]:
        if user["role"] not in ("Administrator", "Maintenance Engineer", "Mining Engineer"):
            st.info("Your role can view maintenance data but not update it.")
        else:
            with st.form("upd_maint"):
                eid = st.selectbox("Equipment", raw["Equipment ID"])
                act = st.selectbox("Action", ["Mark as under maintenance", "Mark as operational",
                                              "Record maintenance completed (resets next due +90 days and alert count)"])
                ok = st.form_submit_button("Apply", type="primary")
            if ok:
                m = raw["Equipment ID"] == eid
                if act.startswith("Mark as under"):
                    raw.loc[m, "Maintenance Status"] = "Under Maintenance"
                elif act.startswith("Mark as operational"):
                    raw.loc[m, "Maintenance Status"] = "Operational"
                else:
                    raw.loc[m, ["Maintenance Status", "Alert Count"]] = ["Operational", 0]
                    raw.loc[m, "Last Maintenance Date"] = pd.Timestamp.today().normalize()
                    raw.loc[m, "Next Maintenance Due"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=90)
                save("equipment", raw)
                st.success(f"{eid} updated. Refresh the page (or switch tabs) to see recalculated lists.")

# Risk assessment

def page_risk(user):
    st.header("Risk Assessment")
    rk = load("risks")
    t1, t2, t3 = st.tabs(["Risk calculator", "Risk register", "Risk matrix"])

    with t1:
        st.latex(r"\text{Risk Score} = \text{Likelihood} \times \text{Consequence}")
        c = st.columns(2)
        L = c[0].slider("Likelihood (1 = rare, 5 = almost certain)", 1, 5, 3)
        C = c[1].slider("Consequence (1 = insignificant, 5 = catastrophic)", 1, 5, 3)
        s = risk_score(L, C)
        cls = classify_risk(s)
        st.metric("Risk score", f"{s} / 25", cls)
        if cls == "Critical":
            st.error(f"CRITICAL RISK ({s}): stop work in the affected area and escalate to management immediately.")
        elif cls == "High":
            st.warning(f"HIGH RISK ({s}): urgent action required - implement additional controls.")
        elif cls == "Medium":
            st.info(f"MEDIUM RISK ({s}): manage with existing controls and monitor.")
        else:
            st.success(f"LOW RISK ({s}): acceptable - routine monitoring.")
        with st.form("add_risk", clear_on_submit=True):
            dept = st.selectbox("Department", DEPARTMENTS)
            hz = st.text_input("Hazard description")
            ok = st.form_submit_button("Add this risk (uses the sliders above) to register")
        if ok:
            if len(hz.strip()) < 5:
                st.error("Please describe the hazard (at least 5 characters).")
            else:
                rid = next_id(rk["Risk ID"], "RSK-", 3)
                new = pd.DataFrame([{"Risk ID": rid, "Department": dept, "Hazard": hz.strip(), "Likelihood": L,
                                     "Consequence": C, "Risk Score": s, "Classification": cls}])
                save("risks", pd.concat([rk, new], ignore_index=True))
                st.success(f"{rid} added to the risk register ({cls}).")

    with t2:
        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        show_df(rk.assign(_o=rk["Classification"].map(order)).sort_values(["_o", "Risk Score"], ascending=[True, False]).drop(columns="_o"))

    with t3:
        grid = [[l * c for c in range(1, 6)] for l in range(5, 0, -1)]
        counts = [[int(((rk["Likelihood"] == l) & (rk["Consequence"] == c)).sum()) for c in range(1, 6)] for l in range(5, 0, -1)]
        text = [[f"{grid[i][j]}<br>({counts[i][j]} risks)" for j in range(5)] for i in range(5)]
        fig = go.Figure(go.Heatmap(z=grid, x=[1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1], text=text, texttemplate="%{text}",
                                   colorscale=[[0, "#2e7d32"], [0.16, "#2e7d32"], [0.17, "#f9a825"], [0.36, "#f9a825"],
                                               [0.37, "#ef6c00"], [0.64, "#ef6c00"], [0.65, "#c62828"], [1, "#c62828"]],
                                   showscale=False))
        fig.update_layout(title="Risk matrix: cell score and number of registered risks",
                          xaxis_title="Consequence", yaxis_title="Likelihood", xaxis=dict(dtick=1), yaxis=dict(dtick=1))
        show_fig(fig)
        st.caption("Green = Low (1-4), amber = Medium (5-9), orange = High (10-16), red = Critical (17-25).")

# Data analysis

def page_analysis(user):
    st.header("Interactive Data Analysis")
    d = load_for(user)
    avail = available_questions(set(d) - {"alerts"})
    if not avail:
        st.info("Your role has no datasets available for analysis.")
        return
    if all(len(d[k]) == 0 for k in d if k != "alerts"):
        st.info("No records yet. Add some data first - the analysis questions are answered from your records.")
        return
    cat = st.radio("Topic", list(avail), horizontal=True)
    question = st.selectbox("Question", list(avail[cat]))

    if "incidents" in d:
        inc = d["incidents"]
        period = st.selectbox("Period for incident questions", ["All data", "Last 30 days", "Last 90 days", "Custom range"])
        if period == "Last 30 days":
            d["incidents"] = inc[inc["Date"] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
        elif period == "Last 90 days":
            d["incidents"] = inc[inc["Date"] >= pd.Timestamp.today() - pd.Timedelta(days=90)]
        elif period == "Custom range":
            r = st.date_input("Select dates", (inc["Date"].min().date() if inc["Date"].notna().any() else date.today(), date.today()))
            if isinstance(r, tuple) and len(r) == 2:
                d["incidents"] = inc[(inc["Date"].dt.date >= r[0]) & (inc["Date"].dt.date <= r[1])]
        if d["incidents"].empty and "incidents" in avail[cat][question][0]:
            st.warning("No incidents in the selected period.")
            return

    st.divider()
    needs, fn = avail[cat][question]
    missing = [k for k in needs if k != "alerts" and len(d.get(k, [])) == 0]
    if missing:
        st.info("This question needs records that have not been added yet: " + ", ".join(missing) + ".")
        return
    try:
        text, table, fig = fn(d)
    except (IndexError, KeyError, ValueError, ZeroDivisionError):
        st.info("Not enough data to answer this question yet. Add more records and try again.")
        return
    st.markdown(f"#### {question}")
    st.markdown(text)
    a, b = st.columns([1, 1])
    with a:
        if table is not None and len(table):
            show_df(table)
    with b:
        show_fig(fig)

# Reports

def build_report_text(user, d):
    lines = ["KHWEZI MINING - HEALTH, SAFETY AND EQUIPMENT MONITORING REPORT",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}   By: {user['full_name']} ({user['role']})", "=" * 60]
    if "workers" in d:
        wk = d["workers"]
        lines += ["", "WORKER SAFETY", f"- Total workers: {len(wk)}", f"- PPE compliance: {ppe_compliance_pct(wk)}%",
                  f"- High-risk workers: {(wk['Risk Level'] == 'High').sum()}",
                  f"- High fatigue workers: {(wk['Fatigue Level'] >= FATIGUE_WARN).sum()}"]
    if "incidents" in d:
        inc = d["incidents"]
        lines += ["", "SAFETY INCIDENTS", f"- Total incidents: {len(inc)}",
                  f"- Near misses: {(inc['Incident Type'] == 'Near Miss').sum()}",
                  f"- High/Critical: {inc['Severity'].isin(['High', 'Critical']).sum()}",
                  f"- Lost-time injuries: {(inc['Lost-Time Injury'] == 'Yes').sum()}",
                  f"- Most affected department: {_mode(inc['Department'])}",
                  f"- Most common cause: {_mode(inc['Cause'])}"]
    if "equipment" in d:
        eq = d["equipment"]
        lines += ["", "EQUIPMENT", f"- Total equipment: {len(eq)}",
                  f"- Operating normally: {(eq['Condition'] == 'Normal').mean() * 100 if len(eq) else 0:.1f}%",
                  f"- Critical condition: {', '.join(eq[eq['Condition'] == 'Critical']['Equipment ID']) or 'none'}",
                  f"- Overdue maintenance: {', '.join(eq[eq['Maintenance Category'] == 'Overdue']['Equipment ID']) or 'none'}",
                  f"- Average availability: {eq['Availability %'].mean() if len(eq) else 0:.1f}%"]
    al = d["alerts"]
    lines += ["", f"ALERTS: {len(al)} active ({(al['Severity'] == 'CRITICAL').sum()} critical)", ""]
    for _, r in al[al["Severity"] == "CRITICAL"].iterrows():
        lines += [format_alert(r), ""]
    lines += ["Note: thresholds are educational prototype values, not manufacturer or mine safety limits."]
    return "\n".join(lines)


def page_reports(user):
    st.header("Reports")
    d = load_for(user)
    st.caption("Reports contain only the data your role may access.")
    txt = build_report_text(user, d)
    st.download_button("Download summary report (.txt)", txt, f"monitoring_report_{date.today()}.txt", type="primary")
    st.text_area("Report preview", txt, height=420)
    st.subheader("Export tables")
    cols = st.columns(4)
    for col, (name, key) in zip(cols, [("Alerts", "alerts"), ("Workers", "workers"), ("Incidents", "incidents"), ("Equipment", "equipment")]):
        if key in d:
            col.download_button(f"{name} (CSV)", d[key].to_csv(index=False), f"{key}.csv", "text/csv", key=f"dl_{key}")

# User management

def page_users(user):
    st.header("User Management")
    users = load_users()
    table = pd.DataFrame([{"Username": u, "Full name": v["full_name"], "Role": v["role"]} for u, v in users.items()])
    show_df(table)
    t1, t2, t3 = st.tabs(["Add user", "Change role / password", "Delete user"])
    with t1:
        with st.form("add_user", clear_on_submit=True):
            c = st.columns(4)
            un = c[0].text_input("Username")
            fn = c[1].text_input("Full name")
            role = c[2].selectbox("Role", ROLES)
            pw = c[3].text_input("Password (min 8 chars)", type="password")
            ok = st.form_submit_button("Create user", type="primary")
        if ok:
            good, msg = add_user(un, pw, role, fn)
            (st.success if good else st.error)(msg)
    with t2:
        with st.form("edit_user"):
            target = st.selectbox("User", list(users))
            role = st.selectbox("New role", ROLES)
            pw = st.text_input("New password (leave blank to keep)", type="password")
            ok = st.form_submit_button("Update")
        if ok:
            if target == user["username"] and role != "Administrator":
                st.error("You cannot remove your own administrator role.")
            else:
                set_role(target, role)
                if pw:
                    good, msg = set_password(target, pw)
                    (st.success if good else st.error)(msg)
                else:
                    st.success("Role updated.")
    with t3:
        target = st.selectbox("User to delete", [u for u in users if u != user["username"]], key="del_user")
        if st.button("Delete user", type="primary") and target:
            delete_user(target)
            st.success(f"User '{target}' deleted.")



# Main

PAGE_FUNCS = {
    "Dashboard": page_dashboard, "Worker Safety": page_workers, "Safety Incidents": page_incidents,
    "Equipment Monitoring": page_equipment, "Maintenance": page_maintenance, "Risk Assessment": page_risk,
    "Data Analysis": page_analysis, "Reports": page_reports, "User Management": page_users,
}

# Data Upload page

UPLOAD_LABELS = {"workers": "Workers", "incidents": "Incidents", "equipment": "Equipment", "risks": "Risks"}
NOT_IN_FILE = "(not in my file)"
ID_FORMAT = {"workers": ("W", 4), "incidents": ("INC-", 4), "equipment": ("EQP-", 3), "risks": ("RSK-", 3)}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _syn(d):
    return {target: {_norm(w) for w in words} for target, words in d.items()}


def _F(name, kind, default=None, syn=(), choices=None, lo=None, hi=None, boolean=False):
    return {"name": name, "kind": kind, "default": default, "syn": list(syn), "choices": choices,
            "lo": lo, "hi": hi, "boolean": boolean}


_ID_WORDS = ["id", "number", "no", "ref", "reference", "code", "tag", "key", "serial"]
_SHIFT_CHOICES = ["Day", "Afternoon", "Night"]
_TODAY = date.today()

FIELD_SPECS = {
    "workers": [
        _F("Worker ID", "id", syn=_ID_WORDS + ["employee id", "employee number", "employee no", "emp no", "staff id", "staff number",
                                               "worker number", "worker no", "badge", "badge number", "clock number", "payroll number"]),
        _F("Department", "text", "Unknown", ["dept", "division", "section", "business unit", "area", "unit", "team"]),
        _F("Job Role", "text", "Unknown", ["role", "job", "job title", "position", "occupation", "designation", "trade", "title"]),
        _F("Shift", "choice", "Day", ["shift worked", "work shift", "shift type", "roster", "crew shift"], _SHIFT_CHOICES),
        _F("PPE Compliance", "choice", "Compliant", ["ppe", "ppe compliant", "ppe status", "hard hat", "hard hat worn", "wears ppe", "ppe worn", "ppe check"],
           ["Compliant", "Non-Compliant"], boolean=True),
        _F("Training Status", "choice", "Valid", ["training", "safety training", "induction", "safety induction", "training valid", "training current"],
           ["Valid", "Expired", "Pending"]),
        _F("Fatigue Level", "int", 3, ["fatigue", "fatigue score", "fatigue rating", "tiredness", "fatigue index", "alertness"], lo=1, hi=10),
        _F("Safety Observations", "int", 0, ["observations", "safety obs", "obs", "hazard reports", "observation count"], lo=0),
        _F("Near Misses", "int", 0, ["near miss", "nearmisses", "near miss count", "close calls"], lo=0),
        _F("Previous Incidents", "int", 0, ["incidents", "past incidents", "prior incidents", "incident history", "previous incident", "incident count"], lo=0),
    ],
    "incidents": [
        _F("Incident ID", "id", syn=_ID_WORDS + ["incident number", "incident no", "incident ref", "case number", "case no", "event id", "report number"]),
        _F("Date", "date", _TODAY, ["incident date", "date of incident", "event date", "date reported", "reported date", "occurred", "date occurred"]),
        _F("Time", "time", "00:00", ["incident time", "time of incident", "time occurred", "hour"]),
        _F("Shift", "choice", "Day", ["shift worked", "work shift", "shift type"], _SHIFT_CHOICES),
        _F("Location", "text", "Unknown", ["site", "place", "area", "where", "incident location", "level", "site area"]),
        _F("Department", "text", "Unknown", ["dept", "division", "section", "business unit", "team"]),
        _F("Incident Type", "text", "Other", ["type", "incident category", "category", "event type", "classification", "kind"]),
        _F("Severity", "choice", "Low", ["severity level", "seriousness", "rating", "risk rating", "level of severity"], ["Low", "Medium", "High", "Critical"]),
        _F("Injury Status", "choice", "No Injury", ["injury", "injury type", "injury outcome", "outcome", "injury class"],
           ["No Injury", "First Aid", "Medical Treatment", "Lost Time Injury"]),
        _F("Lost-Time Injury", "choice", "No", ["lti", "lost time", "lost time injury flag", "lost time injury yes no"], ["Yes", "No"], boolean=True),
        _F("Cause", "text", "Not recorded", ["root cause", "reason", "primary cause", "contributing factor", "cause of incident"]),
        _F("Corrective Action", "text", "Not recorded", ["action taken", "action", "remedial action", "controls", "follow up", "corrective actions", "actions"]),
        _F("Incident Status", "choice", "Closed", ["status", "case status", "investigation status", "state"], ["Open", "Under Investigation", "Closed"]),
    ],
    "equipment": [
        _F("Equipment ID", "id", syn=_ID_WORDS + ["asset id", "asset number", "asset tag", "unit number", "unit id", "fleet number", "fleet no", "machine id", "machine number"]),
        _F("Equipment Type", "text", "Unknown", ["type", "machine type", "asset type", "category", "machine", "class", "equipment class"]),
        _F("Manufacturer", "text", "Unknown", ["make", "brand", "oem", "vendor", "supplier", "maker"]),
        _F("Department", "text", "Unknown", ["dept", "division", "section", "area", "business unit", "site"]),
        _F("Operating Hours", "int", 0, ["hours", "engine hours", "run hours", "hour meter", "smu", "lifetime hours", "total hours", "meter hours"], lo=0),
        _F("Operating Time (h)", "float", 500.0, ["operating time", "uptime", "run time", "hours operated", "available hours", "working hours", "operating hours this month", "uptime h"], lo=0),
        _F("Downtime (h)", "float", 0.0, ["downtime", "down time", "hours down", "breakdown hours", "outage hours", "downtime h", "hours lost"], lo=0),
        _F("Temperature (C)", "float", 60.0, ["temperature", "temp", "operating temperature", "engine temp", "temperature c", "temp c", "temp degrees"], lo=-50, hi=500),
        _F("Vibration (mm/s)", "float", 2.0, ["vibration", "vib", "vibration mm s", "vibration level", "vibration rms"], lo=0),
        _F("Fuel Consumption (L/h)", "float", 0.0, ["fuel", "fuel consumption", "fuel use", "fuel rate", "litres per hour", "l h", "fuel burn"], lo=0),
        _F("Brake Status", "choice", "N/A", ["brakes", "brake condition", "brake"], ["Good", "Worn", "Fault", "N/A"]),
        _F("Tyre Status", "choice", "N/A", ["tyres", "tires", "tyre condition", "tire status", "tire condition"], ["Good", "Worn", "Damaged", "N/A"]),
        _F("Engine Status", "choice", "Good", ["engine", "engine condition", "engine health"], ["Good", "Warning", "Fault"]),
        _F("Maintenance Status", "choice", "Operational", ["status", "availability status", "current status", "state", "equipment status", "asset status"],
           ["Operational", "Under Maintenance"]),
        _F("Last Maintenance Date", "date", _TODAY - timedelta(days=30), ["last service", "last service date", "last maintenance", "previous service", "last serviced", "service date"]),
        _F("Next Maintenance Due", "date", _TODAY + timedelta(days=90), ["next service", "next service due", "next maintenance", "service due", "due date", "maintenance due", "next due"]),
        _F("Alert Count", "int", 0, ["alerts", "number of alerts", "warnings", "fault count", "alarms", "alert history", "alarm count"], lo=0),
    ],
    "risks": [
        _F("Risk ID", "id", syn=_ID_WORDS + ["risk number", "risk no", "risk ref", "hazard id", "hazard number"]),
        _F("Department", "text", "Unknown", ["dept", "division", "section", "area", "business unit", "site"]),
        _F("Hazard", "text", "Not specified", ["risk", "description", "risk description", "hazard description", "threat", "event", "risk event", "title"]),
        _F("Likelihood", "int", 3, ["probability", "chance", "frequency", "l", "likelihood score", "likelihood rating"], lo=1, hi=5),
        _F("Consequence", "int", 3, ["impact", "severity", "effect", "c", "consequence score", "consequence rating", "impact score"], lo=1, hi=5),
    ],
}

CHOICE_SYNONYMS = {
    "Shift": _syn({"Day": ["d", "day", "days", "morning", "am", "first", "1", "early", "day shift"],
                   "Afternoon": ["a", "afternoon", "evening", "pm", "second", "2", "swing", "late", "afternoon shift"],
                   "Night": ["n", "night", "nights", "graveyard", "third", "3", "nightshift", "night shift", "overnight"]}),
    "PPE Compliance": _syn({"Compliant": ["compliant", "yes", "y", "true", "1", "ok", "pass", "passed", "complete", "worn", "wearing", "full", "complies"],
                            "Non-Compliant": ["non compliant", "noncompliant", "not compliant", "no", "n", "false", "0", "fail", "failed", "violation",
                                              "missing", "partial", "incomplete", "not worn"]}),
    "Training Status": _syn({"Valid": ["valid", "current", "yes", "y", "true", "trained", "up to date", "complete", "completed", "ok", "compliant", "done"],
                             "Expired": ["expired", "lapsed", "overdue", "out of date", "no", "n", "false", "invalid", "outdated"],
                             "Pending": ["pending", "scheduled", "due", "in progress", "awaiting", "booked", "not started", "partial", "due soon"]}),
    "Severity": _syn({"Low": ["low", "minor", "l", "1", "insignificant", "negligible", "trivial"],
                      "Medium": ["medium", "moderate", "med", "m", "2", "serious"],
                      "High": ["high", "major", "h", "3", "severe", "significant"],
                      "Critical": ["critical", "c", "4", "catastrophic", "fatal", "extreme", "very high", "fatality"]}),
    "Injury Status": _syn({"No Injury": ["none", "no injury", "nil", "no", "n a", "na", "uninjured", "no harm"],
                           "First Aid": ["first aid", "fa", "fac", "first aid case", "minor injury"],
                           "Medical Treatment": ["medical treatment", "mti", "mtc", "medical", "treated", "medical treatment case", "restricted work"],
                           "Lost Time Injury": ["lost time injury", "lti", "lost time", "lost time injuries", "fatality", "fatal"]}),
    "Lost-Time Injury": _syn({"Yes": ["yes", "y", "true", "1", "lti", "lost time"], "No": ["no", "n", "false", "0", "none", "nil"]}),
    "Incident Status": _syn({"Open": ["open", "new", "reported", "pending", "active", "raised", "outstanding"],
                             "Under Investigation": ["under investigation", "investigating", "in progress", "investigation", "ongoing", "in review", "review", "investigation in progress"],
                             "Closed": ["closed", "resolved", "complete", "completed", "done", "finished", "signed off", "close out"]}),
    "Maintenance Status": _syn({"Operational": ["operational", "running", "active", "in service", "available", "working", "operating", "up", "ok", "online", "good", "yes"],
                                "Under Maintenance": ["under maintenance", "maintenance", "in maintenance", "servicing", "repair", "down", "broken", "out of service",
                                                      "being serviced", "workshop", "offline", "in workshop", "under repair"]}),
    "Engine Status": _syn({"Good": ["good", "ok", "normal", "running", "healthy", "fine", "operational"],
                           "Warning": ["warning", "warn", "caution", "amber", "check", "alert", "degraded"],
                           "Fault": ["fault", "failed", "failure", "error", "red", "critical", "down", "broken"]}),
    "Brake Status": _syn({"Good": ["good", "ok", "normal", "fine", "serviceable", "new", "pass"],
                          "Worn": ["worn", "wearing", "low", "replace soon", "marginal", "warning"],
                          "Fault": ["fault", "faulty", "failed", "fail", "defective", "broken", "critical"],
                          "N/A": ["n a", "na", "none", "not applicable", "not fitted"]}),
    "Tyre Status": _syn({"Good": ["good", "ok", "normal", "fine", "serviceable", "new", "pass"],
                         "Worn": ["worn", "wearing", "low tread", "replace soon", "marginal", "warning"],
                         "Damaged": ["damaged", "punctured", "cut", "blown", "flat", "fault", "failed", "critical"],
                         "N/A": ["n a", "na", "none", "not applicable", "not fitted"]}),
}


def match_choice(field, choices, value, boolean=False):
    """Translate one cell of the user's file into one of the app's allowed values (None = can't tell)."""
    n = _norm(value)
    if n == "":
        return None
    for c in choices:
        if _norm(c) == n:
            return c
    for target, words in CHOICE_SYNONYMS.get(field, {}).items():
        if n in words:
            return target
    if not boolean:
        toks = set(n.split())
        hits = [c for c in choices if set(_norm(c).split()) <= toks]
        if len(hits) == 1:
            return hits[0]
        close = difflib.get_close_matches(n, [_norm(c) for c in choices], n=1, cutoff=0.8)
        if close:
            return next(c for c in choices if _norm(c) == close[0])
    return None


_NULLS = {"", "na", "n/a", "nan", "none", "null", "nil", "-", "--", "?"}


def to_number(value):
    """Read a number out of messy text: '1,200', '3,5', '85 C', '7/10', '12%', '1 200,5'."""
    s = str(value).strip().replace(" ", " ")
    if s.lower() in _NULLS:
        return float("nan")
    m = re.search(r"-?\d[\d ,.]*", s)
    if not m:
        return float("nan")
    t = m.group(0).strip().replace(" ", "").rstrip(",.")
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def norm_time(value):
    """Return 'HH:MM' for 8:30, 08.30, 0830, 8:30 PM, 14:30:00, 8pm ... else None."""
    s = str(value).strip().lower()
    if not s:
        return None
    m = (re.fullmatch(r"(\d{1,2})[:.h](\d{2})(?::\d{2})?\s*([ap]m)?", s)
         or re.fullmatch(r"(\d{1,2})(\d{2})\s*([ap]m)?", s) if len(s) in (3, 4) else None) or re.fullmatch(r"(\d{1,2})()\s*([ap]m)", s)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return f"{h:02d}:{mi:02d}"
        return None
    try:
        ts = pd.to_datetime(s)
        return f"{ts.hour:02d}:{ts.minute:02d}"
    except Exception:
        return None


def shift_from_time(t):
    try:
        h = int(str(t)[:2])
    except ValueError:
        return "Day"
    return "Day" if 6 <= h < 14 else "Afternoon" if 14 <= h < 22 else "Night"


def detect_dayfirst(series):
    """True when the file's slash dates look like day/month/year (the default when it cannot tell)."""
    seen = False
    for v in series.head(500):
        m = re.match(r"^\s*(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})", str(v))
        if m:
            seen = True
            if int(m.group(1)) > 12:
                return True
            if int(m.group(2)) > 12:
                return False
    return seen


def parse_dates(series, dayfirst):
    iso = series.astype(str).str.match(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    def _p(x, df):
        try:
            return pd.to_datetime(x, errors="coerce", dayfirst=df, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(x, errors="coerce", dayfirst=df)
    if iso.any():
        out[iso] = _p(series[iso], False)
    if (~iso).any():
        out[~iso] = _p(series[~iso], dayfirst)
    return out


def _near_miss_fix(v):
    n = _norm(v)
    return "Near Miss" if n in ("near miss", "nearmiss", "close call", "near hit", "nm") else v


def guess_mapping(name, columns):
    """Suggest which of the user's columns holds which app field (by name similarity)."""
    cand = []
    ncol = {c: _norm(c) for c in columns}
    for sp in FIELD_SPECS[name]:
        names = [_norm(sp["name"])] + [_norm(x) for x in sp["syn"]]
        for c, nc in ncol.items():
            if not nc:
                continue
            best = 0.0
            ct = set(nc.split())
            for k in names:
                kt = set(k.split())
                if nc == k:
                    sc = 1.0
                elif len(k) > 2 and kt <= ct:
                    sc = 0.88
                elif len(nc) > 2 and ct <= kt:
                    sc = 0.8
                elif len(k) > 3 and len(nc) > 3:
                    r = difflib.SequenceMatcher(None, nc.replace(" ", ""), k.replace(" ", "")).ratio()
                    sc = r * 0.9 if r >= 0.8 else 0.0
                else:
                    sc = 0.0
                best = max(best, sc)
            if best >= 0.7:
                cand.append((best, sp["name"], c))
    cand.sort(key=lambda t: -t[0])
    chosen, used = {}, set()
    for sc, f, c in cand:
        if f not in chosen and c not in used:
            chosen[f] = c
            used.add(c)
    return chosen


def out_of_range_fields(name, raw, mapping):
    """Fields on a fixed scale (e.g. 1-10) where the file's numbers fall outside that scale."""
    res = []
    for sp in FIELD_SPECS[name]:
        c = mapping.get(sp["name"])
        if sp["kind"] in ("int", "float") and sp["lo"] is not None and sp["hi"] is not None and c in raw.columns:
            nums = raw[c].map(to_number).dropna()
            if len(nums):
                outside = ((nums < sp["lo"]) | (nums > sp["hi"])).mean()
                if outside >= 0.5 or nums.max() > 2 * sp["hi"]:
                    res.append(sp["name"])
    return res


def unmatched_values(name, raw, mapping):
    """For each mapped choice field: the distinct values in the file the app cannot recognise."""
    res = {}
    for sp in FIELD_SPECS[name]:
        c = mapping.get(sp["name"])
        if sp["kind"] == "choice" and c in raw.columns:
            vals = [v for v in raw[c].astype(str).str.strip().unique() if v != ""]
            bad = [v for v in vals if match_choice(sp["name"], sp["choices"], v, sp["boolean"]) is None]
            if bad:
                res[sp["name"]] = bad
    return res


def transform_upload(name, raw, mapping, defaults, value_maps, dayfirst, rescale, existing_ids=(), id_start=0, mode="replace"):
    """Convert the user's table into the app's table. Returns (clean_table, skipped_rows_table, notes)."""
    specs = FIELD_SPECS[name]
    out = pd.DataFrame(index=raw.index)
    problems, notes = {}, []

    def has(f):
        return mapping.get(f) in raw.columns

    def flag(mask, reason):
        for i in mask[mask].index:
            problems.setdefault(i, reason)

    def col(f):
        return raw[mapping[f]].astype(str).str.strip() if has(f) else None

    time_from_date = None
    for sp in specs:
        f, kind = sp["name"], sp["kind"]
        if kind == "id":
            continue
        s = col(f)
        dflt = defaults.get(f, sp["default"])
        blank = (s == "") if s is not None else None
        if kind == "text":
            if s is None:
                out[f] = dflt
            else:
                v = s.where(~blank, dflt)
                out[f] = v.map(_near_miss_fix) if f == "Incident Type" else v
        elif kind in ("int", "float"):
            if s is None:
                out[f] = float(dflt)
                continue
            num = s.map(to_number)
            flag(num.isna() & ~blank, f"'{mapping[f]}' is not a number")
            num = num.where(~blank, float(dflt))
            lo, hi = sp["lo"], sp["hi"]
            if lo is not None and hi is not None and rescale.get(f):
                good = num.dropna()
                if len(good) and good.max() > good.min() and (good.min() < lo or good.max() > hi):
                    num = lo + (num - good.min()) / (good.max() - good.min()) * (hi - lo)
                    notes.append(f"'{mapping[f]}' was rescaled from {good.min():g}-{good.max():g} to {lo}-{hi}.")
            if kind == "int":
                num = num.round()
            if lo is not None:
                flag(num < lo, f"'{mapping[f]}' is below {lo}")
            if hi is not None:
                flag(num > hi, f"'{mapping[f]}' is above {hi}")
            out[f] = num
        elif kind == "choice":
            if s is None:
                out[f] = dflt
                continue
            vm = value_maps.get(f, {})

            def conv(v, f=f, sp=sp, vm=vm, dflt=dflt):
                if v == "":
                    return dflt
                if v in vm:
                    return vm[v]
                return match_choice(f, sp["choices"], v, sp["boolean"])
            mapped = s.map(conv)
            flag(mapped.isna(), f"'{mapping[f]}' has a value that could not be matched")
            out[f] = mapped
        elif kind == "date":
            if s is None:
                out[f] = pd.Timestamp(dflt)
                continue
            d = parse_dates(s, dayfirst)
            flag(d.isna() & ~blank, f"'{mapping[f]}' is not a readable date")
            d = d.where(~blank, pd.Timestamp(dflt))
            out[f] = d
            if f == "Date":
                time_from_date = d
        elif kind == "time":
            if s is None:
                if time_from_date is not None and ((time_from_date.dt.hour + time_from_date.dt.minute) > 0).any():
                    out[f] = time_from_date.dt.strftime("%H:%M")
                    notes.append("Incident times were taken from the date column.")
                else:
                    out[f] = dflt
                continue
            t = s.map(norm_time)
            flag(t.isna() & ~blank, f"'{mapping[f]}' is not a readable time")
            out[f] = t.where(~blank, dflt)

    if name == "incidents":
        if not has("Shift") and has("Time"):
            out["Shift"] = out["Time"].map(shift_from_time)
            notes.append("Shift was worked out from the incident time.")
        elif not has("Shift") and "Time" in out and out["Time"].ne("00:00").any():
            out["Shift"] = out["Time"].map(shift_from_time)
            notes.append("Shift was worked out from the incident time.")
        if not has("Lost-Time Injury"):
            out["Lost-Time Injury"] = np.where(out["Injury Status"] == "Lost Time Injury", "Yes", "No")
            notes.append("Lost-Time Injury was worked out from Injury Status.")
    if name == "risks":
        out["Risk Score"] = out["Likelihood"] * out["Consequence"]
        out["Classification"] = out["Risk Score"].map(lambda v: classify_risk(int(v)) if pd.notna(v) else None)
        notes.append("Risk Score and Classification were calculated from Likelihood x Consequence.")

    # IDs: use the file's ID column, make new ones where blank / not mapped
    idf = COLUMNS[name][0]
    prefix, width = ID_FORMAT[name]
    given = col(idf) if has(idf) else pd.Series("", index=raw.index)
    counter, ids = id_start, []
    for i in raw.index:
        v = given[i]
        if v == "":
            counter += 1
            v = f"{prefix}{counter:0{width}d}"
        ids.append(v)
    out.insert(0, idf, ids)
    if not has(idf):
        notes.append(f"{idf} was created automatically ({prefix}{id_start + 1:0{width}d} onwards).")
    explicit = given != ""
    flag(out[idf].duplicated(keep="first") & explicit, f"duplicate {idf} in your file")
    if mode == "add":
        flag(out[idf].isin(set(map(str, existing_ids))) & explicit, f"{idf} already exists in the app")

    keep = ~out.index.isin(list(problems))
    clean = out.loc[keep, COLUMNS[name]].reset_index(drop=True)
    for sp in specs:
        if sp["kind"] == "int":
            clean[sp["name"]] = clean[sp["name"]].astype(int)
        elif sp["kind"] == "float":
            clean[sp["name"]] = clean[sp["name"]].astype(float)
        elif sp["kind"] == "date":
            clean[sp["name"]] = pd.to_datetime(clean[sp["name"]])
    if name == "risks":
        clean["Risk Score"] = clean["Risk Score"].astype(int)
    skipped = pd.DataFrame([{"Row in your file": i + 2, "Why it was skipped": r} for i, r in sorted(problems.items())],
                           columns=["Row in your file", "Why it was skipped"])
    return clean, skipped, notes


def read_upload(uploaded):
    """Read a CSV / TSV / TXT / Excel upload into {sheet name: table of text}."""
    fname = uploaded.name.lower()
    data = uploaded.getvalue()
    if fname.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, keep_default_na=False)
        except ImportError:
            raise ValueError("Excel files need the 'openpyxl' package: run  pip install openpyxl  - or save the sheet as CSV.")
        return {k: v for k, v in sheets.items() if len(v.columns)}
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    return {"file": df}


def apply_upload(name, clean, mode):
    """Save the converted table. mode is 'replace' or 'add'. Returns a message."""
    if mode == "replace":
        save(name, clean)
        return f"{UPLOAD_LABELS[name]} replaced: the app now holds {len(clean)} record(s)."
    existing = load(name)
    save(name, pd.concat([existing, clean], ignore_index=True))
    return f"{len(clean)} record(s) added. {UPLOAD_LABELS[name]} now has {len(existing) + len(clean)} record(s)."


def page_upload(user):
    st.header("Upload Data")
    st.caption("Upload a CSV or Excel file in ANY layout. You match your columns to the app's fields, the app converts "
               "the values, skips rows it cannot use and tells you exactly why.")
    c1, c2 = st.columns(2)
    name = c1.selectbox("What kind of data is in your file?", list(UPLOAD_LABELS), format_func=lambda k: UPLOAD_LABELS[k], key="up_name")
    mode_label = c2.radio("What should happen to the records already in the app?",
                          ["Replace all current records", "Add to the current records"], key="up_mode")
    mode = "replace" if mode_label.startswith("Replace") else "add"
    specs = FIELD_SPECS[name]
    with st.expander("What the app understands for this data"):
        show_df(pd.DataFrame([{"Field": sp["name"], "Type": {"id": "ID (made automatically if missing)", "text": "text", "int": "whole number",
                                                              "float": "number", "choice": "one of: " + ", ".join(sp["choices"] or []),
                                                              "date": "date", "time": "time of day"}[sp["kind"]],
                               "Used when not in your file": "" if sp["kind"] == "id" else str(sp["default"])} for sp in specs]))
        st.download_button("Download a blank CSV template", pd.DataFrame(columns=COLUMNS[name]).to_csv(index=False),
                           f"{name}_template.csv", "text/csv", key="up_template")
    uploaded = st.file_uploader("Choose your file (CSV, TSV, TXT or Excel)", type=["csv", "tsv", "txt", "xlsx", "xls"], key="up_file")
    if uploaded is None:
        return
    try:
        tables = read_upload(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return
    if not tables:
        st.error("The file has no data.")
        return
    sheet = list(tables)[0]
    if len(tables) > 1:
        sheet = st.selectbox("Which sheet?", list(tables), key=f"up_sheet_{uploaded.name}")
    raw = tables[sheet].copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.astype(str).apply(lambda s: s.str.strip())
    raw = raw[~(raw == "").all(axis=1)]
    if raw.empty:
        st.error("The file has no rows of data.")
        return
    cols = list(raw.columns)
    kp = f"up|{name}|{uploaded.name}|{uploaded.size}|{sheet}"
    st.success(f"Read {len(raw)} row(s) and {len(cols)} column(s) from {uploaded.name}.")
    show_df(raw.head(5))

    # ---- 1) column matching
    st.subheader("1. Match your columns to the app's fields")
    st.caption("The app guessed these from your column names. Change any that are wrong; choose "
               f"'{NOT_IN_FILE}' for fields your file does not have.")
    guess = guess_mapping(name, cols)
    mapping, grid = {}, st.columns(3)
    for k, sp in enumerate(specs):
        f = sp["name"]
        options = [NOT_IN_FILE] + cols
        idx = options.index(guess[f]) if f in guess else 0
        label = f + (" (new IDs are made if not in file)" if sp["kind"] == "id" else "")
        choice = grid[k % 3].selectbox(label, options, index=idx, key=f"{kp}|map|{f}")
        mapping[f] = None if choice == NOT_IN_FILE else choice
    matched = sum(1 for sp in specs if sp["kind"] != "id" and mapping[sp["name"]])
    if matched == 0:
        st.error("None of your columns are matched to a field yet. Use the drop-downs above to match at least one.")
        return

    # ---- 2) values for fields that are not in the file
    defaults = {}
    missing = [sp for sp in specs if sp["kind"] != "id" and not mapping[sp["name"]]]
    if missing:
        with st.expander(f"2. Fields not in your file ({len(missing)}) - what value should the app use?", expanded=False):
            grid2 = st.columns(3)
            for k, sp in enumerate(missing):
                f, w = sp["name"], grid2[k % 3]
                key = f"{kp}|def|{f}"
                if sp["kind"] == "choice":
                    defaults[f] = w.selectbox(f, sp["choices"], index=sp["choices"].index(sp["default"]), key=key)
                elif sp["kind"] == "int":
                    defaults[f] = int(w.number_input(f, value=int(sp["default"]), step=1, key=key))
                elif sp["kind"] == "float":
                    defaults[f] = float(w.number_input(f, value=float(sp["default"]), key=key))
                elif sp["kind"] == "date":
                    defaults[f] = w.date_input(f, value=sp["default"], key=key)
                else:
                    defaults[f] = w.text_input(f, value=str(sp["default"]), key=key)

    # ---- 3) values the app cannot recognise
    value_maps = {}
    unmatched = unmatched_values(name, raw, mapping)
    if unmatched:
        with st.expander(f"3. Values the app does not recognise ({sum(len(v) for v in unmatched.values())}) - what do they mean?", expanded=True):
            for f, vals in unmatched.items():
                sp = next(x for x in specs if x["name"] == f)
                st.markdown(f"**{f}** - column '{mapping[f]}'")
                value_maps[f] = {}
                grid3 = st.columns(3)
                for k, v in enumerate(vals[:12]):
                    pick = grid3[k % 3].selectbox(f'"{v}" means', sp["choices"], index=sp["choices"].index(defaults.get(f, sp["default"])),
                                                  key=f"{kp}|val|{f}|{v}")
                    value_maps[f][v] = pick
                if len(vals) > 12:
                    st.warning(f"{len(vals) - 12} more unrecognised values in '{mapping[f]}' - rows with them will be skipped.")

    # ---- 4) options
    dayfirst, rescale = False, {}
    date_cols = [mapping[sp["name"]] for sp in specs if sp["kind"] == "date" and mapping[sp["name"]]]
    odd = out_of_range_fields(name, raw, mapping)
    if date_cols or odd:
        with st.expander("4. Dates and scales", expanded=bool(odd)):
            if date_cols:
                dayfirst = st.checkbox("Dates like 03/04/2026 are day/month/year (3 April)",
                                       value=detect_dayfirst(raw[date_cols[0]]), key=f"{kp}|dayfirst")
            for f in odd:
                sp = next(x for x in specs if x["name"] == f)
                rescale[f] = st.checkbox(f"'{mapping[f]}' does not fit {sp['lo']}-{sp['hi']} - rescale it to that range?", value=True, key=f"{kp}|rescale|{f}")

    # ---- result
    existing = load(name) if mode == "add" else None
    existing_ids = list(existing[COLUMNS[name][0]].astype(str)) if existing is not None else []
    id_start = 0
    if existing is not None and len(existing):
        nums = [int(m.group(1)) for v in existing_ids if (m := re.search(r"(\d+)$", v))]
        id_start = max(nums) if nums else 0
    clean, skipped, notes = transform_upload(name, raw, mapping, defaults, value_maps, dayfirst, rescale, existing_ids, id_start, mode)
    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows in your file", len(raw))
    m2.metric("Rows ready to load", len(clean))
    m3.metric("Rows skipped", len(skipped))
    for n in notes:
        st.info(n)
    if len(skipped):
        with st.expander(f"Why {len(skipped)} row(s) will be skipped"):
            show_df(skipped.head(200))
    if clean.empty:
        st.error("No rows can be loaded yet. Fix the matching above, or correct the rows listed as skipped.")
        return
    show_df(clean.head(10))
    sure = st.checkbox("Yes, load these rows" + (" and REPLACE the current records" if mode == "replace" else ""), key=f"{kp}|sure")
    if st.button(f"Load {len(clean)} row(s) into the app", type="primary", key=f"{kp}|go") and sure:
        st.success(apply_upload(name, clean, mode))
        st.caption("Open the matching page, or the Dashboard, to see the new data.")



def main():
    ensure_data()
    ensure_users()
    if "user" not in st.session_state:
        apply_theme(LOGIN_THEME)
        login_page()
        return
    user = st.session_state["user"]
    apply_theme(ROLE_THEMES.get(user["role"], LOGIN_THEME))
    allowed = [p for p, perm in PAGES.items() if can(user, perm)]
    if user["role"] == "Administrator":
        allowed.append("Upload Data")
    with st.sidebar:
        st.title("Khwezi Mining 👷👷‍♀️")
        st.markdown(f"**{user['full_name']}**  \n{user['role']}")
        page = st.radio("Navigation", allowed)
        with st.expander("Change my password"):
            with st.form("change_pw", clear_on_submit=True):
                old_pw = st.text_input("Current password", type="password")
                new_pw1 = st.text_input("New password (min 8 characters)", type="password")
                new_pw2 = st.text_input("Confirm new password", type="password")
                go_pw = st.form_submit_button("Update password")
            if go_pw:
                if not authenticate(user["username"], old_pw):
                    st.error("Current password is incorrect.")
                elif new_pw1 != new_pw2:
                    st.error("The new passwords do not match.")
                elif new_pw1 == old_pw:
                    st.error("Choose a password different from the current one.")
                else:
                    pw_ok, pw_msg = set_password(user["username"], new_pw1)
                    (st.success if pw_ok else st.error)(pw_msg)
        if user["role"] == "Administrator":
            with st.expander("Admin tools"):
                st.caption("The app starts empty. This optionally fills it with made-up demo records.")
                sure = st.checkbox("I understand this REPLACES all current records")
                if st.button("Load sample data") and sure:
                    generate_all()
                    st.success("Sample data loaded.")
        if st.button("Log out"):
            del st.session_state["user"]
            st.rerun()
    if page == "Upload Data":
        page_upload(user)
    else:
        PAGE_FUNCS[page](user)


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return True


if _running_in_streamlit():
    _write_theme_config()
    main()
else:

    import sys
    from streamlit.web import cli as stcli
    _write_theme_config()
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--theme.base=dark", "--theme.backgroundColor=#000000",
                "--theme.secondaryBackgroundColor=#16161a", "--theme.textColor=#ffffff"]
    sys.exit(stcli.main())
