"""Exact tools that answer without the LLM (instant and never hallucinated):

  * lookup_section()        - IPC/CrPC/Evidence Act  <->  BNS/BNSS/BSA section finder
  * deadline_from_question()/calculate_deadlines() - legal deadline calculator
"""
import calendar
import re
from datetime import date, timedelta

# --------------------------------------------------------------------------
# 1. Section mapping   {old section: (offence / subject, new section)}
#    Quick-reference table: verify against the official text before relying on it.
# --------------------------------------------------------------------------
IPC_TO_BNS = {
    "34": ("Acts done by several persons in furtherance of common intention", "3(5)"),
    "120B": ("Criminal conspiracy", "61"),
    "124A": ("Sedition (repealed; BNS 152 is a differently worded offence)", "152"),
    "299": ("Culpable homicide", "100"),
    "300": ("Murder (definition)", "101"),
    "302": ("Punishment for murder", "103"),
    "304": ("Punishment for culpable homicide not amounting to murder", "105"),
    "304A": ("Death caused by rash or negligent act", "106(1)"),
    "304B": ("Dowry death", "80"),
    "307": ("Attempt to murder", "109"),
    "323": ("Voluntarily causing hurt", "115"),
    "325": ("Voluntarily causing grievous hurt", "117"),
    "341": ("Wrongful restraint", "126"),
    "354": ("Assault or criminal force to outrage modesty of a woman", "74"),
    "354D": ("Stalking", "78"),
    "363": ("Kidnapping", "137"),
    "375": ("Rape (definition)", "63"),
    "376": ("Punishment for rape", "64"),
    "376D": ("Gang rape", "70"),
    "378": ("Theft (definition)", "303"),
    "379": ("Punishment for theft", "303"),
    "383": ("Extortion (definition)", "308"),
    "384": ("Punishment for extortion", "308"),
    "390": ("Robbery (definition)", "309"),
    "391": ("Dacoity (definition)", "310"),
    "392": ("Punishment for robbery", "309"),
    "395": ("Punishment for dacoity", "310"),
    "406": ("Criminal breach of trust", "316"),
    "411": ("Dishonestly receiving stolen property", "317"),
    "415": ("Cheating (definition)", "318"),
    "420": ("Cheating and dishonestly inducing delivery of property", "318"),
    "498A": ("Cruelty by husband or his relatives", "85/86"),
    "499": ("Defamation (definition)", "356"),
    "500": ("Punishment for defamation", "356"),
    "506": ("Criminal intimidation", "351"),
    "509": ("Insulting the modesty of a woman", "79"),
}
CRPC_TO_BNSS = {
    "41": ("Arrest without warrant", "35"),
    "125": ("Maintenance of wives, children and parents", "144"),
    "154": ("Registration of FIR", "173"),
    "156(3)": ("Magistrate orders investigation", "175(3)"),
    "167": ("Remand and default bail", "187"),
    "173": ("Police report (chargesheet)", "193"),
    "436": ("Bail in bailable offences", "478"),
    "437": ("Bail in non-bailable offences (magistrate)", "480"),
    "438": ("Anticipatory bail", "482"),
    "439": ("Special bail powers of High Court / Sessions Court", "483"),
    "482": ("Inherent powers of the High Court", "528"),
}
IEA_TO_BSA = {
    "65B": ("Admissibility of electronic records (certificate)", "63"),
}
# old act key -> (table, old label, new label)
_FORWARD = {
    "ipc": (IPC_TO_BNS, "IPC", "BNS"),
    "crpc": (CRPC_TO_BNSS, "CrPC", "BNSS"),
    "iea": (IEA_TO_BSA, "Evidence Act", "BSA"),
}
_NEW_TO_OLD = {"bns": "ipc", "bnss": "crpc", "bsa": "iea"}

_ALIASES = {
    "indian penal code": "ipc", "ipc": "ipc",
    "bharatiya nyaya sanhita": "bns", "bns": "bns",
    "code of criminal procedure": "crpc", "cr.p.c.": "crpc", "cr.p.c": "crpc", "crpc": "crpc",
    "bharatiya nagarik suraksha sanhita": "bnss", "bnss": "bnss",
    "indian evidence act": "iea", "evidence act": "iea",
    "bharatiya sakshya adhiniyam": "bsa", "bsa": "bsa",
}
_ACT = "|".join(re.escape(a) for a in sorted(_ALIASES, key=len, reverse=True))
_NUM = r"(\d{1,3}[A-Za-z]?(?:\(\d+\))?)"
_SEC = r"(?:sections?|secs?\.?|s\.)?"
_PAT_NUM_ACT = re.compile(
    rf"(?<![0-9A-Za-z]){_SEC}\s*{_NUM}\s*(?:of\s+(?:the\s+)?)?({_ACT})(?![A-Za-z])", re.I
)
_PAT_ACT_NUM = re.compile(
    rf"(?<![A-Za-z])({_ACT})\s*{_SEC}\s*{_NUM}(?![0-9A-Za-z])", re.I
)
_NOTE = (
    "_Quick-reference mapping. Wording and punishment can differ between the old and "
    "new section, so check the official text on indiacode.nic.in before relying on it._"
)


def _parse_section(question: str):
    m = _PAT_NUM_ACT.search(question)
    if m:
        return _ALIASES[m.group(2).lower()], m.group(1).upper()
    m = _PAT_ACT_NUM.search(question)
    if m:
        return _ALIASES[m.group(1).lower()], m.group(2).upper()
    return None


def _bases(new_value: str) -> set[str]:
    return {re.match(r"\d+", p.strip()).group() for p in new_value.split("/") if p.strip()}


def lookup_section(question: str) -> str | None:
    """Markdown answer for 'Section 420 IPC in BNS' / 'BNS 103' style questions, else None."""
    parsed = _parse_section(question)
    if not parsed:
        return None
    act, num = parsed

    if act in _FORWARD:  # old -> new
        table, old, new = _FORWARD[act]
        if num not in table:
            return None
        title, target = table[num]
        return f"**{old} {num} → {new} {target.replace('/', ' / ')}**\n\n- Subject: {title}\n\n{_NOTE}"

    old_key = _NEW_TO_OLD[act]  # new -> old
    table, old, new = _FORWARD[old_key]
    base = re.match(r"\d+", num).group()
    hits = [(k, v) for k, v in table.items() if base in _bases(v[1])]
    if not hits:
        return None
    lines = [f"**{new} {num} ← {old} {', '.join(k for k, _ in hits)}**", ""]
    for k, (title, target) in hits:
        lines.append(f"- {old} {k}: {title} (now {new} {target.replace('/', ' / ')})")
    return "\n".join(lines) + f"\n\n{_NOTE}"


_LIST_WORDS = re.compile(r"\b(table|list|each|every|mapping|short notes?|brief notes?)\b", re.I)
_ACT_WORDS = re.compile(r"\b(ipc|bns|crpc|bnss|penal code|sanhita)\b", re.I)


def _sort_key(k: str):
    m = re.match(r"(\d+)(.*)", k)
    return int(m.group(1)), m.group(2)


def mapping_table(question: str) -> str | None:
    """Full old -> new section table for 'give me the table / list / short note about each IPC section'."""
    if not _LIST_WORDS.search(question):
        return None
    names_act = _ACT_WORDS.search(question)
    table_of_sections = re.search(r"\btable\b", question, re.I) and re.search(r"\bsections?\b", question, re.I)
    if not (names_act or table_of_sections):
        return None
    only_crpc = re.search(r"\b(crpc|bnss)\b", question, re.I) and not re.search(r"\b(ipc|bns|penal code)\b", question, re.I)
    table, old, new = _FORWARD["crpc" if only_crpc else "ipc"]
    lines = [f"**{old} → {new} section mapping**", "",
             f"| {old} section | Offence / subject | {new} section |", "|---|---|---|"]
    for k in sorted(table, key=_sort_key):
        title, target = table[k]
        lines.append(f"| {k} | {title} | {target.replace('/', ' / ')} |")
    return "\n".join(lines) + f"\n\n{_NOTE}"


# --------------------------------------------------------------------------
# 2. Deadline calculator
# --------------------------------------------------------------------------
KINDS = {
    "cheque": "Cheque bounce (Sec. 138, NI Act)",
    "consumer": "Consumer complaint",
    "rti": "RTI application",
    "arbitration": "Challenging an arbitral award (Sec. 34)",
    "default_bail": "Default bail (BNSS Sec. 187)",
}
START_LABELS = {
    "cheque": "Date you received the bank's cheque-dishonour information",
    "consumer": "Date the problem arose (cause of action)",
    "rti": "Date you filed the RTI application",
    "arbitration": "Date you received the arbitral award",
    "default_bail": "Date of first remand",
}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_INTENT = re.compile(
    r"deadline|last date|last day|time limit|limitation|how many days|by when|due date|"
    r"calculate|expire|how long|when (?:do|should|must|can|will)", re.I)


def _mk(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def find_date(text: str) -> date | None:
    """Finds the first date written as 2026-11-15, 15/11/2026 (day first), 15 Nov 2026 or Nov 15, 2026."""
    t = text.lower()
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return _mk(*m.groups())
    m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", t)
    if m:
        d, mo, y = m.groups()
        return _mk(y, mo, d)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3})[a-z]*\.?,?\s+(\d{4})\b", t)
    if m and m.group(2) in _MONTHS:
        return _mk(m.group(3), _MONTHS[m.group(2)], m.group(1))
    m = re.search(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t)
    if m and m.group(1) in _MONTHS:
        return _mk(m.group(3), _MONTHS[m.group(1)], m.group(2))
    return None


def add_months(d: date, n: int) -> date:
    y, mo = divmod(d.month - 1 + n, 12)
    y += d.year
    return date(y, mo + 1, min(d.day, calendar.monthrange(y, mo + 1)[1]))


def _status(d: date, today: date) -> str:
    delta = (d - today).days
    if delta > 0:
        return f"in {delta} days"
    if delta == 0:
        return "today"
    return f"passed {-delta} days ago"


def calculate_deadlines(kind: str, start: date, today: date | None = None) -> str:
    today = today or date.today()
    rows, notes = [], []
    if kind == "cheque":
        notice_by = start + timedelta(days=30)
        pay_by = notice_by + timedelta(days=15)
        rows = [
            ("Send written demand notice to the drawer", notice_by,
             "Within 30 days of receiving the bank's dishonour information"),
            ("Drawer's 15-day payment window ends", pay_by,
             "Assumes the notice is received on the last possible day"),
            ("File complaint in the Magistrate's court", add_months(pay_by, 1),
             "Within one month after the 15 days expire"),
        ]
        notes.append("If you send the notice earlier, count 15 days from the date the drawer "
                     "receives it and recalculate the last two dates. The cheque must also have "
                     "been presented within 3 months of its date.")
    elif kind == "consumer":
        rows = [("Last date to file the complaint", add_months(start, 24),
                 "2 years from the cause of action (Sec. 69); delay can be condoned for sufficient cause")]
    elif kind == "rti":
        due = start + timedelta(days=30)
        rows = [
            ("PIO must reply by", due, "30 days (48 hours if life or liberty is involved)"),
            ("Last date for first appeal if there is no reply", due + timedelta(days=30),
             "Within 30 days of the reply, or of the date it was due"),
        ]
        notes.append("A second appeal to the Information Commission must be filed within 90 days "
                     "of the first appeal order (or the date it was due).")
    elif kind == "arbitration":
        main = add_months(start, 3)
        rows = [
            ("Last date to apply to set aside the award", main, "Within 3 months of receiving the award"),
            ("Extended date if the court condones delay", main + timedelta(days=30),
             "Court may allow up to 30 more days for sufficient cause, not beyond"),
        ]
    elif kind == "default_bail":
        rows = [
            ("Chargesheet due (offences up to 10 years)", start + timedelta(days=60),
             "If not filed, the accused has a right to default bail"),
            ("Chargesheet due (death, life or 10+ years)", start + timedelta(days=90),
             "Same right after 90 days"),
        ]
        notes.append("Counting starts from the first remand; ask a lawyer how it applies to the case.")
    else:
        raise ValueError(f"Unknown deadline type: {kind}")

    out = [f"**{KINDS[kind]}** - counting from {start.strftime('%d %b %Y')}", "",
           "| Step | Date | Status | Rule |", "|---|---|---|---|"]
    for step, d, rule in rows:
        out.append(f"| {step} | **{d.strftime('%d %b %Y')}** | {_status(d, today)} | {rule} |")
    out.append("")
    out += [f"- {n}" for n in notes]
    out.append("")
    out.append("_Calculated from the standard rules. Holidays, the exact receipt date and court "
               "discretion can change the result, so confirm with a lawyer for a real case._")
    return "\n".join(out)


def detect_kind(question: str) -> str | None:
    q = question.lower()
    if re.search(r"cheque|dishonou?r|bounce", q):
        return "cheque"
    if re.search(r"default bail|statutory bail", q):
        return "default_bail"
    if re.search(r"arbitra", q):
        return "arbitration"
    if "consumer" in q:
        return "consumer"
    if re.search(r"\brti\b|right to information", q):
        return "rti"
    return None


def deadline_from_question(question: str) -> str | None:
    """Only fires when the question names a situation, asks about timing AND contains a date."""
    kind, start = detect_kind(question), find_date(question)
    if kind and start and _INTENT.search(question):
        return calculate_deadlines(kind, start)
    return None