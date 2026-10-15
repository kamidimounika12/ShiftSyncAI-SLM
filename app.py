# ShiftSync AI - Intelligent Factory Shift Handover Assistant
# Prototype only. Does not certify machine safety or authorize restart.

import html
import json
import re
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT_DIR / "models" / "shiftsync-lora-v2"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

st.set_page_config(
    page_title="ShiftSync AI | Factory Handover",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background:#F3F7FA; color:#243447; }
    [data-testid="stHeader"] { background:rgba(243,247,250,.96); }
    [data-testid="stSidebar"] { background:#10243A; }
    [data-testid="stSidebar"] * { color:#E8F1F8; }
    .block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
    .hero { background:linear-gradient(120deg,#10243A,#183651 75%,#0F766E); border-radius:18px; padding:30px 32px; color:white; margin-bottom:22px; box-shadow:0 8px 24px rgba(16,36,58,.12); }
    .hero-kicker { text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; color:#77E4D8; margin-bottom:8px; }
    .hero h1 { color:white; font-size:32px; font-weight:750; margin:0 0 9px 0; }
    .hero p { color:#D6E4EF; margin:0; font-size:15px; line-height:1.6; }
    .section-heading { color:#10243A; font-size:21px; font-weight:750; margin:8px 0 4px 0; }
    .section-caption { color:#64748B; font-size:13px; margin-bottom:16px; }
    .stat-card { background:white; border:1px solid #DCE6ED; border-radius:14px; padding:17px 18px; min-height:112px; }
    .stat-label { font-size:11px; text-transform:uppercase; letter-spacing:1.1px; color:#64748B; font-weight:750; margin-bottom:9px; }
    .stat-value { color:#10243A; font-size:23px; font-weight:750; overflow-wrap:anywhere; }
    .stat-note { color:#64748B; font-size:12px; margin-top:6px; }
    .notice { border-radius:10px; padding:13px 16px; background:#FFF8E8; border:1px solid #F2D99A; color:#795315; font-size:13px; line-height:1.6; margin:12px 0 20px 0; }
    .safe-notice { border-radius:10px; padding:13px 16px; background:#EAF8F6; border:1px solid #B8E7E1; color:#115E59; font-size:13px; line-height:1.6; margin:10px 0; }
    .flag-card { background:white; border:1px solid #F0D7A3; border-left:4px solid #D99A22; border-radius:10px; padding:13px 15px; margin:9px 0; color:#374151; font-size:13px; line-height:1.55; }
    .footer { color:#64748B; font-size:12px; text-align:center; border-top:1px solid #DCE6ED; padding-top:20px; margin-top:30px; }
    div.stButton > button { background:#0F766E; color:white; border:1px solid #0F766E; border-radius:9px; font-weight:650; min-height:43px; }
    div.stButton > button:hover { background:#115E59; color:white; border-color:#115E59; }
    div[data-testid="stDownloadButton"] button { background:#10243A; color:white; border:1px solid #10243A; border-radius:9px; font-weight:650; min-height:43px; }
    div[data-testid="stExpander"] { background:white; border:1px solid #DCE6ED; border-radius:10px; }
    @media (max-width:700px) { .hero{padding:23px 20px;} .hero h1{font-size:25px;} .block-container{padding-left:1rem;padding-right:1rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_items(items):
    result, seen = [], set()
    for item in items:
        item = clean_text(item)
        if not item:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def has_pattern(note, patterns):
    return any(re.search(p, note, re.IGNORECASE) for p in patterns)


def extract_machine_id(note):
    patterns = [
        r"\b(?:machine|equipment)\s*(?:id)?\s*[:#-]?\s*(M[- ]?\d{2,}|[A-Z]{1,4}[- ]?\d{2,})\b",
        r"\b(M-\d{2,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", "-", match.group(1)).upper()
    return "Not provided"


def detect_machine_status(note):
    stopped = has_pattern(note, [
        r"\bstopped\b", r"\bshut\s*down\b", r"\bshut\s+off\b",
        r"\bpowered\s+off\b", r"\bnot running\b", r"\bswitched off\b",
    ])
    running = has_pattern(note, [
        r"\bcurrently running\b", r"\bmachine is running\b",
        r"\bcurrently operating\b", r"\boperating normally\b",
        r"\bmachine is operating\b",
    ])
    if stopped and running:
        return "Conflicting information"
    if stopped:
        return "Stopped"
    if running:
        return "Running"
    return "Not stated"


def cause_is_unknown(note):
    patterns = [
        r"\b(?:cause|root cause)\b.{0,100}\b(?:unknown|unidentified|unclear|undetermined)\b",
        r"\b(?:cause|root cause)\b.{0,100}\bnot\s+(?:yet\s+)?(?:been\s+)?(?:identified|determined|established|known)\b",
        r"\b(?:unknown|unidentified|unclear|undetermined)\b.{0,60}\b(?:cause|root cause)\b",
        r"\b(?:cause|root cause)\b.{0,100}\bnot\s+known\b",
    ]
    return has_pattern(note, patterns)


def inspection_is_pending(note):
    return has_pattern(note, [
        r"\b(?:inspection|maintenance|diagnostic|investigation|cooling inspection)\b[^.!?]{0,100}\b(?:pending|awaiting|outstanding|incomplete)\b",
        r"\b(?:inspection|maintenance|diagnostic|investigation)\b[^.!?]{0,100}\bnot\s+(?:yet\s+)?(?:completed|complete|done)\b",
        r"\b(?:pending|awaiting|outstanding|incomplete)\b[^.!?]{0,100}\b(?:inspection|maintenance|diagnostic|investigation)\b",
    ])


def verification_is_pending(note):
    return has_pattern(note, [
        r"\b(?:verification|verify|validation|post[- ]maintenance test|test run|verification run)\b.{0,100}\b(?:pending|awaiting|outstanding|incomplete|not yet completed|not completed)\b",
        r"\b(?:pending|awaiting|outstanding|incomplete)\b.{0,100}\b(?:verification|validation|verification run|test run|post[- ]maintenance test)\b",
    ])


def repair_is_pending(note):
    return has_pattern(note, [
        r"\b(?:repair|repairs|rectification)\b[^.!?]{0,80}\b(?:pending|awaiting|outstanding|incomplete|not yet completed|not completed)\b",
        r"\b(?:pending|awaiting|outstanding|incomplete)\b[^.!?]{0,80}\b(?:repair|repairs|rectification)\b",
    ])


def restart_is_mentioned(note):
    return bool(re.search(r"\brestart(?:ing)?\b", note, re.IGNORECASE))


def restart_is_suggested(note):
    return has_pattern(note, [
        r"\b(?:suggested|recommended|proposed|requested)\b.{0,100}\brestart(?:ing)?\b",
        r"\brestart(?:ing)?\b.{0,100}\b(?:suggested|recommended|proposed|requested)\b",
        r"\b(?:should|could|wants to|plan to|planning to|try to)\b.{0,60}\brestart(?:ing)?\b",
        r"\bimmediately\s+restart\b",
    ])


def isolation_is_unconfirmed(note):
    return has_pattern(note, [
        r"\b(?:power|energy|electrical)\s+isolation\b[^.!?]{0,100}\b(?:not confirmed|unconfirmed|not verified|has not been confirmed|hasn't been confirmed)\b",
        r"\b(?:isolation|lockout|tagout|lockout/tagout)\b[^.!?]{0,100}\b(?:not confirmed|unconfirmed|not verified|has not been confirmed|hasn't been confirmed)\b",
        r"\b(?:not confirmed|unconfirmed|not verified)\b[^.!?]{0,100}\b(?:power|energy|electrical)\s+isolation\b",
    ])


def restart_approval_is_pending(note):
    return has_pattern(note, [
        r"\brestart\s+(?:approval|authorization|authorisation)\b[^.!?]{0,100}\b(?:pending|awaiting|not granted|not approved|not authorized|not authorised)\b",
        r"\b(?:pending|awaiting)\b[^.!?]{0,100}\brestart\s+(?:approval|authorization|authorisation)\b",
        r"\b(?:approval|authorization|authorisation)\s+to\s+restart\b[^.!?]{0,100}\b(?:pending|awaiting|not granted|not approved)\b",
    ])


def extract_alarm_codes(note):
    """Preserve explicitly reported alarm, error, or fault codes."""
    pattern = (
        r"\b(?:alarm|error|fault)\s*(?:code|number|id)?\s*"
        r"(?:is|=|:|-)?\s*([A-Z]{1,5}[-_]?\d{1,6}[A-Z]?)\b"
    )
    return unique_items(
        match.group(1).upper()
        for match in re.finditer(pattern, note, re.IGNORECASE)
    )


def extract_assigned_actions(note):
    """Retain named technicians and their explicitly assigned tasks."""
    actions = []
    # Supports: "Technician Priya will investigate...", "technician Ravi is
    # assigned to inspect...", and "Technician Suresh will check...".
    pattern = re.compile(
        r"\b(?:technician|engineer|supervisor)\s+"
        r"([A-Z][A-Za-z'-]*)\s+"
        r"(?:(?:is|has been)\s+assigned\s+to\s+|"
        r"will\s+|is\s+going\s+to\s+|"
        r"is\s+assigned\s+to\s+)?"
        r"(inspect|investigate|check|examine|review|repair|replace|"
        r"verify|test|monitor|assess)\b([^.!?]*)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(note):
        name = match.group(1)
        verb = match.group(2).lower()
        remainder = clean_text(match.group(3))
        action = f"Technician {name}: {verb}"
        if remainder:
            action += f" {remainder}"
        if not re.search(r"\b(?:document|record)\b", action, re.IGNORECASE):
            action += "; document the findings"
        actions.append(action + ".")
    return unique_items(actions)


def has_event_date_or_time(note):
    return has_pattern(note, [
        r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        r"\b\d{1,2}\s*(?:am|pm)\b", r"\b(?:today|yesterday|tomorrow)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:morning|evening|night)\s+shift\b", r"\bshift\s+(?:[A-C]|\d+)\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
    ])


def has_assigned_owner(note):
    return has_pattern(note, [
        r"\bassigned\s+to\s+[A-Z][a-z]+",
        r"\bresponsible\s+(?:person|owner|team)\s*[:=-]",
        r"\bowner\s*[:=-]\s*[A-Za-z]", r"\baction\s+owner\s*[:=-]\s*[A-Za-z]",
        r"\b(?:maintenance|engineering|inspection)\s+team\s+(?:will|has been assigned|is responsible)",
        r"\btechnician\s+[A-Z][a-z]+\b", r"\b(?:supervisor|engineer|technician)\s*[:=-]\s*[A-Za-z]",
    ])


def extract_reported_issues(note):
    if detect_machine_status(note) == "Conflicting information":
        return [
            "Conflicting information was reported about whether the machine is stopped or running. The actual machine state has not been confirmed."
        ]

    patterns = [
        (r"\boverheat(?:ing|ed)?\b|\bhigh temperature\b", "Overheating or high temperature was reported."),
        (r"\bunusual vibration\b|\bexcessive vibration\b|\bvibrat(?:e|ion|ing|ed)\b", "Vibration was reported."),
        (r"\bunusual noise\b|\bstrange noise\b|\bnoise\b", "Noise was reported."),
        # Match the more specific phrase before the generic leak pattern.
        (r"\bpressure leak\b|\bleak\s+(?:in|from)\s+(?:the\s+)?pressure\b", "A pressure leak was reported."),
        (r"(?<!pressure )\bleak(?:age|ing|ed)?\b", "A leak was reported."),
        (r"\bpressure drop\b", "A pressure drop was reported."),
        (r"\blow pressure\b|\bpressure abnormal\b", "A pressure issue was reported."),
        (r"\bpower failure\b|\bpower loss\b|\belectrical fault\b|\belectrical issue\b", "An electrical or power issue was reported."),
        (r"\bsmoke\b", "Smoke was reported."),
        (r"\breduced speed\b|\bslowdown\b|\bslowing down\b", "Reduced speed was reported."),
        (r"\bdamaged bearing\b|\bbearing damage\b|\bbearing failure\b|\bbearing problem\b|\bfaulty bearing\b", "A bearing problem or damage was reported."),
        (r"\bmalfunction\b|\bfault\b|\bbreakdown\b|\bfailure\b", "A fault or failure was reported."),
        (r"\bcorrosion\b|\brust\b", "Corrosion or rust was reported."),
    ]
    found = [description for pattern, description in patterns if re.search(pattern, note, re.IGNORECASE)]

    for code in extract_alarm_codes(note):
        found.append(f"Reported alarm/error code: {code}.")

    measurements = [
        (r"\bair temperature\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*K\b", "Air temperature: {} K."),
        (r"\bprocess temperature\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*K\b", "Process temperature: {} K."),
        (r"\brotational speed\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*rpm\b", "Rotational speed: {} rpm."),
        (r"\btorque\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*Nm\b", "Torque: {} Nm."),
    ]
    for pattern, template in measurements:
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            found.append(template.format(match.group(1)))
    if not found:
        found.append("Operator note received; the issue requires human review.")
    return unique_items(found)


def detect_completed_maintenance(note):
    """Extract only work explicitly described as completed/performed."""
    found = []
    checks = [
        (r"\b(?:completed|finished|performed)\s+(?:the\s+)?(?:required\s+)?inspection\b|\binspection\s+(?:was\s+)?completed\b|\binspected\s+(?:the\s+)?(?:machine|equipment|bearing|motor)\b", "An inspection was reported as completed."),
        (r"\brepaired\b|\brepair\s+(?:was\s+)?completed\b|\bcompleted\s+the\s+repair\b", "A repair was reported as completed."),
        (r"\breplaced\s+(?:the\s+)?(?:(?:worn|damaged|faulty)\s+)?bearing\b", "The bearing was reported as replaced."),
        (r"\breplaced\b|\breplacement\s+(?:was\s+)?completed\b|\bcompleted\s+the\s+replacement\b", "A component replacement was reported as completed."),
        (r"\blubricated\b|\blubrication\s+(?:was\s+)?completed\b", "Lubrication was reported as completed."),
        (r"\bcleaned\b|\bcleaning\s+(?:was\s+)?completed\b", "Cleaning was reported as completed."),
        (r"\btightened\b|\bfasteners were tightened\b", "Tightening work was reported as completed."),
        (r"\btested\s+(?:the\s+)?(?:machine|equipment|motor)\b|\bmachine was tested\b|\bequipment was tested\b", "Machine testing was reported; the test results require verification."),
    ]
    sentences = re.split(r"(?<=[.!?])\s+", note)
    bearing_replaced = bool(re.search(r"\breplaced\s+(?:the\s+)?(?:(?:worn|damaged|faulty)\s+)?bearing\b", note, re.IGNORECASE))
    for pattern, description in checks:
        if description == "A component replacement was reported as completed." and bearing_replaced:
            continue
        for sentence in sentences:
            if not re.search(pattern, sentence, re.IGNORECASE):
                continue
            if re.search(r"\b(?:not|never)\s+(?:yet\s+)?(?:been\s+)?(?:completed|finished|done)\b|\b(?:pending|awaiting|incomplete)\b", sentence, re.IGNORECASE):
                continue
            found.append(description)
            break
    return unique_items(found)


def test_results_missing(note):
    mentions_test = has_pattern(note, [r"\btest(?:ed|ing|s)?\b", r"\btested\b"])
    if not mentions_test:
        return False
    return has_pattern(note, [
        r"\b(?:test results?|results?)\b.{0,50}\b(?:not recorded|not documented|not available|missing|unknown|unavailable|not provided)\b",
        r"\b(?:not recorded|not documented|not available|missing|unavailable)\b.{0,50}\b(?:test results?|results?)\b",
        r"\bfinal test results?\s+(?:were|are)\s+not\s+recorded\b",
        r"\bresults?\s+(?:were|are)\s+not\s+recorded\b",
    ])


def build_note_based_report(note):
    note = clean_text(note)
    lower = note.lower()
    machine_id = extract_machine_id(note)
    status = detect_machine_status(note)
    issues = extract_reported_issues(note)
    completed_work = detect_completed_maintenance(note)
    cause_unknown = cause_is_unknown(note)
    inspection_pending = inspection_is_pending(note)
    verification_pending = verification_is_pending(note)
    repair_pending = repair_is_pending(note)
    restart_mentioned = restart_is_mentioned(note)
    restart_suggested = restart_is_suggested(note)
    tests_missing = test_results_missing(note)
    isolation_unconfirmed = isolation_is_unconfirmed(note)
    approval_pending = restart_approval_is_pending(note)
    assigned_actions = extract_assigned_actions(note)
    unresolved, actions, missing = [], [], []

    issue_was_identified = issues != ["Operator note received; the issue requires human review."]

    if cause_unknown:
        unresolved.append("The note explicitly states that the root cause has not been identified.")
    if inspection_pending:
        if re.search(r"\bcooling inspection\b", lower):
            unresolved.append("Cooling inspection is pending and has not been confirmed as completed.")
        else:
            unresolved.append("Inspection or maintenance is explicitly reported as pending or incomplete.")
    if verification_pending:
        unresolved.append("Post-maintenance verification run is pending; repair completion does not confirm that verification has passed.")
    if repair_pending:
        unresolved.append("Repair or rectification is explicitly reported as pending or incomplete.")
    completed_repair = bool(re.search(r"\b(?:completed the repair|repair (?:was )?completed|repaired|repair has been completed)\b", lower))
    if issue_was_identified and not cause_unknown and not inspection_pending and not verification_pending and not repair_pending and not completed_repair:
        unresolved.append("The reported issue requires assessment. The note does not establish whether the cause has been identified or whether the machine is ready for normal operation.")
    if status == "Stopped":
        unresolved.append("The machine is reported stopped. Its current physical condition and maintenance status require verification.")
    elif status == "Conflicting information":
        unresolved.append("The note contains conflicting information about the machine's operating status.")
    if has_pattern(lower, [r"\b(?:not fixed|unresolved|still occurring|continues|not resolved|problem remains|issue remains)\b"]):
        unresolved.append("The note indicates that the reported problem may remain unresolved.")
    if tests_missing:
        unresolved.append("Final test results were not recorded. The outcome of the reported testing remains unverified.")
    if restart_suggested:
        unresolved.append("A restart was suggested, but the note does not establish that restart authorization has been granted.")
    elif restart_mentioned:
        unresolved.append("Restart is mentioned. The context and any required authorization must be independently verified.")
    if isolation_unconfirmed:
        unresolved.append("Power/energy isolation is explicitly reported as unconfirmed. Do not assume isolation has been completed or verified.")
    if approval_pending:
        unresolved.append("Restart approval is explicitly reported as pending. Approval has not been established by the operator note.")
    if not unresolved:
        unresolved.append("The note does not provide enough information to determine whether any issue remains unresolved. Human verification is required.")

    if cause_unknown:
        actions.append("Investigate and document the root cause through the approved maintenance process.")
    if inspection_pending:
        if re.search(r"\bcooling inspection\b", lower):
            actions.append("Complete the pending cooling inspection through approved site procedures and document the findings.")
        else:
            actions.append("Confirm the required inspection or maintenance is completed and documented by authorized personnel.")
    if verification_pending:
        actions.append("Complete the pending post-maintenance verification run according to approved site procedures, record the results, and have qualified personnel verify the machine's condition.")
    if repair_pending:
        actions.append("Complete the pending repair through the approved maintenance process and document the work performed.")
    if issue_was_identified and not cause_unknown and not inspection_pending and not verification_pending and not repair_pending and not completed_repair:
        actions.append("Review the reported issue and verify diagnostic findings, maintenance status, and the machine's current condition.")
    if tests_missing:
        actions.append("Retrieve or perform the required post-maintenance tests under approved procedures, and document the verified results before an operational decision.")
    if status == "Stopped" or restart_mentioned:
        actions.append("Follow site safety and isolation procedures. Verify the machine's condition and obtain required authorization before any restart decision.")
    if status == "Conflicting information":
        actions.append("Confirm the actual machine state with authorized site personnel before recording its operating status.")
    actions.extend(assigned_actions)
    if isolation_unconfirmed:
        actions.append("Confirm and document the applicable power/energy isolation status through authorized site procedures before work or an operational decision.")
    if approval_pending:
        actions.append("Obtain and document the required restart approval from authorized personnel. Do not treat pending approval as granted.")
    if not actions:
        actions.append("Verify the reported information and follow the site's approved maintenance and safety procedures.")

    if machine_id == "Not provided":
        missing.append("Machine identifier is missing.")
    if cause_unknown:
        missing.append("Verified root-cause findings explaining the reported issue.")
    elif not re.search(r"\b(?:cause|root cause)\b", lower):
        missing.append("Root cause or diagnostic findings were not provided.")
    if inspection_pending:
        missing.append("Inspection or maintenance completion details.")
    elif re.search(r"\bpressure leak\b", lower):
        missing.append("Pressure-leak inspection findings and repair completion details were not provided.")
    elif completed_work and verification_pending:
        missing.append("Post-repair inspection findings and verification results were not provided.")
    elif not re.search(r"\b(?:inspection|inspected|diagnostic|maintenance)\b", lower):
        missing.append("Inspection or maintenance findings and completion status were not provided.")
    if verification_pending:
        missing.append("Post-maintenance verification run results and documented outcome.")
    if status == "Not stated":
        missing.append("Current machine operating status was not provided.")
    if isolation_unconfirmed:
        missing.append("Power/energy isolation has not been confirmed; verified isolation status is still required.")
    elif status == "Stopped" or restart_mentioned:
        missing.append("Confirmation of applicable isolation requirements.")
    if approval_pending:
        missing.append("Restart approval is pending; documented authorization has not yet been established.")
    elif restart_mentioned:
        missing.append("Documented restart authorization status.")
    if not has_assigned_owner(note):
        missing.append("Named responsible person or assigned team for follow-up.")
    if not has_event_date_or_time(note):
        missing.append("Event date and time or the relevant shift details.")
    if tests_missing:
        missing.append("Final post-maintenance test results were not recorded; retrieve and document the verified results.")
    elif re.search(r"\btest(?:ed|ing|s)?\b", lower) and not re.search(r"\b(?:test result|test results|results were|results are|passed|failed)\b", lower):
        missing.append("Machine test results.")
    if not missing:
        missing.append("Verify whether additional site-specific handover information is required.")

    return {
        "machine_id": machine_id,
        "reported_issues": issues,
        "machine_status": status,
        "completed_work": completed_work,
        "unresolved_issues": unique_items(unresolved),
        "next_shift_actions": unique_items(actions),
        "missing_information": unique_items(missing),
    }


def build_safety_flags(note, report):
    flags = []
    if restart_is_mentioned(note):
        flags.append("Restart mentioned: independently verify machine condition, applicable isolation requirements, and required authorization. A suggestion is not approval.")
    if isolation_is_unconfirmed(note):
        flags.append("Power/energy isolation is explicitly unconfirmed: do not assume isolation is complete or verified. Follow approved site isolation procedures.")
    if restart_approval_is_pending(note):
        flags.append("Restart approval is explicitly pending: do not treat pending approval as authorization to restart.")
    if report["machine_status"] == "Stopped":
        flags.append("Machine reported stopped: do not infer that it is safe to restart from the handover note.")
    if report["machine_status"] == "Conflicting information":
        flags.append("Conflicting status: confirm the actual machine state with authorized site personnel.")
    if cause_is_unknown(note):
        flags.append("Unknown root cause: investigate the reported fault and document the findings.")
    if inspection_is_pending(note):
        flags.append("Pending inspection or maintenance: confirm completion and documentation before an operational decision.")
    if verification_is_pending(note):
        flags.append("Post-maintenance verification is pending: document and review the required verification results before an operational decision.")
    if repair_is_pending(note):
        flags.append("Repair or rectification is reported as pending: confirm completion through approved maintenance procedures before an operational decision.")
    if re.search(r"\boverheat(?:ing|ed)?\b|\bhigh temperature\b", note, re.IGNORECASE):
        flags.append("Overheating or high temperature reported: follow site procedures and have authorized personnel assess the condition.")
    if re.search(r"\bvibrat(?:e|ion|ing|ed)\b", note, re.IGNORECASE):
        flags.append("Vibration reported: have authorized personnel assess the condition and review relevant inspection findings before an operational decision.")
    if re.search(r"\b(?:damaged bearing|bearing damage|bearing failure|bearing problem|faulty bearing)\b", note, re.IGNORECASE):
        flags.append("Bearing damage or a bearing problem reported: verify repair records and required post-maintenance checks with authorized personnel.")
    if test_results_missing(note):
        flags.append("Test results were not recorded: verify and document the required results before an operational decision.")
    if re.search(r"\b(?:smoke|fire|sparks|electric shock|gas leak)\b", note, re.IGNORECASE):
        flags.append("Potentially hazardous condition mentioned: follow the site's emergency and escalation procedures.")
    if not flags:
        flags.append("No specific safety phrase was detected by the basic checks. This does not mean the machine is safe.")
    return unique_items(flags)


# Optional local Qwen + LoRA model. The rule-based report remains authoritative.
@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load the LoRA adapter. Install with: python -m pip install peft") from exc
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    return tokenizer, model


def generate_raw_output(note, tokenizer, model):
    """Generate an optional comparison draft; never use it as the authoritative report."""
    import torch
    prompts = [
        (
            "### Instruction:\nExtract facts from the factory shift note and return one valid JSON object "
            "with these seven fields: machine_id, reported_issues, machine_status, completed_work, "
            "unresolved_issues, next_shift_actions, missing_information. Use only facts supported by "
            "the note. Preserve exact alarm codes, named task owners, pending isolation and approval "
            "states. Do not invent repairs, inspection results, measurements, approvals, or safety status. "
            "A restart suggestion is not authorization.\n\n"
            f"### Factory shift note:\n{note}\n\n### Structured handover report:\n"
        ),
        (
            "Return only a JSON object with seven fields: machine_id, reported_issues, machine_status, "
            "completed_work, unresolved_issues, next_shift_actions, missing_information. Preserve exact "
            "alarm codes, technician assignments, and explicit unconfirmed/pending safety states. "
            "Do not invent facts or authorize a restart.\n"
            f"Operator note: {note}\nJSON:"
        ),
    ]
    last_problem = "The model returned no text."
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=260, min_new_tokens=8, do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
        if raw:
            return raw
        last_problem = "The model emitted only empty/special tokens for the prompt."
    raise RuntimeError(
        f"LoRA draft generation returned no usable text after two prompt attempts. {last_problem} "
        "Check that the adapter exists, matches Qwen/Qwen2.5-0.5B, and was trained with a compatible prompt template."
    )


def parse_model_json(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def normalize_model_draft(data, rule_report):
    if not isinstance(data, dict):
        return None
    def as_list(value):
        if isinstance(value, list):
            return unique_items([str(x) for x in value if x is not None])
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
    # Rule-derived facts/actions remain authoritative. The model is comparison-only.
    return {
        "machine_id": rule_report["machine_id"],
        "reported_issues": rule_report["reported_issues"],
        "machine_status": rule_report["machine_status"],
        "completed_work": rule_report["completed_work"],
        "unresolved_issues": unique_items(rule_report["unresolved_issues"] + as_list(data.get("unresolved_issues"))),
        "next_shift_actions": list(rule_report["next_shift_actions"]),
        "missing_information": unique_items(rule_report["missing_information"] + as_list(data.get("missing_information"))),
    }


# Header and safety notice
st.markdown('''<div class="hero"><div class="hero-kicker">INDUSTRIAL OPERATIONS · AI ASSISTANT</div><h1>🏭 ShiftSync AI</h1><p>Intelligent Factory Shift Handover Assistant<br>Turn operator notes into structured handovers, identify missing information, and highlight issues for human review.</p></div>''', unsafe_allow_html=True)
st.markdown('''<div class="notice"><strong>Prototype safety notice:</strong> This application may miss or incorrectly interpret information. It does not certify machine safety, replace site procedures, or authorize machinery operation or restart. A qualified person must independently verify the report.</div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ShiftSync AI")
    st.caption("Factory shift handover")
    st.divider()
    st.markdown("### Processing")
    st.markdown("**Primary report**  \nRule-based extraction  \n\n**Optional model**  \nQwen2.5-0.5B + LoRA")
    use_model = st.checkbox("Run optional local LoRA model", value=False, help="The model may take several minutes and use significant memory. The primary report works without it.")
    st.divider()
    st.markdown("### Workflow")
    st.markdown("1. Enter the operator note.\n2. Generate the report.\n3. Review safety flags.\n4. Edit and verify fields.\n5. Confirm review and export JSON.")
    st.divider()
    st.caption("Human review is required for every report.")

st.markdown('<div class="section-heading">01 · Operator handover note</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Enter the operator notes as reported. Missing details will not be treated as verified.</div>', unsafe_allow_html=True)
example_note = ("Machine M-205 experienced overheating during operation. The operator stopped the machine. The cause of overheating has not been identified, and maintenance inspection is still pending. One operator suggested restarting the machine immediately.")
if "operator_note" not in st.session_state:
    st.session_state.operator_note = example_note
with st.container(border=True):
    note = st.text_area("Operator note", key="operator_note", height=155, placeholder="Enter the original operator handover note...", label_visibility="collapsed")
    generate_clicked = st.button("Generate handover report", type="primary", use_container_width=True)

if "report_version" not in st.session_state:
    st.session_state.report_version = 0
if generate_clicked:
    if not clean_text(note):
        st.warning("Please enter an operator note before generating a report.")
    else:
        st.session_state.report_version += 1
        current_note = clean_text(note)
        primary_report = build_note_based_report(current_note)
        st.session_state.primary_report = primary_report
        st.session_state.report_source_note = current_note
        st.session_state.raw_model_output = ""
        st.session_state.model_draft = None
        st.session_state.model_error = ""
        st.session_state.human_reviewed = False
        if use_model:
            try:
                with st.spinner("Loading the local model and generating a draft. This may take a few minutes on a CPU."):
                    tokenizer, model = load_model()
                    raw = generate_raw_output(current_note, tokenizer, model)
                st.session_state.raw_model_output = raw
                parsed = parse_model_json(raw)
                if parsed is not None:
                    st.session_state.model_draft = normalize_model_draft(parsed, primary_report)
                else:
                    st.session_state.model_error = "The model output was not valid JSON. The primary report remains available."
            except Exception as exc:
                st.session_state.model_error = f"{type(exc).__name__}: {exc}"

if "primary_report" in st.session_state:
    report = st.session_state.primary_report
    source_note = st.session_state.report_source_note
    version = st.session_state.report_version
    st.divider()
    st.markdown('<div class="section-heading">02 · Machine overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Extracted from the operator note. Status is not independently verified.</div>', unsafe_allow_html=True)
    machine_id_html = html.escape(report["machine_id"])
    status_html = html.escape(report["machine_status"])
    status_color = "#B45309" if report["machine_status"] in ["Stopped", "Conflicting information"] else ("#0F766E" if report["machine_status"] == "Running" else "#64748B")
    flags = build_safety_flags(source_note, report)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Machine ID</div><div class="stat-value">{machine_id_html}</div><div class="stat-note">Extracted from operator note</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Reported status</div><div class="stat-value" style="color:{status_color}">{status_html}</div><div class="stat-note">Not an independent machine check</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Review flags</div><div class="stat-value">{len(flags)}</div><div class="stat-note">Require human verification</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">03 · Handover report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Review and edit every field before exporting.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        edited_machine_id = st.text_input("Machine ID", value=report["machine_id"], key=f"edited_machine_id_{version}")
        status_options = ["Not stated", "Stopped", "Running", "Conflicting information"]
        status_index = status_options.index(report["machine_status"]) if report["machine_status"] in status_options else 0
        edited_status = st.selectbox("Reported machine status", options=status_options, index=status_index, key=f"edited_machine_status_{version}", help="Based on the note, not a live machine connection.")
        st.markdown("#### Reported issues")
        issues_text = st.text_area("Reported issues", value="\n".join(report["reported_issues"]), key=f"edited_issues_{version}", height=100, help="Use one item per line.")
        st.markdown("#### Completed work")
        completed_text = st.text_area("Completed work", value="\n".join(report["completed_work"]), key=f"edited_completed_{version}", height=90, help="Only include work explicitly reported as completed.")
        st.markdown("#### Unresolved issues")
        unresolved_text = st.text_area("Unresolved issues", value="\n".join(report["unresolved_issues"]), key=f"edited_unresolved_{version}", height=145)
        st.markdown("#### Next-shift actions")
        actions_text = st.text_area("Next-shift actions", value="\n".join(report["next_shift_actions"]), key=f"edited_actions_{version}", height=150)
        st.markdown("#### Missing information")
        missing_text = st.text_area("Missing information", value="\n".join(report["missing_information"]), key=f"edited_missing_{version}", height=150)

    edited_report = {
        "machine_id": clean_text(edited_machine_id) or "Not provided",
        "reported_issues": unique_items(issues_text.splitlines()),
        "machine_status": edited_status,
        "completed_work": unique_items(completed_text.splitlines()),
        "unresolved_issues": unique_items(unresolved_text.splitlines()),
        "next_shift_actions": unique_items(actions_text.splitlines()),
        "missing_information": unique_items(missing_text.splitlines()),
    }

    st.markdown('<div class="section-heading">04 · Safety review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Rule-based flags for concerns that need independent verification. They cannot guarantee safety.</div>', unsafe_allow_html=True)
    for flag in build_safety_flags(source_note, edited_report):
        st.markdown(f'<div class="flag-card">⚠️ &nbsp; {html.escape(flag)}</div>', unsafe_allow_html=True)
    st.markdown('''<div class="safe-notice"><strong>Required human review</strong><br>Compare every field against the original note and site records. Do not treat an AI-generated report or the absence of a flag as proof that a machine is safe to operate.</div>''', unsafe_allow_html=True)

    if st.session_state.get("model_error"):
        st.warning("The optional model did not complete successfully. The primary report is still available.")
        with st.expander("Model diagnostic details"):
            st.code(st.session_state.model_error)
    if st.session_state.get("model_draft") is not None:
        with st.expander("Optional model draft — comparison only"):
            st.caption("This draft supplements rule-detected concerns. It is not an independent safety assessment.")
            st.json(st.session_state.model_draft)
    if st.session_state.get("raw_model_output"):
        with st.expander("Advanced: raw model output"):
            st.caption("Unmodified model output. It may be incomplete, invalid or incorrect.")
            st.code(st.session_state.raw_model_output, language="text")

    st.markdown('<div class="section-heading">05 · Review and export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Confirm that you have reviewed the report before downloading the JSON file.</div>', unsafe_allow_html=True)
    with st.expander("Preview final JSON"):
        st.json(edited_report)
    reviewed = st.checkbox("I have reviewed the report against the original note and will follow the applicable site procedures.", key=f"human_reviewed_{version}")
    st.caption("Exporting the file does not approve maintenance or authorize a machine restart.")
    json_bytes = json.dumps(edited_report, indent=4, ensure_ascii=False).encode("utf-8")
    safe_filename_id = re.sub(r"[^A-Za-z0-9_-]+", "_", edited_report["machine_id"])
    download_col, reset_col = st.columns([3, 1])
    with download_col:
        st.download_button("Download reviewed JSON", data=json_bytes, file_name=f"shiftsync_handover_{safe_filename_id}.json", mime="application/json", disabled=not reviewed, use_container_width=True, key=f"download_json_{version}")
    with reset_col:
        if st.button("Reset", use_container_width=True):
            for key in ["primary_report", "report_source_note", "raw_model_output", "model_draft", "model_error", "human_reviewed", "report_version"]:
                st.session_state.pop(key, None)
            st.session_state.operator_note = example_note
            st.rerun()

st.markdown('''<div class="footer"><strong>ShiftSync AI</strong> · Intelligent Factory Shift Handover Assistant<br>Prototype for demonstration and human-reviewed handovers. Not a safety certification or machine-control system.</div>''', unsafe_allow_html=True)
