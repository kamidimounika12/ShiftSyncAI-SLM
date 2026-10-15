"""Lightweight regression tests for ShiftSync AI's rule-based report.
Run from the project folder with: python test_shiftsync_logic.py
This tests report logic and JSON serialization; it does not certify safety.
"""
import ast
import json
import re
from pathlib import Path

APP_FILE = Path(__file__).with_name("app.py")
source = APP_FILE.read_text(encoding="utf-8")
tree = ast.parse(source)
function_names = {
    "clean_text", "unique_items", "has_pattern", "extract_machine_id",
    "detect_machine_status", "cause_is_unknown", "inspection_is_pending",
    "verification_is_pending", "repair_is_pending", "restart_is_mentioned",
    "restart_is_suggested", "isolation_is_unconfirmed",
    "restart_approval_is_pending", "extract_alarm_codes",
    "extract_assigned_actions", "has_event_date_or_time", "has_assigned_owner",
    "extract_reported_issues", "detect_completed_maintenance",
    "test_results_missing", "build_note_based_report", "build_safety_flags",
}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
namespace = {"re": re, "json": json}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP_FILE), "exec"), namespace)

CASES = [
    ("Test 1", "Machine M-210 experienced abnormal vibration. The technician replaced the worn bearing and completed the repair. A verification run is pending."),
    ("Test 2", "Machine M-105 overheated during operation. The operator stopped the machine. Cooling inspection is pending, and technician Suresh will check it during the next shift."),
    ("Test 3", "Machine M-230 developed a pressure leak. Repair is pending, and technician Ravi is assigned to inspect it on the next shift."),
    ("Test 4", "Machine M-315 showed a pressure drop and alarm code E42. The root cause is unknown. Technician Priya will investigate during the next shift."),
    ("Test 5", "Machine M-420 overheated and was stopped. The power isolation status has not been confirmed. Restart approval is pending."),
]

for name, note in CASES:
    report = namespace["build_note_based_report"](note)
    flags = namespace["build_safety_flags"](note, report)
    # This mirrors the app's JSON export serialization and verifies it round-trips.
    encoded = json.dumps(report, indent=4, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded == report, f"{name}: JSON export did not round-trip"
    assert set(report) == {"machine_id", "reported_issues", "machine_status", "completed_work", "unresolved_issues", "next_shift_actions", "missing_information"}
    assert all(isinstance(report[key], list) for key in ("reported_issues", "completed_work", "unresolved_issues", "next_shift_actions", "missing_information"))
    print(f"PASS {name}: report structure and JSON serialization")

# Specific regression checks for previously observed misses.
r3 = namespace["build_note_based_report"](CASES[2][1])
assert any("pressure leak" in issue.lower() for issue in r3["reported_issues"])
assert not any(issue == "A leak was reported." for issue in r3["reported_issues"])

r4 = namespace["build_note_based_report"](CASES[3][1])
assert any("E42" in issue for issue in r4["reported_issues"])
assert any("Priya" in action and "investigate" in action for action in r4["next_shift_actions"])
assert any("root cause" in item.lower() for item in r4["unresolved_issues"])

r5 = namespace["build_note_based_report"](CASES[4][1])
f5 = namespace["build_safety_flags"](CASES[4][1], r5)
assert any("isolation" in item.lower() and "unconfirmed" in item.lower() for item in r5["unresolved_issues"])
assert any("approval" in item.lower() and "pending" in item.lower() for item in r5["unresolved_issues"])
assert any("isolation" in item.lower() for item in f5)
assert any("approval" in item.lower() for item in f5)

print("PASS targeted regressions: pressure leak, Priya assignment, E42, unknown cause, isolation, pending approval")
print("ALL REGRESSION CHECKS PASSED (prototype logic only; not a safety certification)")
