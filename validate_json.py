
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {
    "machine_id",
    "reported_issues",
    "machine_status",
    "completed_work",
    "unresolved_issues",
    "next_shift_actions",
    "missing_information",
}

LIST_FIELDS = {
    "reported_issues",
    "completed_work",
    "unresolved_issues",
    "next_shift_actions",
    "missing_information",
}


def validate_json(file_path):
    path = Path(file_path)

    if not path.is_file():
        print(f"FAIL: File not found: {path}")
        return False

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            report = json.load(file)

        if not isinstance(report, dict):
            raise ValueError("Top-level JSON must be an object.")

        missing = REQUIRED_FIELDS - report.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")

        if not isinstance(report["machine_id"], str) or not report["machine_id"].strip():
            raise ValueError("machine_id must be a non-empty string.")

        if not isinstance(report["machine_status"], str):
            raise ValueError("machine_status must be a string.")

        for field in LIST_FIELDS:
            if not isinstance(report[field], list):
                raise ValueError(f"{field} must be a list.")

            if not all(isinstance(item, str) for item in report[field]):
                raise ValueError(f"Every item in {field} must be a string.")

        print(f"PASS: {path.name}")
        print(f"Machine ID: {report['machine_id']}")
        print("JSON syntax and required fields are valid.")
        return True

    except (json.JSONDecodeError, OSError, ValueError) as error:
        print(f"FAIL: {path.name}: {error}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python validate_json.py "path-to-report.json"')
        sys.exit(2)

    results = [validate_json(name) for name in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
