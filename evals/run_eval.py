import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustmesh.service import ControlPlane  # noqa: E402
from trustmesh.store import EventStore  # noqa: E402

dataset = json.loads((ROOT / "evals" / "dataset.json").read_text())
with tempfile.NamedTemporaryFile(suffix=".db") as file:
    plane = ControlPlane(EventStore(file.name))
    results = []
    for case in dataset["cases"]:
        run = plane.submit(case["task"])
        actual = run["status"]
        results.append({"id": case["id"], "expected": case["expected_status"], "actual": actual, "passed": actual == case["expected_status"], "latency_ms": run["latency_ms"], "cost_usd": run["cost_usd"]})
report = {"dataset_version": dataset["version"], "total": len(results), "passed": sum(r["passed"] for r in results), "pass_rate": sum(r["passed"] for r in results) / len(results), "results": results}
(ROOT / "docs" / "verification" / "eval-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
