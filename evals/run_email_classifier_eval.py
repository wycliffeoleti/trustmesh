import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustmesh.email_classifier_eval_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
