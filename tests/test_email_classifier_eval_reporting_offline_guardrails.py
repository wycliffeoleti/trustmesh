from __future__ import annotations

import ast
from pathlib import Path

import pytest

_NEW_MODULES = [
    "email_classifier_eval_drift.py",
    "email_classifier_eval_report.py",
    "email_classifier_eval_alert.py",
]

# Modules/attributes whose presence in actual code (not prose) would mean a module reaches
# for network I/O, environment configuration, or an external delivery transport — none of
# which Phase 4 authorizes here. Checked against the parsed AST, not raw text, so a
# docstring that explains what's absent (e.g. "no Slack dependency") cannot false-positive.
_FORBIDDEN_IMPORT_MODULES = {"requests", "httpx", "socket", "smtplib", "urllib", "slack_sdk"}
_FORBIDDEN_ATTRIBUTE_ACCESSES = {("os", "environ"), ("os", "getenv")}


@pytest.mark.parametrize("module_name", _NEW_MODULES)
def test_module_contains_no_network_env_or_webhook_imports(module_name: str) -> None:
    root = Path(__file__).resolve().parent.parent
    source = (root / "trustmesh" / module_name).read_text()
    tree = ast.parse(source, filename=module_name)

    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not (imported_modules & _FORBIDDEN_IMPORT_MODULES), (
        f"{module_name} imports a forbidden network/transport module"
    )

    attribute_accesses = {
        (node.value.id, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    assert not (attribute_accesses & _FORBIDDEN_ATTRIBUTE_ACCESSES), (
        f"{module_name} reads environment configuration directly"
    )
