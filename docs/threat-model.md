# Threat model

| Threat | Control in MVP | Residual risk |
|---|---|---|
| Prompt injection directs a tool | deterministic marker denial; unknown tools default-deny | semantic attacks require stronger classifiers and isolation |
| Tool privilege escalation | registered-tool allowlist and per-tool scopes | no user identity/RBAC yet |
| Sensitive external side effect | human approval before send/publish | approver authenticity needs OIDC |
| Secret/PII leakage in observability | basic secret-pattern redaction before persistence | PII classification is deliberately limited |
| Audit alteration | append-only application API/event ordering | SQLite is not cryptographically tamper-evident |
| Availability/cost abuse | task length bound; local fixed simulated cost | rate limits and budgets are roadmap items |
