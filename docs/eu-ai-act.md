# EU AI Act engineering readiness mapping

This is an engineering-readiness mapping, **not legal advice or a compliance guarantee**. Applicability and risk classification require a factual/legal assessment by the deploying organisation.

| Engineering concern | TrustMesh evidence |
|---|---|
| Risk management and human oversight | policy risk tiers plus explicit approval/resume flow |
| Logging and traceability | trace IDs, ordered audit events, reviewer decisions, structured logs |
| Accuracy/robustness/cybersecurity | versioned offline eval dataset; default-deny tools; abuse tests |
| Transparency to operators | dashboard states, policy reason, OpenAPI contract |
| Data governance | minimal local data flow and redaction hook; retention policy remains deployment work |

For high-risk-system obligations, organisations should additionally establish quality management, data governance, technical documentation, post-market monitoring, incident processes, and conformity assessment appropriate to their role.
