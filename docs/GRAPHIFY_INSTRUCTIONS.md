# Graphify regeneration instructions

This repository includes Graphify scaffolding to generate a repository knowledge graph used for onboarding and architecture diagrams.

To regenerate Graphify outputs locally:

1. Install Graphify (if not installed):

```bash
# Example (depends on your environment)
# pip install graphify
# or follow the project's install instructions
```

2. From the repository root run:

```powershell
.\scripts\update-graphify.ps1
# or
graphify update .
```

3. Generated files will appear under `graphify-out/`:
- `graph.html` — interactive graph
- `GRAPH_REPORT.md` — summary report
- `graph.json` — raw graph data

Notes:
- Graphify execution is not performed in CI by default to avoid heavy compute and external dependency installation.
- If you want CI generation, add a separate workflow and ensure the runner has the `graphify` tool available.
