# AI Handoff Guide

If a new AI assistant takes over this repo, it should:

1. Read [graphify-out/GRAPH_REPORT.md](../graphify-out/GRAPH_REPORT.md).
2. Inspect [graphify-out/graph.json](../graphify-out/graph.json).
3. Open [graphify-out/graph.html](../graphify-out/graph.html).
4. Read [AI_ONBOARDING.md](../AI_ONBOARDING.md).
5. Review [README.md](../README.md).

## What to Understand Before Editing

- which files are entrypoints versus reusable modules
- which outputs are artifacts versus source code
- which pipeline feeds which downstream model
- which folders are persistent memory versus operational outputs

## What Not To Do

- do not rewrite stable pipelines unnecessarily
- do not treat Graphify as product code
- do not break the processed artifact contract
- do not add duplicate model logic in notebooks
