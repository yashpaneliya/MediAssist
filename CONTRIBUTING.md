# 🤝 Contributing to MediAssist API

Hey there! 👋 Thanks for taking the time to contribute to this project. This guide outlines how we work together on the codebase.

---

## 🔀 Branch Strategy

We follow a simple branching model:

- `main` → stable, production-ready code
- `dev` → default branch for all active development
- `feature/<name>` → for new features or improvements
- `bugfix/<name>` → for fixing issues or bugs

**Example:**
- feature/agent-runner-logic
- bugfix/fix-redis-timeout

> Always branch off from `main` when starting a new task else start from dependent branch like `dev` or `feature/<name>`.

---

## 🐞 Issue Reporting

Please create a Github Issue if:

- You encounter a bug
- You want to suggest a feature or improvement
- You plan to work on something — to avoid duplication

Use labels like `bug`, `enhancement`, `question` where applicable.

---

## ✅ Pull Request Strategy

- Always create a PR **into `dev`** (not `main`)
- Name your PR clearly (e.g. `Add agent runner module`)
- Link related issue(s) in the PR description using: Closes #<issue-number>
- Write a short summary of what your PR does
- Ask for review in our group or tag someone directly
- Keep PRs focused & small where possible

---
