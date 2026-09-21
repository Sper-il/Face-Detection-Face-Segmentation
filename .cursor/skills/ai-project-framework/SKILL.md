---
name: ai-project-framework
description: Apply the end-to-end AI project framework (Business Understanding → Feature → Tech → AI Logic → Implementation → Demo → Test) with 5W1H gating, working rules, AI Agent 3-layer architecture, and 2-level evaluation. Use when starting/planning any AI project/feature, when the user mentions "5W1H", "progress status", "implementation plan", "AI agent framework", "Enterprise Brain", or asks for project documentation templates.
---

# AI Project Framework

End-to-end workflow + governance system for AI projects, packaged as a reusable framework.

---

## Quick Reference

| Phase | Question | Output |
|-------|----------|--------|
| 1. Business Understanding | What does the user actually need? | Requirement doc |
| 2. Features | Which features in-scope / out-of-scope? | Feature list |
| 3. Tech Solution | Which tech/architecture? | Tech stack + rationale |
| 4. AI Logic | How does the AI part work? | AI solution design |
| 5. Implementation | Plan first, then code | Plan + code |
| 6. Demo | What does it look like running? | Demo build |
| 7. Test | Correct + stable? | Test report (model + full pipeline) |

**Golden rule:** "Vibe coding" is allowed **only at Implementation**, and **only after a clear plan exists**.

---

## 5W1H Gate (Bắt buộc trước khi triển khai)

Before implementing any feature, all of these must be answered:

| # | Question | When |
|---|----------|------|
| 1 | **What** — what is being built, scope | First |
| 2 | **Why** — why this approach (vs alternatives) | Right after What |
| 3 | **Where / When / Which** — where applied, timing, choice | Before implementation |
| 4 | **How** — concrete tech, tools, code | **Last** |

If anyone (human or AI agent) proposes a "How" without What/Why/Where-When-Which → **stop and ask for clarification first**.

---

## Required Project Files

For each feature/module, create these 3 files (use templates in `templates/`):

| File | Purpose |
|------|---------|
| `progress_status_<feature>.md` | Track progress: description, goal, pipeline, subtasks |
| `description_<feature>.md` | Detailed requirement, purpose, input, output |
| `implementation_plan_<feature>.md` | Technical plan: research → build → test → evaluation |

Optionally also add `working_rules.md` (project-level rules AI/devs must read).

---

## Working Rules (AI Reference)

AI agent **must** follow these rules when coding:

1. **Naming Convention** — file/variable/function/branch naming
2. **Folder Convention** — standard repo structure
3. **Discussion Rule** — where to discuss, how to record decisions
4. **Edition Rule** — how to edit code/docs (review/version)

**Before coding, AI must read the project's `working_rules.md`.**

---

## AI Agent 3-Layer Architecture

```
                         AI Agent
                            │
              ┌─────────────┴─────────────┐
              │                           │
         Prompting                       Rule
              │                           │
    Description ──▶ Skill          (Make decision)
              │                           │
     [Behavioral Layer]           [Decision Layer]
              │                           │
              └─────────────┬─────────────┘
                             ▼
                       Brain Layer
                  (The Second Brain)
                    Knowledge Base
```

| Layer | Role |
|-------|------|
| **Behavioral Layer** (Description → Skill) | Defines *what* the agent does / how it behaves |
| **Decision Layer** (Rule) | Defines *when* to use which skill, when to ask, when to refuse |
| **Brain Layer** (Knowledge Base / Enterprise Brain) | Domain knowledge that both layers reference |

**Design principle:** Skill and Rule must both be anchored to the Brain. If the Brain is wrong/missing, both layers will fail even with correct logic.

---

## Evaluation — 2 Levels (Bắt buộc)

| Level | Measures | When |
|-------|----------|------|
| **Model level** | Accuracy, Precision/Recall, Response time, output quality | Evaluate AI logic standalone, before integration |
| **Full pipeline** | End-to-end: real input → real output, total latency, system error rate | Evaluate after full integration |

> ⚠️ **Do not stop at model level and declare the system done.** Model quality ≠ pipeline quality. Issues can hide in data pipeline, integration, accumulated latency.

---

## Context — Why It Matters

Context is decisive when working with AI. Wrong/missing context → wrong AI solution even with correct technique.

Related concepts:
- **AI Fusion** — combining multiple models/techniques/sources (not relying on one).
- **Enterprise Brain (Second Brain)** — centralized knowledge: context/domain knowledge/processes/decisions that AI references across the project.

Team roles for AI projects:

| Role | Responsibility |
|------|----------------|
| Product Owner | Define requirements, priority, product value |
| Domain Expert | Domain knowledge, ensure correctness |
| AI Architecture | Design overall AI system |
| AI Engineer | Build/train/deploy model/logic |
| AI Operator | Operate & monitor AI system post-deploy |
| Security | Secure AI data & system |

---

## Audit Checklist

When auditing an AI project (own or review), check:

```
Metric
├── Problem Define         → Is the problem defined clearly?
├── Feature → Out of Scope → Which features exist, which excluded (why)?
├── Solution → Tech → AI   → Which tech, which AI part specifically?
├── Implementation         → Code status, quality
└── Evaluation
      ├── Model level
      └── Full pipeline
```

A project with Implementation but missing Full-pipeline Evaluation is **not production-ready**.

---

## Additional Resources

- **Progress Status template** → see [templates/progress_status.md](templates/progress_status.md)
- **Description template** → see [templates/description.md](templates/description.md)
- **Implementation Plan template** → see [templates/implementation_plan.md](templates/implementation_plan.md)
- **Working Rules template** → see [templates/working_rules.md](templates/working_rules.md)
- **Full framework (verbatim)** → see [reference.md](reference.md)
