---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email
  - nlp
  - support
  - real-world
  - agent
---

# 📧 Email Triage OpenEnv

> **A real-world email support triage environment for training and evaluating AI agents.**

This [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment simulates a customer support inbox. Agents must **classify**, **prioritise**, **reply to**, **escalate**, or **archive** support emails across three tasks of increasing difficulty.

This is a genuine real-world task — every company with a customer support function deals with email triage. Getting it right matters: missed security incidents can cost millions, while spam flooding agents' time reduces productivity.

---

## 🎯 Environment Overview

| Property | Value |
|---|---|
| **Domain** | Customer Support / Email |
| **Tasks** | 3 (easy → medium → hard) |
| **Max Episodes** | Unlimited (stateless per episode) |
| **Reward Range** | `[-0.5, 1.0]` per step |
| **Final Score Range** | `[0.0, 1.0]` |
| **Action Space** | 5 action types (classify, draft_reply, escalate, archive, request_info) |
| **Observation Space** | Current email + inbox state |

---

## 📬 Tasks

### Task 1: `classify_only` — Easy
- **Inbox**: 5 emails
- **Max Steps**: 10
- **Objective**: Classify each email with the correct **category** and **priority**.
- **What's tested**: Basic text understanding, priority reasoning, spam detection.
- **Expected baseline score**: 0.55–0.75

### Task 2: `triage_and_reply` — Medium
- **Inbox**: 8 emails
- **Max Steps**: 20
- **Objective**: Classify + draft professional replies. Escalate where required. Archive spam.
- **What's tested**: Reply quality, escalation judgment, multi-step planning.
- **Expected baseline score**: 0.40–0.65

### Task 3: `crisis_response` — Hard
- **Inbox**: 12 emails
- **Max Steps**: 30
- **Objective**: Handle a critical inbox with security incidents, SLA violations, outages, GDPR requests, and billing disputes. Every wrong decision is costly.
- **What's tested**: Priority under pressure, nuanced escalation, compliance awareness, handling adversarial/ambiguous emails.
- **Expected baseline score**: 0.30–0.55

---

## 🔧 Action Space

```python
class EmailTriageAction(BaseModel):
    action_type: ActionType       # classify | draft_reply | escalate | archive | request_info
    email_id: str                 # ID of the email being acted on
    category: Optional[Category]  # billing | technical | security | general_inquiry |
                                  # spam | outage | feature_request | compliance
    priority: Optional[Priority]  # critical | high | medium | low
    reply_text: Optional[str]     # For draft_reply and request_info
    escalation_reason: Optional[str]
    reasoning: Optional[str]      # Agent's chain-of-thought (not graded, used for debugging)
```

---

## 👁️ Observation Space

```python
class EmailTriageObservation(BaseModel):
    current_email: Optional[Email]  # The email to process
    inbox_remaining: int            # How many emails are left
    step_feedback: str              # Human-readable feedback from last action
    last_action_valid: bool
    last_action_error: Optional[str]
    episode_done: bool
    task_name: str
```

The `Email` object contains: `email_id`, `sender`, `subject`, `body`, `received_at`, `attachments`, `thread_history`.

---

## 🏆 Reward Function

Rewards are **non-sparse** — the agent receives signal on every step.

| Action | Reward Components |
|---|---|
| **CLASSIFY** | `0.4 × category_accuracy + 0.3 × priority_accuracy` |
| **DRAFT_REPLY** | `0.6 × reply_quality_score + 0.1 if also classified` |
| **ESCALATE** | `0.5 × escalation_correctness` |
| **ARCHIVE** | `+0.8` for spam, `-0.3` for real emails |
| **REQUEST_INFO** | `+0.3` if info genuinely missing, `+0.1` otherwise |
| **Critical miss** | `-0.5` penalty for archiving a critical email |
| **Wrong email_id** | `-0.05` fixed penalty |

**Reply quality** is measured by keyword coverage (domain-specific terms expected in good replies) + length bonus + professionalism bonus.

**Final episode score** = `grade_episode(processed_records)` → normalized 0.0–1.0.

---

## 🚀 Quick Start

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

Then test:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_name": "classify_only"}'
```

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Python Client

```python
from client import EmailTriageEnv
from models import EmailTriageAction, ActionType, Category, Priority

with EmailTriageEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_name="triage_and_reply")
    email = result.observation.current_email

    action = EmailTriageAction(
        action_type=ActionType.DRAFT_REPLY,
        email_id=email.email_id,
        category=Category.BILLING,
        priority=Priority.HIGH,
        reply_text="Dear Alice, We've identified the duplicate charge and will process a refund within 3-5 business days. We apologize for the inconvenience. — Support Team",
    )
    result = env.step(action)
    print(f"Reward: {result.reward:.2f} | Feedback: {result.observation.step_feedback}")
```

---

## 🤖 Baseline Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export EMAIL_TRIAGE_URL="http://localhost:7860"

# Run all three tasks
python inference.py
```

**Expected output:**
```
[START] task=classify_only env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify(email=email_001, cat=billing, pri=high) reward=0.70 done=false error=null
[STEP] step=2 action=classify(email=email_002, cat=technical, pri=critical) reward=0.70 done=false error=null
...
[END] success=true steps=5 score=0.612 rewards=0.70,0.70,0.40,0.70,0.80

[START] task=triage_and_reply env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=14 score=0.521 rewards=...

[START] task=crisis_response env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=22 score=0.443 rewards=...
```

---

## 🧪 Tests

```bash
pip install pytest pytest-asyncio
pytest test_environment.py -v
```

---

## 📁 Project Structure

```
email_triage_env/
├── __init__.py           # Public API exports
├── models.py             # Pydantic models (Action, Observation, State, StepResult)
├── client.py             # Async + sync Python client
├── inference.py          # Baseline inference script (OpenAI client)
├── test_environment.py   # Unit + integration tests
├── openenv.yaml          # OpenEnv spec manifest
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server (HTTP + WebSocket)
    └── environment.py    # Core logic: tasks, graders, reward functions
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/info` | Task metadata |
| `POST` | `/reset` | Start episode `{task_name, session_id?}` |
| `POST` | `/step` | Take action `{action, session_id}` |
| `GET` | `/state` | Full episode state `?session_id=...` |
| `WS` | `/ws` | WebSocket persistent session |

---

## 🏗️ Design Decisions

- **Real-world utility**: Email triage is a universal business problem. Agents trained here transfer to production support tools.
- **Non-sparse rewards**: Every action gets a meaningful reward signal, making RL feasible.
- **Partial credit**: Wrong-but-close classifications (billing vs compliance) get partial reward — more realistic than binary scoring.
- **Severity-aware penalties**: Archiving a critical security email costs `-0.5`, teaching agents that stakes matter.
- **Reply quality via keywords**: Rather than LLM-judged quality (expensive), we use domain keyword coverage as a proxy that remains deterministic and reproducible.
- **Escalation as a first-class action**: Forces the agent to reason about human-in-the-loop handoffs — critical for real production deployments.

---

## 📊 Baseline Scores (Qwen2.5-72B)

| Task | Score | Notes |
|---|---|---|
| classify_only | ~0.61 | Struggles with billing vs compliance distinction |
| triage_and_reply | ~0.52 | Reply quality varies; often over-escalates |
| crisis_response | ~0.44 | Misses some subtle security priorities |

---

## 📜 License

MIT License. Built for the OpenEnv Hackathon by Meta & Hugging Face.