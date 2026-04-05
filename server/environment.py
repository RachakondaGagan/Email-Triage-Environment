"""
Email Triage Environment — Core Logic
======================================
Implements three tasks of increasing difficulty:
  Task 1 (easy):   classify_only    — classify 5 emails by category + priority
  Task 2 (medium): triage_and_reply — classify + draft appropriate replies for 8 emails
  Task 3 (hard):   crisis_response  — handle a 12-email inbox with security incidents,
                                       outages, compliance flags, and escalation decisions
Each task is graded 0.0–1.0.
"""
from __future__ import annotations

import random
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from models import (
    ActionType,
    Category,
    Email,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    Priority,
    ProcessedEmailRecord,
    StepResult,
)

# ---------------------------------------------------------------------------
# Seed corpus of realistic support emails
# ---------------------------------------------------------------------------

_EMAIL_CORPUS: List[Dict] = [
    # --- Billing ---
    {
        "id": "email_001",
        "sender": "alice.johnson@example.com",
        "subject": "Double charged this month",
        "body": (
            "Hi Support,\n\nI've been charged twice for my subscription this month. "
            "My invoice shows two identical $49.99 charges on March 3rd and March 4th. "
            "Please refund the duplicate charge immediately.\n\nThanks,\nAlice"
        ),
        "received_at": "2024-03-05T09:12:00Z",
        "true_category": Category.BILLING,
        "true_priority": Priority.HIGH,
        "requires_reply": True,
        "escalate_required": False,
        "ideal_reply_keywords": ["refund", "duplicate", "apologize", "billing", "resolve"],
    },
    # --- Technical ---
    {
        "id": "email_002",
        "sender": "bob.chen@techcorp.io",
        "subject": "API rate limit keeps returning 429",
        "body": (
            "Hello,\n\nOur production integration has been hitting 429 Too Many Requests "
            "errors since 14:00 UTC today despite being well within our tier limits (1000 req/min). "
            "This is blocking our customers. Request IDs: REQ-8821, REQ-8822.\n\nBob Chen, CTO TechCorp"
        ),
        "received_at": "2024-03-05T14:35:00Z",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.CRITICAL,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["investigate", "engineering", "escalate", "rate limit", "urgent"],
    },
    # --- Security ---
    {
        "id": "email_003",
        "sender": "security@partner.com",
        "subject": "URGENT: Possible credential compromise in your system",
        "body": (
            "Our security team detected what appears to be leaked API keys from your platform "
            "being used from 23 distinct IPs in Eastern Europe. Keys belong to customer IDs "
            "4421 and 4422. Please treat this as critical. Time-sensitive.\n\n— Partner Security Team"
        ),
        "received_at": "2024-03-05T03:00:00Z",
        "true_category": Category.SECURITY,
        "true_priority": Priority.CRITICAL,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["security", "revoke", "incident", "team", "immediately"],
    },
    # --- General Inquiry ---
    {
        "id": "email_004",
        "sender": "curious.user@gmail.com",
        "subject": "What are your data retention policies?",
        "body": (
            "Hi,\n\nI'm considering signing up for your service. "
            "Could you tell me how long you retain user data and whether "
            "you comply with GDPR? Thanks!"
        ),
        "received_at": "2024-03-05T10:00:00Z",
        "true_category": Category.COMPLIANCE,
        "true_priority": Priority.MEDIUM,
        "requires_reply": True,
        "escalate_required": False,
        "ideal_reply_keywords": ["gdpr", "data", "retain", "privacy", "policy"],
    },
    # --- Spam ---
    {
        "id": "email_005",
        "sender": "promo@spammersite.biz",
        "subject": "YOU WON A FREE IPHONE!!!",
        "body": "Click here to claim your free iPhone now! Limited time offer! Act fast!!! http://bit.ly/scam",
        "received_at": "2024-03-05T08:00:00Z",
        "true_category": Category.SPAM,
        "true_priority": Priority.LOW,
        "requires_reply": False,
        "escalate_required": False,
        "ideal_reply_keywords": [],
    },
    # --- Outage report ---
    {
        "id": "email_006",
        "sender": "ops@enterprise-client.com",
        "subject": "Dashboard completely down for 2 hours",
        "body": (
            "Our entire team cannot access the dashboard since 08:00 EST. "
            "This is affecting 200 users in production. We have SLA guarantees. "
            "Status page shows 'all systems operational' which is incorrect. "
            "Escalating to account manager if not resolved in 30 minutes."
        ),
        "received_at": "2024-03-05T13:05:00Z",
        "true_category": Category.OUTAGE,
        "true_priority": Priority.CRITICAL,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["investigating", "outage", "engineering", "sla", "update"],
    },
    # --- Feature request ---
    {
        "id": "email_007",
        "sender": "developer@startup.co",
        "subject": "Can you add Slack webhook integration?",
        "body": (
            "Hi team,\n\nWe'd love to see native Slack webhook support for "
            "alert notifications. Currently we're building our own middleware. "
            "Happy to beta test! Thanks,\nDev Team at Startup.co"
        ),
        "received_at": "2024-03-05T11:30:00Z",
        "true_category": Category.FEATURE_REQUEST,
        "true_priority": Priority.LOW,
        "requires_reply": True,
        "escalate_required": False,
        "ideal_reply_keywords": ["roadmap", "feature", "request", "noted", "team"],
    },
    # --- Billing + compliance ---
    {
        "id": "email_008",
        "sender": "legal@bigcorp.com",
        "subject": "Invoice discrepancy — formal notice",
        "body": (
            "Dear Accounts,\n\nWe have identified a discrepancy of $12,400 between "
            "your invoices Q4-2023 and Q1-2024 and our purchase orders. "
            "We require a corrected invoice and written explanation within 5 business days "
            "per our contract clause 8.3, failing which we will initiate a formal dispute."
        ),
        "received_at": "2024-03-04T16:00:00Z",
        "true_category": Category.BILLING,
        "true_priority": Priority.HIGH,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["invoice", "discrepancy", "finance", "review", "contact"],
    },
    # --- Technical (medium) ---
    {
        "id": "email_009",
        "sender": "user99@example.net",
        "subject": "Can't reset my password",
        "body": (
            "Hello, I've tried resetting my password 3 times but never receive the email. "
            "Checked spam too. My email is user99@example.net. Please help."
        ),
        "received_at": "2024-03-05T07:45:00Z",
        "true_category": Category.TECHNICAL,
        "true_priority": Priority.MEDIUM,
        "requires_reply": True,
        "escalate_required": False,
        "ideal_reply_keywords": ["password", "reset", "email", "account", "verify"],
    },
    # --- Security (subtle) ---
    {
        "id": "email_010",
        "sender": "hr@acmecorp.org",
        "subject": "Employee offboarding — please revoke access",
        "body": (
            "Hi,\n\nEmployee John Doe (john.doe@acmecorp.org, employee ID 8842) "
            "left the company as of March 4th. Please immediately revoke all "
            "API keys, SSO access, and data export permissions. "
            "Confirm in writing once done. — HR Department"
        ),
        "received_at": "2024-03-05T09:00:00Z",
        "true_category": Category.SECURITY,
        "true_priority": Priority.HIGH,
        "requires_reply": True,
        "escalate_required": False,
        "ideal_reply_keywords": ["revoke", "access", "confirm", "offboarding", "completed"],
    },
    # --- Outage + billing (complex) ---
    {
        "id": "email_011",
        "sender": "cto@fintech-startup.io",
        "subject": "RE: Outage + billing credit request",
        "body": (
            "This is our third email about last week's 6-hour outage. "
            "Per your SLA, we are entitled to a service credit of 30x downtime hours = $1,800. "
            "Also, our automated payments still ran during the outage window — we expect both "
            "the SLA credit AND a refund of those charges. If unresolved by EOD we will "
            "file a chargeback with our bank."
        ),
        "received_at": "2024-03-05T15:00:00Z",
        "true_category": Category.BILLING,
        "true_priority": Priority.CRITICAL,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["sla", "credit", "refund", "escalate", "resolve", "outage"],
    },
    # --- Compliance/legal ---
    {
        "id": "email_012",
        "sender": "dpo@eu-regulator.eu",
        "subject": "Formal GDPR Data Subject Access Request",
        "body": (
            "This is a formal Data Subject Access Request (DSAR) under GDPR Article 15 "
            "on behalf of our constituent, Maria Schmidt (maria.schmidt@example.de). "
            "You have 30 days to comply. Please acknowledge receipt within 72 hours."
        ),
        "received_at": "2024-03-05T08:30:00Z",
        "true_category": Category.COMPLIANCE,
        "true_priority": Priority.HIGH,
        "requires_reply": True,
        "escalate_required": True,
        "ideal_reply_keywords": ["dsar", "gdpr", "acknowledge", "compliance", "days"],
    },
]


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {
    "classify_only": {
        "description": (
            "You are a Tier-1 support agent. Your inbox contains 5 support emails. "
            "For each email, classify it with the correct CATEGORY and PRIORITY. "
            "You do not need to draft replies — just classify accurately."
        ),
        "email_ids": ["email_001", "email_002", "email_003", "email_004", "email_005"],
        "max_steps": 10,
        "difficulty": "easy",
    },
    "triage_and_reply": {
        "description": (
            "You are a senior support agent handling 8 emails. "
            "For each email: (1) CLASSIFY it with category + priority, "
            "(2) DRAFT a professional, helpful reply — or ARCHIVE spam/irrelevant emails. "
            "Escalate emails that require it. Replies will be evaluated for quality."
        ),
        "email_ids": [
            "email_001", "email_002", "email_004", "email_005",
            "email_007", "email_008", "email_009", "email_010",
        ],
        "max_steps": 20,
        "difficulty": "medium",
    },
    "crisis_response": {
        "description": (
            "CRITICAL INBOX: You are an on-call support lead. 12 emails require triage. "
            "Multiple security incidents, SLA violations, and compliance requests need urgent attention. "
            "Classify, reply, escalate, and archive as appropriate. "
            "Prioritise correctly — missing a critical security email costs heavy penalties. "
            "Your decisions will be graded on accuracy, reply quality, and escalation judgement."
        ),
        "email_ids": [
            "email_001", "email_002", "email_003", "email_004", "email_005",
            "email_006", "email_007", "email_008", "email_009", "email_010",
            "email_011", "email_012",
        ],
        "max_steps": 30,
        "difficulty": "hard",
    },
}


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _category_score(assigned: Optional[Category], true: Category) -> float:
    if assigned == true:
        return 1.0
    # Partial credit for close-but-wrong
    partial_map = {
        (Category.BILLING, Category.COMPLIANCE): 0.3,
        (Category.COMPLIANCE, Category.BILLING): 0.3,
        (Category.OUTAGE, Category.TECHNICAL): 0.4,
        (Category.TECHNICAL, Category.OUTAGE): 0.4,
        (Category.SECURITY, Category.COMPLIANCE): 0.2,
    }
    return partial_map.get((assigned, true), 0.0)


def _priority_score(assigned: Optional[Priority], true: Priority) -> float:
    order = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
    if assigned is None:
        return 0.0
    diff = abs(order.index(assigned) - order.index(true))
    return max(0.0, 1.0 - diff * 0.35)


def _reply_quality_score(reply: Optional[str], keywords: List[str]) -> float:
    """Score a reply based on coverage of expected keywords (case-insensitive)."""
    if not reply or not keywords:
        return 0.0 if keywords else 1.0
    reply_lower = reply.lower()
    hits = sum(1 for kw in keywords if kw.lower() in reply_lower)
    length_bonus = 0.1 if len(reply) > 80 else 0.0
    professional_bonus = 0.1 if any(p in reply_lower for p in ["dear", "hello", "hi ", "thank", "regards", "sincerely"]) else 0.0
    base = hits / len(keywords)
    return min(1.0, base + length_bonus + professional_bonus)


def _escalation_score(escalated: bool, required: bool) -> float:
    if required and escalated:
        return 1.0
    if not required and not escalated:
        return 1.0
    if required and not escalated:
        return 0.0   # missed critical escalation
    # escalated when not required — small penalty
    return 0.5


# ---------------------------------------------------------------------------
# Per-step reward computation
# ---------------------------------------------------------------------------

def compute_step_reward(
    action: EmailTriageAction,
    email_meta: Dict,
    task_name: str,
) -> Tuple[float, str]:
    """
    Returns (reward, feedback_string).
    reward is in [-0.5, 1.0] depending on action quality.
    """
    reward = 0.0
    parts = []

    action_type = action.action_type
    true_cat   = email_meta["true_category"]
    true_pri   = email_meta["true_priority"]
    req_reply  = email_meta["requires_reply"]
    req_esc    = email_meta["escalate_required"]
    kws        = email_meta["ideal_reply_keywords"]

    # --- CLASSIFY ---
    if action_type == ActionType.CLASSIFY:
        cat_s = _category_score(action.category, true_cat)
        pri_s = _priority_score(action.priority, true_pri)
        reward += 0.4 * cat_s + 0.3 * pri_s
        parts.append(f"category_score={cat_s:.2f} priority_score={pri_s:.2f}")

    # --- DRAFT_REPLY ---
    elif action_type == ActionType.DRAFT_REPLY:
        if not req_reply:
            reward -= 0.1   # wasting time replying to spam
            parts.append("penalty: reply sent to non-reply-needed email")
        else:
            q = _reply_quality_score(action.reply_text, kws)
            reward += 0.6 * q
            parts.append(f"reply_quality={q:.2f}")
        # Partial credit if also classified
        if action.category and action.priority:
            reward += 0.1

    # --- ESCALATE ---
    elif action_type == ActionType.ESCALATE:
        esc_s = _escalation_score(True, req_esc)
        reward += 0.5 * esc_s
        if not req_esc:
            parts.append("penalty: unnecessary escalation")
        else:
            parts.append("correct escalation")

    # --- ARCHIVE ---
    elif action_type == ActionType.ARCHIVE:
        if true_cat == Category.SPAM:
            reward += 0.8
            parts.append("correct: archived spam")
        elif not req_reply and true_pri in (Priority.LOW,):
            reward += 0.4
            parts.append("acceptable: archived low-priority")
        else:
            reward -= 0.3
            parts.append("penalty: archived important email")

    # --- REQUEST_INFO ---
    elif action_type == ActionType.REQUEST_INFO:
        # Only good if the email genuinely lacks info
        has_contact = bool(re.search(r"\b(account|id|order|ticket)\b", email_meta["body"], re.I))
        if not has_contact:
            reward += 0.3
            parts.append("reasonable: requested missing info")
        else:
            reward += 0.1
            parts.append("marginal: info already provided")

    # --- Critical miss penalty ---
    if true_pri == Priority.CRITICAL and action_type == ActionType.ARCHIVE:
        reward -= 0.5
        parts.append("CRITICAL PENALTY: archived critical email")

    reward = round(max(-0.5, min(1.0, reward)), 4)
    feedback = "; ".join(parts) if parts else "action processed"
    return reward, feedback


# ---------------------------------------------------------------------------
# Episode-end grader (0.0 – 1.0)
# ---------------------------------------------------------------------------

def grade_episode(
    processed: List[ProcessedEmailRecord],
    email_corpus: List[Dict],
    task_name: str,
) -> float:
    """Compute final normalized score for the entire episode."""
    if not processed:
        return 0.0

    corpus_map = {e["id"]: e for e in email_corpus}
    total = len(processed)
    score_sum = 0.0

    for rec in processed:
        meta = corpus_map.get(rec.email_id)
        if not meta:
            continue
        item_score = 0.0

        # Classification accuracy (weighted 40%)
        if rec.assigned_category is not None:
            item_score += 0.4 * _category_score(rec.assigned_category, meta["true_category"])
        if rec.assigned_priority is not None:
            item_score += 0.2 * _priority_score(rec.assigned_priority, meta["true_priority"])

        # Reply quality (weighted 25%)
        if meta["requires_reply"]:
            if rec.reply_draft:
                item_score += 0.25 * _reply_quality_score(rec.reply_draft, meta["ideal_reply_keywords"])
            elif rec.archived:
                if meta["true_category"] != Category.SPAM:
                    item_score -= 0.15   # archived real email
        else:
            if rec.archived:
                item_score += 0.15   # correctly archived

        # Escalation decision (weighted 15%)
        item_score += 0.15 * _escalation_score(rec.escalated, meta["escalate_required"])

        score_sum += max(0.0, min(1.0, item_score))

    raw = score_sum / total
    # Hard task bonus: scale up slightly for crisis_response
    if task_name == "crisis_response":
        raw = raw  # no artificial scaling — keep honest
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Main Environment class
# ---------------------------------------------------------------------------

_EMAIL_MAP: Dict[str, Dict] = {e["id"]: e for e in _EMAIL_CORPUS}


class EmailTriageEnvironment:
    """
    Core OpenEnv environment for email triage.

    Each episode:
      1. reset(task_name) → loads inbox of emails for the chosen task.
      2. step(action)     → processes one action on the current email.
                            When an email is fully handled, advances to next.
      3. state()          → returns full episode state.
    """

    def __init__(self, task_name: str = "triage_and_reply"):
        self._task_name = task_name
        self._state: Optional[EmailTriageState] = None
        self._inbox_meta: List[Dict] = []
        self._current_email_index: int = 0
        self._current_record: Optional[ProcessedEmailRecord] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, task_name: Optional[str] = None) -> EmailTriageObservation:
        if task_name:
            self._task_name = task_name

        task_cfg = TASKS.get(self._task_name)
        if task_cfg is None:
            raise ValueError(f"Unknown task '{self._task_name}'. Choose from: {list(TASKS)}")

        email_ids = task_cfg["email_ids"]
        inbox_emails = [
            Email(
                email_id=_EMAIL_MAP[eid]["id"],
                sender=_EMAIL_MAP[eid]["sender"],
                subject=_EMAIL_MAP[eid]["subject"],
                body=_EMAIL_MAP[eid]["body"],
                received_at=_EMAIL_MAP[eid]["received_at"],
            )
            for eid in email_ids
        ]

        self._inbox_meta = [_EMAIL_MAP[eid] for eid in email_ids]
        self._current_email_index = 0
        self._step_count = 0

        self._state = EmailTriageState(
            task_name=self._task_name,
            task_description=task_cfg["description"],
            inbox=inbox_emails,
            processed=[],
            current_step=0,
            max_steps=task_cfg["max_steps"],
            total_reward=0.0,
            done=False,
        )

        self._current_record = ProcessedEmailRecord(
            email_id=inbox_emails[0].email_id
        )

        return EmailTriageObservation(
            current_email=inbox_emails[0],
            inbox_remaining=len(inbox_emails) - 1,
            step_feedback=f"Task '{self._task_name}' started. {task_cfg['description']}",
            last_action_valid=True,
            episode_done=False,
            task_name=self._task_name,
        )

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, action: EmailTriageAction) -> StepResult:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step(), or episode is done.")

        self._step_count += 1
        self._state.current_step = self._step_count

        # Validate action targets current email
        current_email = self._state.inbox[self._current_email_index]
        error_msg: Optional[str] = None

        if action.email_id != current_email.email_id:
            error_msg = (
                f"Action email_id '{action.email_id}' doesn't match "
                f"current email '{current_email.email_id}'. Action ignored."
            )
            obs = EmailTriageObservation(
                current_email=current_email,
                inbox_remaining=len(self._state.inbox) - self._current_email_index - 1,
                step_feedback=error_msg,
                last_action_valid=False,
                last_action_error=error_msg,
                episode_done=False,
                task_name=self._task_name,
            )
            return StepResult(observation=obs, reward=-0.05, done=False,
                              info={"error": error_msg})

        # Get ground truth for this email
        meta = self._inbox_meta[self._current_email_index]

        # Compute reward
        reward, feedback = compute_step_reward(action, meta, self._task_name)
        self._state.total_reward += reward

        # Update the processing record
        rec = self._current_record
        if action.action_type == ActionType.CLASSIFY:
            rec.assigned_category = action.category
            rec.assigned_priority = action.priority
        elif action.action_type == ActionType.DRAFT_REPLY:
            rec.reply_draft = action.reply_text
            rec.action_taken = ActionType.DRAFT_REPLY
            if action.category:
                rec.assigned_category = action.category
            if action.priority:
                rec.assigned_priority = action.priority
        elif action.action_type == ActionType.ESCALATE:
            rec.escalated = True
            rec.action_taken = ActionType.ESCALATE
        elif action.action_type == ActionType.ARCHIVE:
            rec.archived = True
            rec.action_taken = ActionType.ARCHIVE
        elif action.action_type == ActionType.REQUEST_INFO:
            rec.action_taken = ActionType.REQUEST_INFO
            rec.reply_draft = action.reply_text

        rec.step_reward += reward

        # Decide whether to advance to the next email.
        # An email is "done" after DRAFT_REPLY, ARCHIVE, ESCALATE, or REQUEST_INFO.
        terminal_actions = {
            ActionType.DRAFT_REPLY,
            ActionType.ARCHIVE,
            ActionType.ESCALATE,
            ActionType.REQUEST_INFO,
        }
        if self._task_name == "classify_only":
            terminal_actions.add(ActionType.CLASSIFY)

        advance_email = action.action_type in terminal_actions

        if advance_email:
            self._state.processed.append(deepcopy(rec))
            self._current_email_index += 1

        # Check episode end conditions
        inbox_exhausted = self._current_email_index >= len(self._state.inbox)
        steps_exhausted = self._step_count >= self._state.max_steps
        done = inbox_exhausted or steps_exhausted

        if done:
            # If we ran out of steps, save any unfinished record
            if not inbox_exhausted and self._current_record:
                self._state.processed.append(deepcopy(rec))
            self._state.done = True

        # Prepare next email observation
        if not done:
            next_email = self._state.inbox[self._current_email_index]
            if advance_email:
                self._current_record = ProcessedEmailRecord(email_id=next_email.email_id)
        else:
            next_email = None

        obs = EmailTriageObservation(
            current_email=next_email,
            inbox_remaining=max(0, len(self._state.inbox) - self._current_email_index - 1),
            step_feedback=feedback,
            last_action_valid=True,
            last_action_error=None,
            episode_done=done,
            task_name=self._task_name,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self._step_count,
                "feedback": feedback,
                "total_reward": self._state.total_reward,
                "emails_processed": len(self._state.processed),
            },
        )

    # ------------------------------------------------------------------ #
    #  state                                                               #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> EmailTriageState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return deepcopy(self._state)

    # ------------------------------------------------------------------ #
    #  final_score  (call after episode done)                             #
    # ------------------------------------------------------------------ #

    def final_score(self) -> float:
        if self._state is None:
            return 0.0
        return grade_episode(self._state.processed, self._inbox_meta, self._task_name)