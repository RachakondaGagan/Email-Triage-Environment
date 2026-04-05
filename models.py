"""
Email Triage Environment — Pydantic Models
==========================================
Typed contracts for Action, Observation, State, and StepResult.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    CRITICAL = "critical"    # Service outage, security breach
    HIGH     = "high"        # Billing dispute, data loss
    MEDIUM   = "medium"      # Feature request, general question
    LOW      = "low"         # Feedback, newsletter subscription


class Category(str, Enum):
    BILLING         = "billing"
    TECHNICAL       = "technical"
    SECURITY        = "security"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM            = "spam"
    OUTAGE          = "outage"
    FEATURE_REQUEST = "feature_request"
    COMPLIANCE      = "compliance"


class ActionType(str, Enum):
    CLASSIFY     = "classify"       # Set category + priority
    DRAFT_REPLY  = "draft_reply"    # Write a reply to the email
    ESCALATE     = "escalate"       # Escalate to human agent
    ARCHIVE      = "archive"        # Archive as resolved / irrelevant
    REQUEST_INFO = "request_info"   # Ask customer for more info


# ---------------------------------------------------------------------------
# Email dataclass (used in Observation)
# ---------------------------------------------------------------------------

class Email(BaseModel):
    email_id: str
    sender: str
    subject: str
    body: str
    received_at: str                 # ISO-8601 string
    attachments: List[str] = []      # attachment names
    thread_history: List[Dict[str, str]] = []  # previous messages


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class EmailTriageAction(BaseModel):
    """Agent action for one email."""
    action_type: ActionType
    email_id: str

    # For CLASSIFY
    category: Optional[Category] = None
    priority: Optional[Priority] = None

    # For DRAFT_REPLY / REQUEST_INFO
    reply_text: Optional[str] = None

    # For ESCALATE
    escalation_reason: Optional[str] = None

    # Shared notes / justification (used in grading)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailTriageObservation(BaseModel):
    """What the agent sees after each step."""
    current_email: Optional[Email] = None
    inbox_remaining: int = 0            # How many emails left
    step_feedback: str = ""             # Human-readable feedback on last action
    last_action_valid: bool = True
    last_action_error: Optional[str] = None
    episode_done: bool = False
    task_name: str = "email_triage"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ProcessedEmailRecord(BaseModel):
    email_id: str
    assigned_category: Optional[Category] = None
    assigned_priority: Optional[Priority] = None
    action_taken: Optional[ActionType] = None
    reply_draft: Optional[str] = None
    escalated: bool = False
    archived: bool = False
    step_reward: float = 0.0


class EmailTriageState(BaseModel):
    """Full observable state of the episode."""
    task_name: str
    task_description: str
    inbox: List[Email]
    processed: List[ProcessedEmailRecord] = []
    current_step: int = 0
    max_steps: int = 20
    total_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# StepResult  (matches openenv-core convention)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}