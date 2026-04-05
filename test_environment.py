"""
Email Triage Environment — Unit Tests
Tests environment logic, graders, and reward functions directly (no server required).
Run: pytest test_environment.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

import pytest

from models import ActionType, Category, EmailTriageAction, Priority
from server.environment import (
    EmailTriageEnvironment,
    TASKS,
    _category_score,
    _priority_score,
    _reply_quality_score,
    _escalation_score,
    grade_episode,
)


# ---------------------------------------------------------------------------
# Unit tests for scoring helpers
# ---------------------------------------------------------------------------

class TestCategoryScore:
    def test_exact_match(self):
        assert _category_score(Category.BILLING, Category.BILLING) == 1.0

    def test_no_match(self):
        assert _category_score(Category.SPAM, Category.BILLING) == 0.0

    def test_partial_billing_compliance(self):
        s = _category_score(Category.BILLING, Category.COMPLIANCE)
        assert 0.0 < s < 1.0

    def test_partial_outage_technical(self):
        s = _category_score(Category.OUTAGE, Category.TECHNICAL)
        assert 0.0 < s < 1.0


class TestPriorityScore:
    def test_exact_match(self):
        assert _priority_score(Priority.CRITICAL, Priority.CRITICAL) == 1.0

    def test_one_off(self):
        s = _priority_score(Priority.HIGH, Priority.CRITICAL)
        assert s == 0.65  # one-off in new scale

    def test_way_off(self):
        s = _priority_score(Priority.LOW, Priority.CRITICAL)
        assert s == 0.0  # three-off = 0.0 in new scale

    def test_none_priority(self):
        assert _priority_score(None, Priority.HIGH) == 0.0


class TestReplyQuality:
    def test_all_keywords_hit(self):
        reply = "We will refund the duplicate billing charge, we sincerely apologize."
        kws = ["refund", "duplicate", "apologize", "billing"]
        score = _reply_quality_score(reply, kws)
        assert score > 0.7

    def test_no_keywords_hit(self):
        reply = "Thank you for writing to us."
        kws = ["security", "revoke", "incident"]
        score = _reply_quality_score(reply, kws)
        assert score < 0.3

    def test_empty_keywords(self):
        assert _reply_quality_score("anything", []) == 1.0

    def test_empty_reply(self):
        assert _reply_quality_score("", ["hello"]) == 0.0


class TestEscalationScore:
    def test_correct_escalation(self):
        assert _escalation_score(True, True) == 1.0

    def test_correct_no_escalation(self):
        assert _escalation_score(False, False) == 1.0

    def test_missed_escalation(self):
        assert _escalation_score(False, True) == 0.0

    def test_unnecessary_escalation(self):
        s = _escalation_score(True, False)
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# Environment integration tests
# ---------------------------------------------------------------------------

class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = EmailTriageEnvironment("classify_only")
        obs = env.reset()
        assert obs.current_email is not None
        assert obs.episode_done is False
        assert obs.inbox_remaining == 4  # 5 total, 1 shown

    def test_all_tasks_reset(self):
        for task_name in TASKS:
            env = EmailTriageEnvironment(task_name)
            obs = env.reset()
            assert obs.task_name == task_name
            assert obs.current_email is not None

    def test_invalid_task_raises(self):
        env = EmailTriageEnvironment("nonexistent_task")
        with pytest.raises(ValueError):
            env.reset()

    def test_state_after_reset(self):
        env = EmailTriageEnvironment("classify_only")
        env.reset()
        state = env.state
        assert state.task_name == "classify_only"
        assert len(state.inbox) == 5
        assert state.done is False


class TestEnvironmentStep:
    def setup_method(self):
        self.env = EmailTriageEnvironment("classify_only")
        self.obs = self.env.reset()
        self.first_email_id = self.obs.current_email.email_id

    def test_classify_action(self):
        action = EmailTriageAction(
            action_type=ActionType.CLASSIFY,
            email_id=self.first_email_id,
            category=Category.BILLING,
            priority=Priority.HIGH,
        )
        result = self.env.step(action)
        assert result.reward >= 0  # billing + high is correct for email_001
        assert result.done is False

    def test_wrong_email_id_penalized(self):
        action = EmailTriageAction(
            action_type=ActionType.CLASSIFY,
            email_id="email_999",
            category=Category.BILLING,
            priority=Priority.HIGH,
        )
        result = self.env.step(action)
        assert result.reward < 0
        assert result.observation.last_action_valid is False

    def test_archive_spam_rewarded(self):
        # Navigate to email_005 (spam) — index 4 in classify_only
        # CLASSIFY is not terminal; use DRAFT_REPLY or ARCHIVE to advance emails
        env = EmailTriageEnvironment("classify_only")
        obs = env.reset()
        # Advance through emails 001-004 using DRAFT_REPLY (terminal action)
        for _ in range(4):
            current_id = obs.current_email.email_id
            action = EmailTriageAction(
                action_type=ActionType.DRAFT_REPLY,
                email_id=current_id,
                category=Category.BILLING,
                priority=Priority.MEDIUM,
                reply_text="Thank you for contacting us. We will look into this shortly.",
            )
            result = env.step(action)
            obs = result.observation
        # Now we should be at email_005 (spam)
        assert obs.current_email is not None
        assert obs.current_email.email_id == "email_005"
        action = EmailTriageAction(
            action_type=ActionType.ARCHIVE,
            email_id=obs.current_email.email_id,
        )
        result = env.step(action)
        assert result.reward > 0.5  # archiving spam is well rewarded

    def test_episode_ends_after_all_emails(self):
        env = EmailTriageEnvironment("classify_only")
        obs = env.reset()
        done = False
        for _ in range(20):
            if done:
                break
            action = EmailTriageAction(
                action_type=ActionType.CLASSIFY,
                email_id=obs.current_email.email_id,
                category=Category.GENERAL_INQUIRY,
                priority=Priority.MEDIUM,
            )
            result = env.step(action)
            obs = result.observation
            done = result.done
        assert done is True

    def test_final_score_in_range(self):
        env = EmailTriageEnvironment("classify_only")
        obs = env.reset()
        done = False
        for _ in range(20):
            if done:
                break
            action = EmailTriageAction(
                action_type=ActionType.CLASSIFY,
                email_id=obs.current_email.email_id,
                category=Category.BILLING,
                priority=Priority.HIGH,
            )
            result = env.step(action)
            obs = result.observation
            done = result.done
        score = env.final_score()
        assert 0.0 <= score <= 1.0


class TestGrader:
    def test_empty_processed(self):
        score = grade_episode([], [], "classify_only")
        assert score == 0.0

    def test_perfect_classification(self):
        """Agent that classifies everything correctly should score well."""
        from server.environment import _EMAIL_MAP
        from models import ProcessedEmailRecord
        task_ids = TASKS["classify_only"]["email_ids"]
        meta_list = [_EMAIL_MAP[eid] for eid in task_ids]
        processed = [
            ProcessedEmailRecord(
                email_id=m["id"],
                assigned_category=m["true_category"],
                assigned_priority=m["true_priority"],
                action_taken=ActionType.CLASSIFY,
                archived=(m["true_category"] == Category.SPAM),
            )
            for m in meta_list
        ]
        score = grade_episode(processed, meta_list, "classify_only")
        assert score > 0.55  # perfect classification + correct spam archive

    def test_all_wrong_classification(self):
        from server.environment import _EMAIL_MAP
        from models import ProcessedEmailRecord
        task_ids = TASKS["classify_only"]["email_ids"]
        meta_list = [_EMAIL_MAP[eid] for eid in task_ids]
        processed = [
            ProcessedEmailRecord(
                email_id=m["id"],
                assigned_category=Category.SPAM,
                assigned_priority=Priority.LOW,
                action_taken=ActionType.CLASSIFY,
            )
            for m in meta_list
        ]
        score = grade_episode(processed, meta_list, "classify_only")
        assert score < 0.3


class TestRewardNotSparse:
    """Ensure rewards fire at each step, not just at end."""
    def test_multiple_nonzero_rewards(self):
        env = EmailTriageEnvironment("triage_and_reply")
        obs = env.reset()
        nonzero_rewards = []
        for _ in range(16):
            if obs.current_email is None:
                break
            action = EmailTriageAction(
                action_type=ActionType.DRAFT_REPLY,
                email_id=obs.current_email.email_id,
                category=Category.GENERAL_INQUIRY,
                priority=Priority.MEDIUM,
                reply_text="Thank you for reaching out. Our team will look into this and get back to you shortly. We apologize for any inconvenience.",
            )
            result = env.step(action)
            if result.reward != 0.0:
                nonzero_rewards.append(result.reward)
            obs = result.observation
        assert len(nonzero_rewards) >= 3, "Reward should be non-sparse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])