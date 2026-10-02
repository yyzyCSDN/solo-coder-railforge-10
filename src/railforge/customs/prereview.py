"""Cross-border freight document pre-review.

Merges four rule planes into a single, traceable conclusion:

1. customs open-control windows (open_gate / closed_gate);
2. dangerous-goods isolation (shared cargo segregation rules);
3. border time-slot capacity (border_slot);
4. train formation envelope (formation).

Conclusions are evaluated strictly "as of" the document event time by
:func:`evaluate`; the stateful wrapper :class:`CrossBorderPrereviewService`
adds the submission protocol around it:

* an open-control failure, a late revision and a conflicting resubmission are
  explicit *denials* that commit a ``rejected`` control event with a typed
  reason code;
* every denial is keyed by ``(submission_id, revision)`` and therefore replays
  deterministically — reposting the same packet returns the stored receipt
  without emitting another event or moving the event-time watermark.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Mapping

from railforge.cargo.segregation import Load, violations

CONCLUSION_TOPIC = "customs.prereview.conclusion"
REPLAY_TOPIC = "customs.prereview.replayed"

# Reason codes used on rejected conclusions.
R_CLOSED_GATE = "closed_gate"
R_ISOLATION = "isolation_conflict"
R_FORMATION = "formation_exceeded"
R_NO_SLOT = "no_border_slot"
R_LATE_REVISION = "late_revision"
R_DUPLICATE = "duplicate_conflict"
APPROVED = "approved"
REJECTED = "rejected"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Finding:
    """A single traceable rule-plane finding attached to a conclusion."""

    code: str
    plane: str
    detail: str


@dataclass(frozen=True)
class OpenControlWindow:
    """A customs open-control window at a border crossing point."""

    point: str
    start: datetime
    end: datetime

    def open_at(self, at: datetime) -> bool:
        x = _utc(at)
        return _utc(self.start) <= x < _utc(self.end)


@dataclass(frozen=True)
class BorderSlot:
    """A bookable border-crossing time slot with fixed crossing capacity."""

    slot_id: str
    point: str
    start: datetime
    end: datetime
    capacity: int = 1

    def contains(self, at: datetime) -> bool:
        x = _utc(at)
        return _utc(self.start) <= x < _utc(self.end)


@dataclass(frozen=True)
class WagonEntry:
    """A wagon line on the pre-review document."""

    wagon: str
    position: int
    cargo_group: str
    weight_t: float
    length_m: float


@dataclass(frozen=True)
class FormationLimits:
    """Train formation envelope enforced at the border."""

    max_wagons: int
    max_gross_t: float
    max_length_m: float
    hazmat_cert_required: bool = False


@dataclass(frozen=True)
class PrereviewSubmission:
    """A cross-border freight document packet submitted at an event time."""

    submission_id: str
    revision: int
    train_id: str
    point: str
    event_time: datetime
    crossing_time: datetime
    wagons: tuple[WagonEntry, ...]
    hazmat_certified: bool = False


@dataclass(frozen=True)
class PrereviewConclusion:
    """The traceable pre-review conclusion for one submission revision."""

    submission_id: str
    revision: int
    train_id: str
    point: str
    event_time: datetime
    crossing_time: datetime
    decision: str
    reason: str | None
    findings: tuple[Finding, ...]
    slot_id: str | None
    supersedes: str | None = None

    @property
    def approved(self) -> bool:
        return self.decision == APPROVED

    @staticmethod
    def denied(submission: PrereviewSubmission, reason: str,
               findings: tuple[Finding, ...], supersedes: str | None = None) -> "PrereviewConclusion":
        return PrereviewConclusion(
            submission.submission_id, submission.revision, submission.train_id,
            submission.point, _utc(submission.event_time), _utc(submission.crossing_time),
            REJECTED, reason, findings, None, supersedes,
        )


# ---------------------------------------------------------------------------
# Typed denials
# ---------------------------------------------------------------------------


class PrereviewRejected(RuntimeError):
    """An explicit pre-review denial. Carries the committed control receipt."""

    reason = "rejected"

    def __init__(self, message: str, receipt=None):
        super().__init__(message)
        self.receipt = receipt

    @property
    def conclusion(self):
        return None if self.receipt is None else self.receipt.result


class ClosedGateError(PrereviewRejected):
    reason = R_CLOSED_GATE


class IsolationConflictError(PrereviewRejected):
    reason = R_ISOLATION


class FormationExceededError(PrereviewRejected):
    reason = R_FORMATION


class NoBorderSlotError(PrereviewRejected):
    reason = R_NO_SLOT


class LateRevisionError(PrereviewRejected):
    reason = R_LATE_REVISION


class DuplicateConflictError(PrereviewRejected):
    reason = R_DUPLICATE


_REASON_ERRORS: Mapping[str, type[PrereviewRejected]] = {
    R_CLOSED_GATE: ClosedGateError,
    R_ISOLATION: IsolationConflictError,
    R_FORMATION: FormationExceededError,
    R_NO_SLOT: NoBorderSlotError,
}


# ---------------------------------------------------------------------------
# Pure, event-time evaluation
# ---------------------------------------------------------------------------


def choose_slot(slots: list[BorderSlot], point: str, crossing_time: datetime,
                booked_counts: dict[str, int] | None = None) -> BorderSlot | None:
    """Pick the earliest feasible slot for *point* that still has capacity."""
    booked_counts = booked_counts or {}
    candidates = [
        s for s in slots
        if s.point == point and s.contains(crossing_time)
        and booked_counts.get(s.slot_id, 0) < s.capacity
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: (_utc(s.start), s.slot_id))


def evaluate(submission: PrereviewSubmission,
             open_windows: list[OpenControlWindow],
             slots: list[BorderSlot],
             isolation_rules: dict[frozenset[str], int],
             formation: FormationLimits,
             booked_counts: dict[str, int] | None = None) -> PrereviewConclusion:
    """Merge the four rule planes into one conclusion, evaluated as-of event time.

    Rules are evaluated in a fixed order so the conclusion is reproducible:
    open-control, dangerous-goods isolation, train formation, border slot.
    The decision is ``approved`` only when every plane passes; otherwise the
    first failed plane supplies the denial reason and all findings are kept.
    """
    findings: list[Finding] = []
    reason: str | None = None

    def fail(code: str, plane: str, detail: str) -> None:
        nonlocal reason
        findings.append(Finding(code, plane, detail))
        if reason is None:
            reason = code

    # Plane 1 — customs open control.
    if not any(w.point == submission.point and w.open_at(submission.event_time)
               for w in open_windows):
        fail(R_CLOSED_GATE, "open_control",
             f"customs control not open at {submission.point} "
             f"as of {_utc(submission.event_time).isoformat()}")

    # Plane 2 — dangerous-goods isolation.
    loads = [
        Load(w.wagon, w.cargo_group, w.position)
        for w in sorted(submission.wagons, key=lambda x: x.position)
    ]
    bad_pairs = violations(loads, isolation_rules)
    for a_wagon, b_wagon, required, actual in bad_pairs:
        fail(R_ISOLATION, "hazmat_isolation",
             f"{a_wagon}<->{b_wagon} need {required} buffer wagon(s), have {actual}")

    # Plane 3 — train formation envelope.
    wagon_count = len(submission.wagons)
    gross_t = sum(w.weight_t for w in submission.wagons)
    length_m = sum(w.length_m for w in submission.wagons)
    if wagon_count > formation.max_wagons:
        fail(R_FORMATION, "formation",
             f"wagons {wagon_count} exceed limit {formation.max_wagons}")
    if gross_t > formation.max_gross_t:
        fail(R_FORMATION, "formation",
             f"gross {gross_t}t exceeds limit {formation.max_gross_t}t")
    if length_m > formation.max_length_m:
        fail(R_FORMATION, "formation",
             f"length {length_m}m exceeds limit {formation.max_length_m}m")
    hazmat_groups = {g for key in isolation_rules for g in key}
    carries_hazmat = any(w.cargo_group in hazmat_groups for w in submission.wagons)
    if formation.hazmat_cert_required and carries_hazmat and not submission.hazmat_certified:
        fail(R_FORMATION, "formation", "hazmat certification missing")

    # Plane 4 — border time slot (only consumed when the hard planes all pass).
    slot = None
    if reason is None:
        slot = choose_slot(slots, submission.point, submission.crossing_time, booked_counts)
        if slot is None:
            fail(R_NO_SLOT, "border_slot",
                 f"no capacity at {submission.point} "
                 f"for crossing {_utc(submission.crossing_time).isoformat()}")

    if reason is not None:
        return PrereviewConclusion.denied(submission, reason, tuple(findings))
    return PrereviewConclusion(
        submission.submission_id, submission.revision, submission.train_id,
        submission.point, _utc(submission.event_time), _utc(submission.crossing_time),
        APPROVED, None, tuple(findings), slot.slot_id if slot else None,
    )


# ---------------------------------------------------------------------------
# Stateful submission protocol (watermark, idempotency, replay)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayView:
    """Result of reposting an already-known submission revision."""

    receipt: object
    replayed: bool
    replay_count: int
    superseded: bool = False


class CrossBorderPrereviewService:
    """Event-time pre-review workflow on top of the atomic control plane."""

    def __init__(self, control):
        self.control = control
        # (submission_id, revision) -> committed receipt (approved or denied).
        self._outcomes: dict[tuple[str, int], object] = {}
        # submission_id -> highest event_time already accepted.
        self._watermark: dict[str, datetime] = {}
        # submission_id -> current revision number.
        self._head_revision: dict[str, int] = {}
        # submission_id -> revision of the slot booking currently held.
        self._booked_revision: dict[str, int] = {}
        # (submission_id, revision) -> number of replays observed.
        self._replay_count: dict[tuple[str, int], int] = {}
        # (submission_id, revision) -> SHA-ish fingerprint of the packet body.
        self._fingerprints: dict[tuple[str, int], str] = {}
        # (key, fingerprint) -> duplicate-conflict denial for a tampered body.
        self._conflicts: dict[tuple[tuple[str, int], str], object] = {}
        self._conflict_count: int = 0

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _fingerprint(submission: PrereviewSubmission) -> str:
        parts = [
            submission.submission_id, str(submission.revision), submission.train_id,
            submission.point, _utc(submission.event_time).isoformat(),
            _utc(submission.crossing_time).isoformat(), str(submission.hazmat_certified),
        ]
        for w in sorted(submission.wagons, key=lambda x: (x.position, x.wagon)):
            parts.append("|".join([w.wagon, str(w.position), w.cargo_group,
                                   str(w.weight_t), str(w.length_m)]))
        return "\n".join(parts)

    def _booked_counts(self, slots: list[BorderSlot]) -> dict[str, int]:
        """Capacity consumed by currently-held approved bookings."""
        counts: dict[str, int] = {}
        for submission_id, revision in self._booked_revision.items():
            receipt = self._outcomes.get((submission_id, revision))
            slot_id = receipt.result.slot_id if receipt else None
            if slot_id is not None:
                counts[slot_id] = counts.get(slot_id, 0) + 1
        return counts

    def _commit_conclusion(self, operation_id: str, at: datetime,
                           conclusion: PrereviewConclusion, evidence=None):
        return self.control.run(
            operation_id, CONCLUSION_TOPIC, conclusion.train_id, at,
            lambda: conclusion,
            {"submission_id": conclusion.submission_id, "revision": conclusion.revision,
             "decision": conclusion.decision,
             **({"reason": conclusion.reason} if conclusion.reason else {}),
             **dict(evidence or {})},
        )

    def _raise_for(self, receipt) -> None:
        reason = receipt.result.reason
        error_cls = _REASON_ERRORS.get(reason, PrereviewRejected)
        raise error_cls(f"pre-review rejected: {reason}", receipt)

    def _record_replay(self, operation_id: str, at: datetime, receipt,
                       key: tuple[str, int], superseded: bool):
        self._replay_count[key] = self._replay_count.get(key, 0) + 1
        count = self._replay_count[key]
        conclusion = receipt.result
        self.control.run(
            f"{operation_id}:replay{count}", REPLAY_TOPIC, conclusion.train_id, at,
            lambda: {"submission_id": conclusion.submission_id,
                     "revision": conclusion.revision, "decision": conclusion.decision,
                     "replay_of": receipt.operation_id, "replay_count": count},
            {"submission_id": conclusion.submission_id, "revision": conclusion.revision,
             "replayed": True, "replay_count": count},
        )
        return ReplayView(receipt, True, count, superseded)

    def _deny_conflict(self, operation_id: str, at: datetime,
                       submission: PrereviewSubmission, original,
                       key: tuple[str, int], fingerprint: str) -> None:
        self._conflict_count += 1
        n = self._conflict_count
        finding = Finding(R_DUPLICATE, "submission",
                          f"revision {submission.revision} of {submission.submission_id} "
                          f"already committed with a different packet body")
        conclusion = PrereviewConclusion.denied(
            submission, R_DUPLICATE, (finding,),
            supersedes=original.operation_id if hasattr(original, "operation_id") else None,
        )
        receipt = self._commit_conclusion(f"{operation_id}:conflict{n}", at, conclusion,
                                          {"replay_of": getattr(original, "operation_id", None)})
        self._conflicts[(key, fingerprint)] = receipt
        raise DuplicateConflictError(
            f"pre-review rejected: {R_DUPLICATE}", receipt)

    # -- public API --------------------------------------------------------

    def submit(self, operation_id: str, submission: PrereviewSubmission,
               open_windows: list[OpenControlWindow],
               slots: list[BorderSlot],
               isolation_rules: dict[frozenset[str], int],
               formation: FormationLimits,
               at: datetime | None = None):
        """Submit (or replay) one document revision, evaluated at event time.

        On the first sighting of a ``(submission_id, revision)`` it evaluates
        the four rule planes and returns the committed
        :class:`OperationReceipt` for an approval, or raises a
        :class:`PrereviewRejected` subclass carrying the denied receipt for an
        explicit rejection (closed gate / isolation / formation / slot /
        late revision / duplicate conflict).

        Reposting an already-known revision never re-evaluates the planes or
        moves the watermark: an identical packet returns a
        :class:`ReplayView` over the stored receipt (approvals *and* denials),
        while a tampered packet body first raises :class:`DuplicateConflictError`
        and replays that denial on its own subsequent reposts.
        """
        at = _utc(at or submission.event_time)
        event_time = _utc(submission.event_time)
        key = (submission.submission_id, submission.revision)
        fingerprint = self._fingerprint(submission)

        # 1) Exact resubmission — idempotent replay of the stored conclusion,
        #    or of a previously recorded duplicate-conflict denial.
        previous = self._outcomes.get(key)
        if previous is not None:
            if self._fingerprints[key] == fingerprint:
                superseded = (self._head_revision.get(submission.submission_id)
                              != submission.revision)
                return self._record_replay(operation_id, at, previous, key,
                                           superseded)
            conflict = self._conflicts.get((key, fingerprint))
            if conflict is not None:
                return self._record_replay(operation_id, at, conflict, key,
                                           superseded=True)
            self._deny_conflict(operation_id, at, submission, previous,
                                key, fingerprint)

        # 2) Late revision — event time behind the per-submission watermark.
        watermark = self._watermark.get(submission.submission_id)
        if watermark is not None and event_time < watermark:
            finding = Finding(
                R_LATE_REVISION, "event_time",
                f"event_time {event_time.isoformat()} precedes watermark "
                f"{watermark.isoformat()} for {submission.submission_id}",
            )
            conclusion = PrereviewConclusion.denied(submission, R_LATE_REVISION, (finding,))
            receipt = self._commit_conclusion(f"{operation_id}:late", at, conclusion)
            # Late denials are themselves replayable and do not move the watermark.
            self._outcomes[key] = receipt
            self._fingerprints[key] = fingerprint
            raise LateRevisionError(
                f"pre-review rejected: {R_LATE_REVISION}", receipt)

        # 3) A newer revision supersedes the slot booking of an older one.
        supersedes = None
        prior_revision = self._head_revision.get(submission.submission_id)
        if prior_revision is not None and submission.revision < prior_revision:
            # Stale revision number on a new event — treated as a late revision.
            finding = Finding(
                R_LATE_REVISION, "event_time",
                f"revision {submission.revision} superseded by {prior_revision}",
            )
            conclusion = PrereviewConclusion.denied(submission, R_LATE_REVISION, (finding,))
            receipt = self._commit_conclusion(f"{operation_id}:stale", at, conclusion)
            self._outcomes[key] = receipt
            self._fingerprints[key] = fingerprint
            raise LateRevisionError(
                f"pre-review rejected: {R_LATE_REVISION}", receipt)
        if prior_revision is not None and submission.revision > prior_revision:
            supersedes = f"{submission.submission_id}#r{prior_revision}"

        # 4) First sighting — merge the four rule planes as-of event time.
        conclusion = evaluate(
            submission, open_windows, slots, isolation_rules, formation,
            booked_counts=self._booked_counts(slots),
        )
        if supersedes is not None:
            conclusion = replace(conclusion, supersedes=supersedes)

        receipt = self._commit_conclusion(operation_id, at, conclusion)

        self._outcomes[key] = receipt
        self._fingerprints[key] = fingerprint
        self._watermark[submission.submission_id] = event_time
        if prior_revision is not None:
            self._booked_revision.pop(submission.submission_id, None)
        self._head_revision[submission.submission_id] = submission.revision
        if conclusion.approved:
            self._booked_revision[submission.submission_id] = submission.revision

        if not conclusion.approved:
            self._raise_for(receipt)
        return receipt

    def replay(self, operation_id: str, submission_id: str, revision: int,
               at: datetime) -> ReplayView:
        """Replay a stored conclusion without re-evaluating the rule planes."""
        key = (submission_id, revision)
        receipt = self._outcomes.get(key)
        if receipt is None:
            raise KeyError(f"no stored conclusion for {submission_id} r{revision}")
        superseded = self._head_revision.get(submission_id) != revision
        return self._record_replay(operation_id, at, receipt, key, superseded)

    def outcome(self, submission_id: str, revision: int):
        receipt = self._outcomes.get((submission_id, revision))
        return None if receipt is None else receipt.result

    def watermark(self, submission_id: str) -> datetime | None:
        return self._watermark.get(submission_id)

    def journal(self):
        """Ordered snapshot of every committed pre-review control event."""
        topics = {CONCLUSION_TOPIC, REPLAY_TOPIC}
        return [e for e in self.control.stream.snapshot() if e.topic in topics]

    def traceable(self) -> bool:
        """Conclusions are traceable only while the event chain still verifies."""
        return self.control.stream.verify_chain() and self.control.trail.verify()
