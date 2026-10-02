"""Cross-border freight document pre-review.

Merges customs gate opening control, dangerous-goods segregation, border
slots and train consist into one traceable conclusion per document event.
Conclusions are decided at event time and are a pure function of the event
log, so replaying the log reproduces them exactly.

Rejection and replay behaviour:

- gate control: an event outside any open gate window is rejected
  (``gate_closed``); a train without an upcoming border slot is rejected
  (``slot_missing``).
- late revision: a revision whose event time is after
  ``slot.start - revision_cutoff`` is rejected (``revision_late``); the
  previously accepted revision stays in force.
- duplicate submission: a repeated ``(document_id, revision)`` with
  identical content replays the first conclusion (``replayed=True``) and
  never advances state; identical revision with different content is
  rejected (``duplicate_revision``).
- stream integrity: duplicate sequences, regressing event times and
  unknown event kinds raise ``ValueError``.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from railforge.cargo.segregation import Load, violations as segregation_violations

SUBMIT = 'submit'
REVISE = 'revise'

ACCEPTED = 'accepted'
REJECTED = 'rejected'

GATE_CLOSED = 'gate_closed'
SLOT_MISSING = 'slot_missing'
REVISION_LATE = 'revision_late'
CONSIST_INVALID = 'consist_invalid'
SEGREGATION_VIOLATION = 'segregation_violation'
DUPLICATE_REVISION = 'duplicate_revision'
REVISION_OUT_OF_ORDER = 'revision_out_of_order'


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class GateWindow:
    scope: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class BorderSlot:
    train: str
    scope: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class WagonEntry:
    position: int
    wagon: str
    hazmat_class: str | None = None


@dataclass(frozen=True)
class DocumentDraft:
    document_id: str
    train: str
    scope: str
    revision: int
    wagons: tuple[WagonEntry, ...]

    def __post_init__(self):
        object.__setattr__(self, 'wagons', tuple(self.wagons))


@dataclass(frozen=True)
class PreReviewEvent:
    sequence: int
    kind: str
    at: datetime
    draft: DocumentDraft


@dataclass(frozen=True)
class CheckOutcome:
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreReviewConclusion:
    document_id: str
    train: str
    revision: int
    status: str
    checks: tuple[CheckOutcome, ...]
    reasons: tuple[str, ...]
    decided_at: datetime
    replayed: bool = False


def _gate_window(windows, scope, at):
    x = _utc(at)
    for w in windows:
        if w.scope not in (scope, '*'):
            continue
        if _utc(w.start) <= x < _utc(w.end):
            return w
    return None


def _border_slot(slots, train, scope, at):
    x = _utc(at)
    options = [s for s in slots if s.train == train and s.scope in (scope, '*') and x < _utc(s.start)]
    if not options:
        return None
    return min(options, key=lambda s: (_utc(s.start), s.train))


def _consist_error(wagons):
    if not wagons:
        return 'empty consist'
    positions = sorted(w.position for w in wagons)
    if positions != list(range(1, len(positions) + 1)):
        return 'position gap'
    ids = [w.wagon for w in wagons]
    if len(ids) != len(set(ids)):
        return 'duplicate wagon'
    return None


_REASON_BY_CHECK = {
    'gate_open': GATE_CLOSED,
    'border_slot': SLOT_MISSING,
    'revision_window': REVISION_LATE,
    'consist': CONSIST_INVALID,
    'segregation': SEGREGATION_VIOLATION,
}


class DocumentPreReviewEngine:
    """Stateful pre-review over an ordered event log; see module docstring."""

    def __init__(self, gate_windows=(), border_slots=(), segregation_rules=None,
                 revision_cutoff=timedelta(0)):
        if revision_cutoff < timedelta(0):
            raise ValueError('revision cutoff')
        for w in gate_windows:
            if _utc(w.end) <= _utc(w.start):
                raise ValueError('gate window')
        for s in border_slots:
            if _utc(s.end) <= _utc(s.start):
                raise ValueError('border slot')
        self._windows = tuple(gate_windows)
        self._slots = tuple(border_slots)
        self._rules = dict(segregation_rules or {})
        self._cutoff = revision_cutoff
        self._log: list[PreReviewEvent] = []
        self._sequences: set[int] = set()
        self._seen: dict[tuple[str, int], DocumentDraft] = {}
        self._conclusions: dict[tuple[str, int], PreReviewConclusion] = {}
        self._drafts: dict[str, DocumentDraft] = {}

    @property
    def log(self):
        return tuple(self._log)

    def conclusion(self, document_id, revision):
        return self._conclusions.get((document_id, revision))

    def accepted(self, document_id):
        return self._drafts.get(document_id)

    def handle(self, event: PreReviewEvent) -> PreReviewConclusion:
        if event.kind not in (SUBMIT, REVISE):
            raise ValueError('event kind')
        if event.sequence in self._sequences:
            raise ValueError('event sequence')
        if self._log and _utc(event.at) < _utc(self._log[-1].at):
            raise ValueError('event time regresses')
        self._sequences.add(event.sequence)
        self._log.append(event)
        return self._decide(event)

    def _decide(self, event):
        draft = event.draft
        key = (draft.document_id, draft.revision)
        seen = self._seen.get(key)
        if seen is not None:
            if seen == draft:
                return replace(self._conclusions[key], replayed=True)
            return self._reject(event, 'duplicate',
                                'revision content differs from first submission',
                                DUPLICATE_REVISION)
        current = self._drafts.get(draft.document_id)
        if event.kind == SUBMIT:
            if current is not None:
                return self._reject(event, 'version',
                                    f'document already accepted at revision {current.revision}',
                                    REVISION_OUT_OF_ORDER)
        elif current is None:
            return self._reject(event, 'version', 'no accepted submission to revise',
                                REVISION_OUT_OF_ORDER)
        elif draft.revision <= current.revision:
            return self._reject(event, 'version',
                                f'revision {draft.revision} not ahead of accepted revision {current.revision}',
                                REVISION_OUT_OF_ORDER)
        checks = self._checks(event)
        reasons = tuple(_REASON_BY_CHECK[c.check] for c in checks if not c.passed)
        status = ACCEPTED if not reasons else REJECTED
        conclusion = PreReviewConclusion(draft.document_id, draft.train, draft.revision,
                                         status, tuple(checks), reasons, _utc(event.at))
        self._seen[key] = draft
        self._conclusions[key] = conclusion
        if status == ACCEPTED:
            self._drafts[draft.document_id] = draft
        return conclusion

    def _checks(self, event):
        draft, at = event.draft, _utc(event.at)
        checks = []
        window = _gate_window(self._windows, draft.scope, at)
        checks.append(CheckOutcome(
            'gate_open', window is not None,
            f'open {_utc(window.start).isoformat()}..{_utc(window.end).isoformat()}'
            if window else 'no open gate window'))
        slot = _border_slot(self._slots, draft.train, draft.scope, at)
        checks.append(CheckOutcome(
            'border_slot', slot is not None,
            f'slot {_utc(slot.start).isoformat()}..{_utc(slot.end).isoformat()}'
            if slot else 'no border slot'))
        if event.kind == REVISE:
            if slot is None:
                checks.append(CheckOutcome('revision_window', False, 'no border slot for deadline'))
            else:
                deadline = _utc(slot.start) - self._cutoff
                late = at > deadline
                checks.append(CheckOutcome(
                    'revision_window', not late,
                    f'deadline {deadline.isoformat()}' if not late
                    else f'late by {(at - deadline).total_seconds() / 60:g} minutes'))
        error = _consist_error(draft.wagons)
        checks.append(CheckOutcome('consist', error is None,
                                   error or f'{len(draft.wagons)} wagons in order'))
        loads = [Load(w.wagon, w.hazmat_class, w.position) for w in draft.wagons if w.hazmat_class]
        bad = segregation_violations(loads, self._rules)
        checks.append(CheckOutcome(
            'segregation', not bad,
            '; '.join(f'{a}~{b} need {need} got {got}' for a, b, need, got in bad)
            or 'no violations'))
        return checks

    @staticmethod
    def _reject(event, check, detail, reason):
        draft = event.draft
        return PreReviewConclusion(draft.document_id, draft.train, draft.revision, REJECTED,
                                   (CheckOutcome(check, False, detail),), (reason,),
                                   _utc(event.at))


def replay(events, gate_windows=(), border_slots=(), segregation_rules=None,
           revision_cutoff=timedelta(0)):
    """Rebuild conclusions from scratch, ordered by event time."""
    engine = DocumentPreReviewEngine(gate_windows, border_slots, segregation_rules,
                                     revision_cutoff)
    ordered = sorted(events, key=lambda e: (_utc(e.at), e.sequence))
    return [engine.handle(e) for e in ordered]
