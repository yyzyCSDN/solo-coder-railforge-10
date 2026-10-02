import unittest
from datetime import datetime, timezone

from railforge.api import RailForgeService
from railforge.workflows.control import RailForgeWorkflowControl
from railforge.customs.prereview import (
    BorderSlot,
    ClosedGateError,
    CrossBorderPrereviewService,
    DuplicateConflictError,
    FormationLimits,
    LateRevisionError,
    NoBorderSlotError,
    OpenControlWindow,
    PrereviewRejected,
    PrereviewSubmission,
    ReplayView,
    WagonEntry,
    evaluate,
)

U = timezone.utc


def _at(hour, minute=0, day=10):
    return datetime(2026, 3, day, hour, minute, tzinfo=U)


def _windows():
    # Customs open control: 06:00–22:00 daily at the crossing.
    return [OpenControlWindow("BR-T", _at(6, day=10), _at(22, day=10))]


def _slots(capacity=1):
    return [
        BorderSlot("S1", "BR-T", _at(8, day=10), _at(10, day=10), capacity),
        BorderSlot("S2", "BR-T", _at(10, day=10), _at(12, day=10), capacity),
    ]


# Class 1.1 explosives and flammable gas must be separated by >=1 buffer wagon.
ISOLATION = {frozenset({"EX", "FG"}): 1}
FORMATION = FormationLimits(max_wagons=10, max_gross_t=900.0, max_length_m=120.0,
                            hazmat_cert_required=True)


def _submission(rev=1, event_time=None, crossing_time=None, wagons=None,
                sid="DOC-1", train="T-501", certified=True):
    return PrereviewSubmission(
        submission_id=sid,
        revision=rev,
        train_id=train,
        point="BR-T",
        event_time=event_time or _at(9),
        crossing_time=crossing_time or _at(9),
        wagons=tuple(wagons if wagons is not None else [
            WagonEntry("W1", 1, "GEN", 80.0, 18.0),
            WagonEntry("W2", 2, "GEN", 80.0, 18.0),
        ]),
        hazmat_certified=certified,
    )


class PrereviewApprovalTests(unittest.TestCase):
    def test_merges_all_four_planes_into_approved_conclusion(self):
        svc = CrossBorderPrereviewService(RailForgeWorkflowControl())
        receipt = svc.submit("op-1", _submission(), _windows(), _slots(),
                             ISOLATION, FORMATION)
        conclusion = receipt.result
        self.assertTrue(conclusion.approved)
        self.assertEqual(conclusion.slot_id, "S1")
        self.assertEqual(receipt.event.payload["decision"], "approved")
        self.assertIsNone(conclusion.reason)
        self.assertTrue(svc.traceable())

    def test_chosen_slot_is_earliest_feasible_at_crossing_time(self):
        sub = _submission(crossing_time=_at(10, 30))
        conclusion = evaluate(sub, _windows(), _slots(), ISOLATION,
                              FormationLimits(10, 900, 120))
        self.assertTrue(conclusion.approved)
        self.assertEqual(conclusion.slot_id, "S2")


class PrereviewRejectionTests(unittest.TestCase):
    def setUp(self):
        self.control = RailForgeWorkflowControl()
        self.svc = CrossBorderPrereviewService(self.control)

    def test_closed_customs_control_is_explicit_denial(self):
        # Event at 23:00 — outside the 06:00–22:00 open-control window.
        sub = _submission(event_time=_at(23), crossing_time=_at(23))
        with self.assertRaises(ClosedGateError) as ctx:
            self.svc.submit("op-closed", sub, _windows(), _slots(),
                            ISOLATION, FORMATION)
        receipt = ctx.exception.receipt
        self.assertEqual(receipt.result.reason, "closed_gate")
        self.assertEqual(receipt.result.decision, "rejected")
        self.assertEqual(receipt.result.findings[0].plane, "open_control")
        self.assertEqual(receipt.event.payload["decision"], "rejected")
        self.assertEqual(receipt.event.payload["reason"], "closed_gate")
        # Denial itself is a committed, traceable event.
        self.assertTrue(self.svc.traceable())

    def test_dangerous_goods_isolation_conflict_is_denied(self):
        wagons = [
            WagonEntry("W1", 1, "EX", 80.0, 18.0),
            WagonEntry("W2", 2, "FG", 80.0, 18.0),  # adjacent, buffer required
        ]
        with self.assertRaises(PrereviewRejected) as ctx:
            self.svc.submit("op-iso", _submission(wagons=wagons), _windows(),
                            _slots(), ISOLATION, FORMATION)
        self.assertEqual(ctx.exception.reason, "isolation_conflict")
        self.assertEqual(ctx.exception.receipt.result.findings[0].plane,
                         "hazmat_isolation")
        # A denial must not consume border-slot capacity.
        self.assertEqual(self.svc._booked_counts(_slots()), {})

    def test_formation_envelope_breach_is_denied(self):
        wagons = [WagonEntry(f"W{i}", i, "GEN", 100.0, 20.0) for i in range(1, 12)]
        with self.assertRaises(PrereviewRejected) as ctx:
            self.svc.submit("op-form", _submission(wagons=wagons), _windows(),
                            _slots(), ISOLATION, FORMATION)
        self.assertEqual(ctx.exception.reason, "formation_exceeded")
        self.assertIn("wagons 11 exceed limit 10",
                      ctx.exception.receipt.result.findings[0].detail)

    def test_no_border_slot_capacity_is_denied(self):
        # First submission fills S1 at 09:00 (capacity 1).
        self.svc.submit("op-a", _submission(sid="DOC-A", train="T-A"),
                        _windows(), _slots(capacity=1), ISOLATION, FORMATION)
        with self.assertRaises(NoBorderSlotError) as ctx:
            self.svc.submit("op-b", _submission(sid="DOC-B", train="T-B"),
                            _windows(), _slots(capacity=1), ISOLATION, FORMATION)
        self.assertEqual(ctx.exception.reason, "no_border_slot")
        self.assertEqual(ctx.exception.receipt.result.findings[0].plane,
                         "border_slot")

    def test_extra_capacity_admits_second_train(self):
        slots = _slots(capacity=2)
        self.svc.submit("op-a", _submission(sid="DOC-A", train="T-A"),
                        _windows(), slots, ISOLATION, FORMATION)
        receipt = self.svc.submit("op-b", _submission(sid="DOC-B", train="T-B"),
                                  _windows(), slots, ISOLATION, FORMATION)
        self.assertTrue(receipt.result.approved)
        self.assertEqual(receipt.result.slot_id, "S1")


class LateRevisionTests(unittest.TestCase):
    def setUp(self):
        self.svc = CrossBorderPrereviewService(RailForgeWorkflowControl())

    def test_revision_with_older_event_time_is_rejected(self):
        self.svc.submit("op-r1", _submission(rev=1, event_time=_at(12)),
                        _windows(), _slots(), ISOLATION, FORMATION)
        # r2 arrives with an event time behind r1's watermark.
        late = _submission(rev=2, event_time=_at(11))
        with self.assertRaises(LateRevisionError) as ctx:
            self.svc.submit("op-r2-late", late, _windows(), _slots(),
                            ISOLATION, FORMATION)
        receipt = ctx.exception.receipt
        self.assertEqual(receipt.result.reason, "late_revision")
        self.assertEqual(receipt.result.findings[0].plane, "event_time")
        # Watermark unchanged.
        self.assertEqual(self.svc.watermark("DOC-1"), _at(12))

    def test_late_revision_denial_is_replayable(self):
        self.svc.submit("op-r1", _submission(rev=1, event_time=_at(12)),
                        _windows(), _slots(), ISOLATION, FORMATION)
        late = _submission(rev=2, event_time=_at(11))
        with self.assertRaises(LateRevisionError):
            self.svc.submit("op-r2-late", late, _windows(), _slots(),
                            ISOLATION, FORMATION)
        # Reposting the same late revision replays the stored denial: no new
        # conclusion event, no re-evaluation, watermark untouched.
        before = len([e for e in self.svc.journal()
                      if e.topic == "customs.prereview.conclusion"])
        view = self.svc.submit("op-r2-replay", late, _windows(), _slots(),
                               ISOLATION, FORMATION)
        self.assertIsInstance(view, ReplayView)
        self.assertEqual(view.receipt.result.reason, "late_revision")
        self.assertEqual(view.replay_count, 1)
        conclusions = [e for e in self.svc.journal()
                       if e.topic == "customs.prereview.conclusion"]
        self.assertEqual(len(conclusions), before)  # r1 approval + late denial only
        self.assertEqual(self.svc.watermark("DOC-1"), _at(12))

    def test_forward_revision_supersedes_and_rebooks_slot(self):
        r1 = self.svc.submit("op-r1", _submission(rev=1, crossing_time=_at(9)),
                             _windows(), _slots(), ISOLATION, FORMATION)
        self.assertEqual(r1.result.slot_id, "S1")
        r2 = self.svc.submit("op-r2", _submission(rev=2, crossing_time=_at(10, 30)),
                             _windows(), _slots(), ISOLATION, FORMATION)
        self.assertTrue(r2.result.approved)
        self.assertEqual(r2.result.slot_id, "S2")
        self.assertEqual(r2.result.supersedes, "DOC-1#r1")
        # Old booking released: S1 free again for another train.
        other = self.svc.submit(
            "op-other", _submission(sid="DOC-9", crossing_time=_at(9)),
            _windows(), _slots(), ISOLATION, FORMATION)
        self.assertEqual(other.result.slot_id, "S1")


class DuplicateSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.svc = CrossBorderPrereviewService(RailForgeWorkflowControl())

    def test_identical_resubmission_replays_without_new_conclusion(self):
        first = self.svc.submit("op-1", _submission(), _windows(), _slots(),
                                ISOLATION, FORMATION)
        replay = self.svc.submit("op-1-again", _submission(), _windows(),
                                 _slots(), ISOLATION, FORMATION)
        self.assertIsInstance(replay, ReplayView)
        self.assertTrue(replay.replayed)
        self.assertIs(replay.receipt, first)
        self.assertEqual(replay.replay_count, 1)

        # A second replay bumps the counter and keeps returning the same receipt.
        replay2 = self.svc.submit("op-1-again-2", _submission(), _windows(),
                                  _slots(), ISOLATION, FORMATION)
        self.assertIs(replay2.receipt, first)
        self.assertEqual(replay2.replay_count, 2)

        # Exactly one conclusion event — replays must not duplicate it.
        conclusions = [e for e in self.svc.journal()
                       if e.topic == "customs.prereview.conclusion"]
        self.assertEqual(len(conclusions), 1)
        self.assertTrue(self.svc.traceable())

    def test_same_revision_with_different_packet_is_duplicate_conflict(self):
        self.svc.submit("op-1", _submission(certified=True), _windows(),
                        _slots(), ISOLATION, FORMATION)
        tampered = _submission(certified=False)  # same submission_id + revision
        with self.assertRaises(DuplicateConflictError) as ctx:
            self.svc.submit("op-1-tampered", tampered, _windows(), _slots(),
                            ISOLATION, FORMATION)
        self.assertEqual(ctx.exception.receipt.result.reason,
                         "duplicate_conflict")
        # Replaying the identical tampered packet returns that same denial
        # (first repost caches it) without emitting another conclusion event.
        conclusions_before = len([
            e for e in self.svc.journal()
            if e.topic == "customs.prereview.conclusion"])
        view = self.svc.submit("op-1-tampered-2", tampered, _windows(),
                               _slots(), ISOLATION, FORMATION)
        self.assertIsInstance(view, ReplayView)
        self.assertIs(view.receipt, ctx.exception.receipt)
        self.assertEqual(len([e for e in self.svc.journal()
                              if e.topic == "customs.prereview.conclusion"]),
                         conclusions_before)

    def test_rejected_packet_resubmission_replays_denial(self):
        sub = _submission(event_time=_at(23), crossing_time=_at(23))
        with self.assertRaises(ClosedGateError):
            self.svc.submit("op-closed", sub, _windows(), _slots(),
                            ISOLATION, FORMATION)
        view = self.svc.replay("op-replay", "DOC-1", 1, _at(23, 5))
        self.assertTrue(view.replayed)
        self.assertEqual(view.receipt.result.reason, "closed_gate")
        self.assertEqual(view.replay_count, 1)

    def test_replay_unknown_conclusion_raises(self):
        with self.assertRaises(KeyError):
            self.svc.replay("op-x", "UNKNOWN", 9, _at(9))


class TraceabilityTests(unittest.TestCase):
    def test_event_time_is_recorded_and_stream_chain_verifies(self):
        control = RailForgeWorkflowControl()
        svc = CrossBorderPrereviewService(control)
        receipt = svc.submit("op-t", _submission(event_time=_at(9, 30)),
                             _windows(), _slots(), ISOLATION, FORMATION)
        event = receipt.event
        self.assertEqual(event.occurred_at, _at(9, 30))
        self.assertEqual(event.topic, "customs.prereview.conclusion")
        self.assertEqual(event.payload["submission_id"], "DOC-1")
        self.assertEqual(event.payload["revision"], 1)
        self.assertTrue(control.stream.verify_chain())
        self.assertTrue(control.trail.verify())

    def test_service_entrypoint_uses_shared_control_plane(self):
        service = RailForgeService()
        receipt = service.prereview_freight(
            "op-api", _submission(), _windows(), _slots(), ISOLATION, FORMATION)
        self.assertTrue(receipt.result.approved)
        self.assertTrue(service.workflow().stream.verify_chain())
        # Feature surface and control accessor share one stateful service.
        self.assertIs(service.workflow().prereview(), service.workflow().prereview())


if __name__ == "__main__":
    unittest.main()
