import unittest
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.customs.document_pre_review import (
    ACCEPTED, CONSIST_INVALID, DUPLICATE_REVISION, GATE_CLOSED, REJECTED, REVISE,
    REVISION_LATE, REVISION_OUT_OF_ORDER, SEGREGATION_VIOLATION, SLOT_MISSING,
    SUBMIT, BorderSlot, DocumentDraft, DocumentPreReviewEngine, GateWindow,
    PreReviewEvent, WagonEntry, replay,
)

U = timezone.utc
DAY = datetime(2026, 3, 1, tzinfo=U)


def at(hour, minute=0):
    return DAY + timedelta(hours=hour, minutes=minute)


def make_engine():
    windows = [GateWindow('ERE', at(8), at(20))]
    slots = [BorderSlot('TRAIN-1', 'ERE', at(18), at(18, 30))]
    rules = {frozenset({'3', '5'}): 2}
    return DocumentPreReviewEngine(windows, slots, rules, timedelta(hours=2))


def draft(revision=1, wagons=None, train='TRAIN-1', document_id='DOC-1', scope='ERE'):
    if wagons is None:
        wagons = (WagonEntry(1, 'W1'), WagonEntry(2, 'W2', '3'), WagonEntry(3, 'W3'),
                  WagonEntry(4, 'W4'), WagonEntry(5, 'W5', '5'))
    return DocumentDraft(document_id, train, scope, revision, wagons)


def event(sequence, kind, hour, minute=0, **kwargs):
    return PreReviewEvent(sequence, kind, at(hour, minute), draft(**kwargs))


class DocumentPreReviewTests(unittest.TestCase):

    def test_accepted_submission_is_traceable(self):
        engine = make_engine()
        conclusion = engine.handle(event(1, SUBMIT, 9))
        self.assertEqual(conclusion.status, ACCEPTED)
        self.assertEqual(conclusion.reasons, ())
        self.assertEqual(conclusion.decided_at, at(9))
        self.assertFalse(conclusion.replayed)
        self.assertEqual([c.check for c in conclusion.checks],
                         ['gate_open', 'border_slot', 'consist', 'segregation'])
        self.assertTrue(all(c.passed for c in conclusion.checks))
        self.assertIn('08:00', conclusion.checks[0].detail)
        self.assertIn('18:00', conclusion.checks[1].detail)
        self.assertEqual(engine.accepted('DOC-1'), draft())
        self.assertIs(engine.conclusion('DOC-1', 1), conclusion)

    def test_gate_closed_rejection(self):
        engine = make_engine()
        conclusion = engine.handle(event(1, SUBMIT, 6))
        self.assertEqual(conclusion.status, REJECTED)
        self.assertEqual(conclusion.reasons, (GATE_CLOSED,))
        self.assertEqual(conclusion.checks[0].detail, 'no open gate window')
        self.assertIsNone(engine.accepted('DOC-1'))

    def test_border_slot_missing_rejection(self):
        engine = make_engine()
        conclusion = engine.handle(event(1, SUBMIT, 9, train='TRAIN-9'))
        self.assertEqual(conclusion.status, REJECTED)
        self.assertEqual(conclusion.reasons, (SLOT_MISSING,))

    def test_consist_invalid_rejection(self):
        cases = [
            (WagonEntry(1, 'W1'), WagonEntry(2, 'W2'), WagonEntry(4, 'W4')),
            (WagonEntry(1, 'W1'), WagonEntry(2, 'W1')),
            (),
        ]
        for seq, wagons in enumerate(cases, start=1):
            with self.subTest(wagons=wagons):
                engine = make_engine()
                conclusion = engine.handle(event(seq, SUBMIT, 9, wagons=wagons))
                self.assertEqual(conclusion.status, REJECTED)
                self.assertIn(CONSIST_INVALID, conclusion.reasons)

    def test_segregation_violation_rejection(self):
        wagons = (WagonEntry(1, 'W1'), WagonEntry(2, 'W2', '3'), WagonEntry(3, 'W3', '5'))
        engine = make_engine()
        conclusion = engine.handle(event(1, SUBMIT, 9, wagons=wagons))
        self.assertEqual(conclusion.status, REJECTED)
        self.assertEqual(conclusion.reasons, (SEGREGATION_VIOLATION,))
        detail = conclusion.checks[-1].detail
        self.assertIn('W2', detail)
        self.assertIn('W3', detail)

    def test_late_revision_rejected_and_original_survives(self):
        engine = make_engine()
        first = engine.handle(event(1, SUBMIT, 9))
        self.assertEqual(first.status, ACCEPTED)
        late = engine.handle(event(2, REVISE, 16, 30, revision=2))
        self.assertEqual(late.status, REJECTED)
        self.assertEqual(late.reasons, (REVISION_LATE,))
        self.assertEqual(engine.accepted('DOC-1').revision, 1)
        again = engine.handle(event(3, REVISE, 16, 45, revision=2))
        self.assertEqual(again.status, REJECTED)
        self.assertEqual(again.reasons, (REVISION_LATE,))
        self.assertTrue(again.replayed)
        self.assertEqual(again.decided_at, late.decided_at)

    def test_revision_before_deadline_accepted(self):
        engine = make_engine()
        engine.handle(event(1, SUBMIT, 9))
        revised = engine.handle(event(2, REVISE, 15, 59, revision=2))
        self.assertEqual(revised.status, ACCEPTED)
        self.assertEqual(engine.accepted('DOC-1').revision, 2)

    def test_duplicate_submission_replays_and_conflicts(self):
        engine = make_engine()
        first = engine.handle(event(1, SUBMIT, 9))
        same = engine.handle(event(2, SUBMIT, 10))
        self.assertTrue(same.replayed)
        self.assertEqual(same.status, first.status)
        self.assertEqual(same.decided_at, first.decided_at)
        self.assertEqual(len(engine.log), 2)
        changed = engine.handle(event(3, SUBMIT, 11, wagons=(WagonEntry(1, 'W9'),)))
        self.assertEqual(changed.status, REJECTED)
        self.assertEqual(changed.reasons, (DUPLICATE_REVISION,))
        self.assertEqual(engine.accepted('DOC-1').revision, 1)
        self.assertEqual(len(engine.log), 3)

    def test_out_of_order_rejections(self):
        engine = make_engine()
        orphan = engine.handle(event(1, REVISE, 9, revision=2))
        self.assertEqual(orphan.reasons, (REVISION_OUT_OF_ORDER,))
        engine.handle(event(2, SUBMIT, 10))
        jumped = engine.handle(event(3, REVISE, 11, revision=3))
        self.assertEqual(jumped.status, ACCEPTED)
        stale = engine.handle(event(4, REVISE, 12, revision=2))
        self.assertEqual(stale.reasons, (REVISION_OUT_OF_ORDER,))
        resubmit = engine.handle(event(5, SUBMIT, 13, revision=4))
        self.assertEqual(resubmit.reasons, (REVISION_OUT_OF_ORDER,))
        self.assertEqual(engine.accepted('DOC-1').revision, 3)

    def test_stream_integrity_rejections(self):
        engine = make_engine()
        engine.handle(event(1, SUBMIT, 9))
        with self.assertRaises(ValueError):
            engine.handle(event(1, SUBMIT, 10))
        with self.assertRaises(ValueError):
            engine.handle(event(2, SUBMIT, 8, document_id='DOC-2'))
        with self.assertRaises(ValueError):
            engine.handle(PreReviewEvent(3, 'void', at(11), draft(document_id='DOC-3')))

    def test_replay_matches_online(self):
        engine = make_engine()
        events = [
            event(1, SUBMIT, 9),
            event(2, REVISE, 10, revision=2, wagons=(WagonEntry(1, 'W1'), WagonEntry(2, 'W2', '3'),
                                                     WagonEntry(3, 'W3'), WagonEntry(4, 'W4B'),
                                                     WagonEntry(5, 'W5', '5'))),
            event(3, REVISE, 10, 5, revision=2, wagons=(WagonEntry(1, 'W1'), WagonEntry(2, 'W2', '3'),
                                                        WagonEntry(3, 'W3'), WagonEntry(4, 'W4B'),
                                                        WagonEntry(5, 'W5', '5'))),
            event(4, REVISE, 10, 10, revision=2),
            event(5, REVISE, 16, 30, revision=3),
            event(6, REVISE, 16, 45, revision=3),
        ]
        online = [engine.handle(e) for e in events]
        self.assertEqual([c.status for c in online],
                         [ACCEPTED, ACCEPTED, ACCEPTED, REJECTED, REJECTED, REJECTED])
        self.assertEqual([c.replayed for c in online],
                         [False, False, True, False, False, True])
        rebuilt = replay(engine.log, gate_windows=[GateWindow('ERE', at(8), at(20))],
                         border_slots=[BorderSlot('TRAIN-1', 'ERE', at(18), at(18, 30))],
                         segregation_rules={frozenset({'3', '5'}): 2},
                         revision_cutoff=timedelta(hours=2))
        self.assertEqual(rebuilt, online)

    def test_feature_commits_atomically(self):
        service = RailForgeService()
        engine = make_engine()
        events = [event(1, SUBMIT, 9), event(2, SUBMIT, 10)]
        receipt = service.features.pre_review_documents('op-pre-1', engine, events, at(12))
        self.assertEqual(receipt.event.payload['status'], 'committed')
        self.assertEqual(receipt.event.payload['control_path'], 'atomic')
        self.assertEqual(receipt.result['accepted'], 2)
        self.assertEqual(receipt.result['rejected'], 0)
        self.assertEqual(receipt.result['replayed'], 1)
        self.assertEqual(receipt.result['conclusions'][0].status, ACCEPTED)
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())
        again = service.features.pre_review_documents('op-pre-1', engine, events, at(12))
        self.assertIs(again, receipt)
        self.assertEqual(len(engine.log), 2)


if __name__ == '__main__':
    unittest.main()
