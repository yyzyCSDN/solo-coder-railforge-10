from __future__ import annotations

from railforge.cargo.segregation import violations
from railforge.commercial.contract_capacity import allocate as allocate_capacity
from railforge.control.recovery import choose as choose_recovery
from railforge.customs.holds import active
from railforge.crew.duty import legal as duty_legal
from railforge.crew.relief_points import choose as choose_relief
from railforge.eta.event_fusion import fused_delay
from railforge.inventory.wagon_pool import allocate
from railforge.intermodal.connections import choose as choose_connection
from railforge.maintenance.release import releasable
from railforge.possession.windows import conflicts as possession_conflicts
from railforge.planning.loco_circulation import circulate
from railforge.planning.train_path_engine import build, section_load, train_delay
from railforge.routing.clearance import route_ok
from railforge.signaling.block_occupancy import conflicts as block_conflicts
from railforge.yard.hump import classify
from railforge.yard.switching import lock_route


class RailForgeFeatureService:
    """Fifteen product workflows, all committed through the common control plane."""

    def __init__(self, control):
        self.control = control

    def coordinate_route(self, operation_id, requests, sections, existing, headway, at):
        def action():
            plan = build(requests, sections, existing, headway)
            return {"plan": plan, "load": section_load(plan), "delay": train_delay(plan, requests)}
        return self.control.run(operation_id, "feature.route.coordinated", operation_id, at, action)

    def plan_loco_cycle(self, operation_id, legs, units, turn_minutes, at):
        return self.control.run(operation_id, "feature.loco.cycle", operation_id, at,
                                lambda: circulate(legs, units, turn_minutes))

    def arrange_crew_handoff(self, operation_id, duty_rows, max_duty, max_drive, max_span,
                             relief_options, required_capacity, required_caps, max_risk, max_duration, at):
        def action():
            if not duty_legal(duty_rows, max_duty, max_drive, max_span):
                raise ValueError("duty is not legal")
            relief = choose_relief(relief_options, required_capacity, required_caps, max_risk, max_duration)
            if relief is None:
                raise ValueError("no relief option")
            return {"duty": True, "relief": relief}
        return self.control.run(operation_id, "feature.crew.handoff", operation_id, at, action)

    def check_route_clearance(self, operation_id, profile, segments, wagons, axle_limits, at):
        def action():
            if not route_ok(profile, segments):
                raise ValueError("route clearance denied")
            rejected = [w.id for w in wagons if not axle_limits(w)]
            if rejected:
                raise ValueError("axle limit: " + ",".join(rejected))
            return {"clear": True, "wagons": len(wagons)}
        return self.control.run(operation_id, "feature.route.clearance", operation_id, at, action)

    def rebalance_yard(self, operation_id, cuts, tracks, occupied, route, switches, owner, at):
        def action():
            assignment = classify(cuts, tracks, occupied)
            locked = lock_route(route, switches, owner)
            return {"assignment": assignment, "switches": locked}
        return self.control.run(operation_id, "feature.yard.rebalance", operation_id, at, action)

    def book_intermodal_chain(self, operation_id, inbound, candidates, unload, load, cutoff,
                              wagons, wagon_type, location, on, count, required_days, forbidden, at):
        def action():
            move = choose_connection(inbound, candidates, unload, load, cutoff)
            if move is None:
                raise ValueError("no intermodal connection")
            selected = allocate(wagons, wagon_type, location, on, count, required_days, forbidden)
            return {"move": move, "wagons": selected}
        return self.control.run(operation_id, "feature.intermodal.chain", operation_id, at, action)

    def approve_possession(self, operation_id, candidate, rows, occupancies, occupancy, clearance, at):
        def action():
            if possession_conflicts(candidate, rows):
                raise ValueError("possession conflict")
            if block_conflicts(occupancy, occupancies, clearance):
                raise ValueError("block conflict")
            return {"approved": True}
        return self.control.run(operation_id, "feature.possession.approved", operation_id, at, action)

    def release_vehicle(self, operation_id, evidence, policy, check_at, brake_rows, gradient, at):
        def action():
            if not releasable(evidence, policy, check_at):
                raise ValueError("maintenance evidence incomplete")
            percentage = self.control.assess_brakes(operation_id + ":brakes", brake_rows, gradient, operation_id, at).result
            return {"released": True, "brake_percentage": percentage}
        return self.control.run(operation_id, "feature.vehicle.release", operation_id, at, action)

    def reserve_capacity(self, operation_id, requests, resources, at):
        return self.control.run(operation_id, "feature.capacity.reserved", operation_id, at,
                                lambda: allocate_capacity(requests, resources))

    def preclear_customs(self, operation_id, holds, scope, check_at, loads, rules, at):
        def action():
            if active(holds, scope, check_at):
                raise ValueError("customs hold active")
            bad = violations(loads, rules)
            if bad:
                raise ValueError("cargo segregation conflict")
            return {"precleared": True}
        return self.control.run(operation_id, "feature.customs.precleared", operation_id, at, action)

    def recover_disruption(self, operation_id, options, capacity, capabilities, max_risk, max_duration,
                           requests, sections, existing, headway, at):
        def action():
            option = choose_recovery(options, capacity, capabilities, max_risk, max_duration)
            if option is None:
                raise ValueError("no recovery option")
            plan = build(requests, sections, existing, headway)
            return {"option": option, "plan": plan, "delay": train_delay(plan, requests)}
        return self.control.run(operation_id, "feature.disruption.recovered", operation_id, at, action)

    def validate_hazmat_consist(self, operation_id, consist_validator, loads, rules, at):
        def action():
            ends = consist_validator()
            bad = violations(loads, rules)
            if bad:
                raise ValueError("hazmat separation conflict")
            return {"ends": ends, "compliant": True}
        return self.control.run(operation_id, "feature.hazmat.validated", operation_id, at, action)

    def reposition_wagons(self, operation_id, wagons, wagon_type, location, on, count, required_days,
                          forbidden, legs, units, turn_minutes, at):
        def action():
            selected = allocate(wagons, wagon_type, location, on, count, required_days, forbidden)
            cycle = circulate(legs, units, turn_minutes)
            return {"wagons": selected, "cycle": cycle}
        return self.control.run(operation_id, "feature.wagons.repositioned", operation_id, at, action)

    def calculate_compensation(self, operation_id, arrival, release, tariff, claim_case, claim_steps, at):
        from railforge.billing.demurrage import charge
        from railforge.customer.claims import ready
        def action():
            return {"demurrage": charge(arrival, release, tariff), "claim_ready": ready(claim_case, claim_steps)}
        return self.control.run(operation_id, "feature.compensation.calculated", operation_id, at, action)

    def timeline(self, operation_id, events, observations, max_lateness, at):
        def action():
            from railforge.events.cursor import CursorConsumer
            consumer = self.control._consumers.setdefault("timeline", CursorConsumer())
            accepted = [consumer.accept(e) for e in events]
            return {"accepted": accepted, "delay": fused_delay(observations, max_lateness)}
        return self.control.run(operation_id, "feature.timeline.reconciled", operation_id, at, action)

    def prereview_freight(self, operation_id, submission, open_windows, slots,
                          isolation_rules, formation, at=None):
        """Merge customs open-control, hazmat isolation, border slots and train
        formation into a traceable pre-review conclusion keyed on event time."""
        return self.control.prereview().submit(
            operation_id, submission, open_windows, slots, isolation_rules,
            formation, at=at,
        )

