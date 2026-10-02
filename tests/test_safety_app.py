import sys
import types

import numpy as np
import pytest

# safety_app only needs plot_one_box from utils.plots, which pulls in torch,
# matplotlib and seaborn. Stub it for the import so these run in the unit job.
_stub = types.ModuleType("utils.plots")
_stub.plot_one_box = lambda *args, **kwargs: None
_real = sys.modules.get("utils.plots")
sys.modules["utils.plots"] = _stub
try:
    import safety_app
finally:
    if _real is not None:
        sys.modules["utils.plots"] = _real
    else:
        del sys.modules["utils.plots"]

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Polygon covering the left half of a 200x200 frame.
ZONE = [[0, 0], [100, 0], [100, 200], [0, 200]]


def det(label, x0, y0, x1, y1):
    """One entry of the box_list securade.py builds: [label, x0, y0, x1, y1, conf]."""
    return [label, x0, y0, x1, y1, "0.90"]


# Two people side by side, each with room for PPE boxes inside.
PERSON_A = det("Person", 10, 10, 60, 190)
PERSON_B = det("Person", 120, 10, 170, 190)


def on_a(label):
    return det(label, 20, 20, 50, 50)


def on_b(label):
    return det(label, 130, 20, 160, 50)


@pytest.fixture
def image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def drawn(monkeypatch):
    """Record (label, color) for every box safety_app draws."""
    calls = []
    monkeypatch.setattr(
        safety_app, "draw_box", lambda img, label, x0, y0, x1, y1, color: calls.append((label, color))
    )
    return calls


class TestColorRules:
    @pytest.mark.parametrize(
        "flags, items, expected",
        [
            ((True, True, True), ["a", "b", "c"], True),
            ((True, True, True), ["a", "b"], False),
            ((True, True, False), ["a", "b"], True),
            ((True, False, True), ["a"], False),
            ((False, True, True), ["b", "c"], True),
            ((True, False, False), ["a"], True),
            ((False, True, False), ["a"], False),
            ((False, False, True), ["c"], True),
            ((False, False, False), ["a", "b", "c"], False),
        ],
    )
    def test_should_color_needs_every_enabled_item(self, flags, items, expected):
        assert safety_app.should_color(*flags, "a", "b", "c", items) is expected

    @pytest.mark.parametrize(
        "flags, items, expected",
        [
            ((True, True, True), ["c"], True),
            ((True, True, True), [], False),
            ((True, True, False), ["c"], False),
            ((True, False, True), ["c"], True),
            ((False, True, True), ["a"], False),
            ((True, False, False), ["a"], True),
            ((False, True, False), ["b"], True),
            ((False, False, True), ["a", "b"], False),
            ((False, False, False), ["a", "b", "c"], False),
        ],
    )
    def test_may_color_needs_any_enabled_item(self, flags, items, expected):
        assert safety_app.may_color(*flags, "a", "b", "c", items) is expected


class TestGeometry:
    def test_overlap(self):
        assert safety_app.does_overlap(0, 0, 10, 10, 5, 5, 15, 15)
        assert not safety_app.does_overlap(0, 0, 10, 10, 20, 20, 30, 30)

    def test_touching_edges_do_not_overlap(self):
        assert not safety_app.does_overlap(0, 0, 10, 10, 10, 0, 20, 10)

    def test_box_intersects_polygon(self):
        assert safety_app.does_intersect_poly(50, 50, 150, 150, ZONE)  # straddles the edge
        assert safety_app.does_intersect_poly(10, 10, 20, 20, ZONE)  # fully inside
        assert not safety_app.does_intersect_poly(120, 10, 170, 190, ZONE)


class TestDetectPPE:
    # Policy as the PPE page saves it by default: hardhat + vest required, flag missing hardhats.
    DEFAULT = dict(hardhats=True, vests=True, masks=False, no_hardhats=True, no_vests=False, no_masks=False)
    VIOLATIONS_ONLY = dict(hardhats=False, vests=False, masks=False, no_hardhats=True, no_vests=False, no_masks=False)

    def test_no_detections(self, image, drawn):
        assert safety_app.detect_ppe(image, [], **self.DEFAULT) is False
        assert drawn == []

    def test_ppe_without_person_is_ignored(self, image, drawn):
        assert safety_app.detect_ppe(image, [on_a("NO-Hardhat")], **self.DEFAULT) is False
        assert drawn == []

    def test_compliant_person_is_green(self, image, drawn):
        boxes = [PERSON_A, on_a("Hardhat"), det("Safety Vest", 20, 80, 50, 120)]
        assert safety_app.detect_ppe(image, boxes, **self.DEFAULT) is True
        assert [color for _, color in drawn] == [GREEN]

    def test_violation_is_red(self, image, drawn):
        assert safety_app.detect_ppe(image, [PERSON_A, on_a("NO-Hardhat")], **self.DEFAULT) is True
        assert [color for _, color in drawn] == [RED]

    def test_partial_ppe_is_neither(self, image, drawn):
        # Hardhat but no vest: not compliant, and no enabled violation class either.
        assert safety_app.detect_ppe(image, [PERSON_A, on_a("Hardhat")], **self.DEFAULT) is False
        assert [color for _, color in drawn] == [BLUE]

    def test_conflicting_detections_are_blue_but_flagged(self, image, drawn):
        boxes = [PERSON_A, on_a("Hardhat"), det("Safety Vest", 20, 80, 50, 120), det("NO-Hardhat", 20, 130, 50, 160)]
        assert safety_app.detect_ppe(image, boxes, **self.DEFAULT) is True
        assert [color for _, color in drawn] == [BLUE]

    def test_disabled_violation_class_is_ignored(self, image, drawn):
        assert safety_app.detect_ppe(image, [PERSON_A, on_a("NO-Safety Vest")], **self.DEFAULT) is False

    def test_item_must_be_fully_inside_person_box(self, image, drawn):
        sticking_out = det("NO-Hardhat", 5, 5, 40, 40)  # starts left of and above the person
        assert safety_app.detect_ppe(image, [PERSON_A, sticking_out], **self.DEFAULT) is False

    def test_item_belongs_to_the_person_that_contains_it(self, image, drawn):
        assert safety_app.detect_ppe(image, [PERSON_A, PERSON_B, on_b("NO-Hardhat")], **self.VIOLATIONS_ONLY) is True
        assert [color for _, color in drawn] == [BLUE, RED]

    def test_label_lists_items_found_on_person(self, image, drawn):
        safety_app.detect_ppe(image, [PERSON_A, on_a("NO-Hardhat")], **self.DEFAULT)
        assert drawn[0][0] == "['NO-Hardhat']"

    @pytest.mark.xfail(strict=True, reason="flag is overwritten per person, so only the last person counts")
    def test_violation_is_reported_whatever_the_detection_order(self, image, drawn):
        violator_first = [PERSON_A, on_a("NO-Hardhat"), PERSON_B, on_b("Hardhat")]
        assert safety_app.detect_ppe(image, violator_first, **self.VIOLATIONS_ONLY) is True


class TestDetectProximity:
    def test_no_detections(self, image, drawn):
        assert safety_app.detect_proximity(image, [], True, True) is False

    @pytest.mark.parametrize("label, machines, vehicles", [("machinery", True, False), ("vehicle", False, True)])
    def test_person_overlapping_enabled_class(self, image, drawn, label, machines, vehicles):
        boxes = [PERSON_A, det(label, 40, 100, 110, 180)]
        assert safety_app.detect_proximity(image, boxes, machines, vehicles) is True
        assert (label, BLUE) in drawn
        assert ("Person", RED) in drawn

    @pytest.mark.parametrize("label, machines, vehicles", [("machinery", False, True), ("vehicle", True, False)])
    def test_disabled_class_is_ignored(self, image, drawn, label, machines, vehicles):
        boxes = [PERSON_A, det(label, 40, 100, 110, 180)]
        assert safety_app.detect_proximity(image, boxes, machines, vehicles) is False
        assert drawn == [("Person", GREEN)]

    def test_person_clear_of_machinery(self, image, drawn):
        boxes = [PERSON_A, det("machinery", 120, 100, 190, 180)]
        assert safety_app.detect_proximity(image, boxes, True, True) is False
        assert ("Person", GREEN) in drawn

    def test_machinery_without_person(self, image, drawn):
        assert safety_app.detect_proximity(image, [det("machinery", 40, 100, 110, 180)], True, True) is False

    def test_any_person_too_close_raises_the_flag(self, image, drawn):
        machinery = det("machinery", 40, 100, 110, 180)  # overlaps A only
        for boxes in ([PERSON_A, PERSON_B, machinery], [PERSON_B, PERSON_A, machinery]):
            assert safety_app.detect_proximity(image, boxes, True, True) is True


class TestDetectZone:
    def zone(self, image, boxes, persons=True, machines=False, vehicles=False, inclusion=False, max_allowed=0, poly=ZONE):
        return safety_app.detect_zone(image, boxes, poly, persons, machines, vehicles, inclusion, max_allowed)

    def test_missing_polygon_returns_none(self, image, drawn):
        assert self.zone(image, [PERSON_A], poly=None) is None

    def test_zone_outline_is_drawn_on_the_frame(self, image, drawn):
        self.zone(image, [])
        assert image.any()

    def test_no_detections(self, image, drawn):
        assert self.zone(image, []) is False

    def test_person_inside_exclusion_zone(self, image, drawn):
        assert self.zone(image, [PERSON_A]) is True
        assert drawn == [("Person", RED)]

    def test_person_outside_exclusion_zone(self, image, drawn):
        assert self.zone(image, [PERSON_B]) is False
        assert drawn == [("Person", GREEN)]

    def test_box_straddling_the_zone_edge_counts(self, image, drawn):
        assert self.zone(image, [det("Person", 80, 10, 130, 190)]) is True

    @pytest.mark.parametrize(
        "label, toggles",
        [("Person", dict(persons=True)), ("machinery", dict(persons=False, machines=True)), ("vehicle", dict(persons=False, vehicles=True))],
    )
    def test_each_enabled_class_is_checked(self, image, drawn, label, toggles):
        assert self.zone(image, [det(label, 10, 10, 60, 190)], **toggles) is True

    @pytest.mark.parametrize("label", ["machinery", "vehicle"])
    def test_disabled_class_is_ignored(self, image, drawn, label):
        assert self.zone(image, [det(label, 10, 10, 60, 190)]) is False
        assert drawn == []

    def test_any_object_in_zone_raises_the_flag(self, image, drawn):
        for boxes in ([PERSON_A, PERSON_B], [PERSON_B, PERSON_A]):
            assert self.zone(image, boxes) is True

    def test_max_allowed_tolerates_that_many(self, image, drawn):
        assert self.zone(image, [PERSON_A], max_allowed=1) is False
        both_inside = [PERSON_A, det("Person", 30, 10, 90, 190)]
        assert self.zone(image, both_inside, max_allowed=1) is True

    @pytest.mark.xfail(strict=True, reason="max_allowed is compared with the count in the whole frame, not in the zone")
    def test_max_allowed_counts_only_objects_in_the_zone(self, image, drawn):
        one_inside_one_outside = [PERSON_A, PERSON_B]
        assert self.zone(image, one_inside_one_outside, max_allowed=1) is False

    def test_inclusion_zone_flags_when_empty(self, image, drawn):
        assert self.zone(image, [PERSON_B], inclusion=True) is True
        assert drawn == [("Person", RED)]

    def test_inclusion_zone_clears_when_occupied(self, image, drawn):
        assert self.zone(image, [PERSON_A], inclusion=True) is False
        assert drawn == [("Person", GREEN)]

    def test_inclusion_zone_with_no_detections_is_flagged(self, image, drawn):
        assert self.zone(image, [], inclusion=True) is True

    @pytest.mark.xfail(strict=True, reason="a policy saved without a drawn zone reaches detect_zone as an empty polygon")
    def test_empty_polygon_does_not_crash(self, image, drawn):
        assert self.zone(image, [PERSON_A], poly=[]) is False
