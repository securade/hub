import numpy as np
import pytest

from conftest import detections_json

# Boxes are [y1, x1, y2, x2] in 0-1000 range.
A = [100, 100, 400, 400]
B_OVERLAPS_A = [110, 110, 410, 410]
C = [600, 600, 900, 900]


@pytest.fixture
def image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_slug_and_name(make_plugin):
    plugin = make_plugin()
    assert plugin.SLUG == "agent_detector"
    assert plugin.NAME == "Agent Detector"


class TestConfidenceSlider:
    """Regression tests: the slider is a confidence threshold, not an IoU threshold."""

    REPLY = detections_json([(0.9, A), (0.7, B_OVERLAPS_A), (0.6, C), (0.3, [300, 700, 500, 950])])

    @pytest.mark.parametrize(
        "threshold, expected",
        [
            (0.2, 3),  # 0.9, 0.6, 0.3 kept; 0.7 suppressed by NMS (overlaps 0.9)
            (0.5, 2),  # 0.9, 0.6
            (0.8, 1),  # 0.9 only
            (0.95, 0),
        ],
    )
    def test_threshold_filters_by_confidence(self, make_plugin, image, threshold, expected):
        plugin = make_plugin(self.REPLY)
        result = plugin.extract_bounding_boxes(image, "", "thing", threshold)
        assert len(result) == expected

    def test_confidence_equal_to_threshold_is_kept(self, make_plugin, image):
        plugin = make_plugin(detections_json([(1.0, A)]))
        assert len(plugin.extract_bounding_boxes(image, "", "thing", 1.0)) == 1

    def test_threshold_does_not_change_overlap_suppression(self, make_plugin, image):
        # Both overlapping boxes pass the confidence filter at 0.1 and 0.6;
        # NMS must drop the lower-confidence one either way.
        reply = detections_json([(0.9, A), (0.8, B_OVERLAPS_A)])
        for threshold in (0.1, 0.6):
            plugin = make_plugin(reply)
            result = plugin.extract_bounding_boxes(image, "", "thing", threshold)
            assert len(result) == 1
            assert result[0]["confidence"] == 0.9

    def test_run_passes_slider_through(self, make_plugin, image):
        plugin = make_plugin(detections_json([(0.6, A)]))
        out = plugin.run(image, prompt="thing", confidence_threshold=0.9)
        assert np.array_equal(out, image)  # nothing detected -> unannotated copy
        plugin = make_plugin(detections_json([(0.6, A)]))
        out = plugin.run(image, prompt="thing", confidence_threshold=0.5)
        assert not np.array_equal(out, image)  # box drawn


class TestRun:
    def test_missing_prompt_returns_input(self, make_plugin, image):
        plugin = make_plugin()
        assert plugin.run(image) is image

    def test_api_error_returns_input(self, make_plugin, image):
        plugin = make_plugin()

        def boom(**kwargs):
            raise RuntimeError("api down")

        plugin.client.chat.completions.create = boom
        assert np.array_equal(plugin.run(image, prompt="thing"), image)

    def test_malformed_json_returns_unannotated_copy(self, make_plugin, image):
        plugin = make_plugin("not json at all")
        out = plugin.run(image, prompt="thing")
        assert np.array_equal(out, image)

    def test_request_contains_prompt_and_image(self, make_plugin, image):
        plugin = make_plugin("[]")
        plugin.run(image, prompt="red car")
        content = plugin.client.calls[0]["messages"][0]["content"]
        assert "red car" in content[0]["text"]
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


class TestHelpers:
    def test_extract_json_from_array_in_prose(self, make_plugin):
        plugin = make_plugin()
        assert plugin.extract_json_from_response('Sure! [{"a": 1}] done') == '[{"a": 1}]'

    def test_extract_json_wraps_single_object(self, make_plugin):
        plugin = make_plugin()
        assert plugin.extract_json_from_response('{"a": 1}') == '[{"a": 1}]'

    def test_extract_json_no_json(self, make_plugin):
        assert make_plugin().extract_json_from_response("nothing") == "[]"

    def test_normalize_coordinates(self, make_plugin):
        assert make_plugin().normalize_coordinates(500, 250, 200, 400) == (100, 100)

    def test_validate_bbox_converts_yxyx_to_pixel_xyxy(self, make_plugin):
        assert make_plugin().validate_and_adjust_bbox([100, 200, 500, 600], 1000, 1000) == [200, 100, 600, 500]

    @pytest.mark.parametrize(
        "bbox",
        [[1, 2, 3], [0, 0, 1001, 10], [0, 0, 5, 5], ["a", "b", "c", "d"]],
    )
    def test_validate_bbox_rejects_bad_input(self, make_plugin, bbox):
        assert make_plugin().validate_and_adjust_bbox(bbox, 1000, 1000) is None

    def test_iou_identical_and_disjoint(self, make_plugin):
        plugin = make_plugin()
        assert plugin.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
        assert plugin.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_nms_keeps_highest_confidence(self, make_plugin):
        plugin = make_plugin()
        dets = [
            {"confidence": 0.5, "bbox": [0, 0, 10, 10]},
            {"confidence": 0.9, "bbox": [1, 1, 10, 10]},
            {"confidence": 0.4, "bbox": [50, 50, 60, 60]},
        ]
        kept = plugin.non_max_suppression(dets)
        assert [d["confidence"] for d in kept] == [0.9, 0.4]

    def test_nms_empty(self, make_plugin):
        assert make_plugin().non_max_suppression([]) == []
