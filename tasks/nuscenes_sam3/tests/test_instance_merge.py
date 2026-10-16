"""Synthetic original-point merge tests; no models, GT or nuScenes required."""

import copy
from itertools import permutations
import json
import math
import unittest

from tasks.nuscenes_sam3.python.instance_merge import merge_instances_by_points


def instance(identifier=1, indices=(), camera="CAM_FRONT", class_id=4, score=.8):
    return {"global_instance_id": identifier, "camera": camera, "class_id": class_id,
            "class_name": "car" if class_id <= 10 else "driveable_surface",
            "score": score, "point_indices": list(indices)}


class PointMergeTests(unittest.TestCase):
    def test_basic_three_shared_at_overlap_boundary(self):
        records = [instance(8, range(10)), instance(21, range(7, 17), "CAM_BACK")]
        result = merge_instances_by_points(records)
        self.assertEqual(result["mapping"], {8: 8, 21: 8})
        self.assertEqual(len(result["groups"]), 1)
        group, pair = result["groups"][0], result["pairs"][0]
        self.assertEqual(group["member_instance_ids"], [8, 21])
        self.assertEqual(group["source_cameras"], ["CAM_BACK", "CAM_FRONT"])
        self.assertEqual((group["merged_instance_id"], group["member_count"], group["merged"]), (8, 2, True))
        self.assertEqual((pair["shared_points"], pair["union_points"]), (3, 17))
        self.assertEqual(pair["overlap_smaller"], .3)
        self.assertAlmostEqual(pair["point_iou"], 3 / 17)
        self.assertTrue(pair["eligible"])
        self.assertEqual(pair["decision"], "accepted")
        self.assertEqual(group["point_indices"], list(range(17)))
        self.assertEqual(group["source_point_observations"], 20)
        self.assertEqual(group["deduplicated_shared_observations"], 3)

    def test_partial_visibility_uses_smaller_overlap_not_iou(self):
        result = merge_instances_by_points([
            instance(1, range(100)), instance(2, range(3), "CAM_BACK")])
        pair = result["pairs"][0]
        self.assertEqual(pair["overlap_smaller"], 1)
        self.assertEqual(pair["point_iou"], .03)
        self.assertEqual(pair["decision"], "accepted")

    def test_disjoint_pair_is_logged_and_rejected(self):
        result = merge_instances_by_points([
            instance(1, range(3)), instance(2, range(3, 6), "CAM_BACK")])
        self.assertEqual(len(result["pairs"]), 1)
        pair = result["pairs"][0]
        self.assertEqual((pair["shared_points"], pair["point_iou"], pair["overlap_smaller"]), (0, 0, 0))
        self.assertFalse(pair["eligible"])
        self.assertEqual(pair["decision"], "rejected_threshold")
        self.assertEqual(result["mapping"], {1: 1, 2: 2})
        self.assertEqual(result["metadata"]["rejected_threshold"], 1)

    def test_same_camera_or_different_class_never_compared(self):
        for other in (instance(2, range(3)), instance(2, range(3), "CAM_BACK", class_id=5)):
            with self.subTest(other=other):
                result = merge_instances_by_points([instance(1, range(3)), other])
                self.assertEqual(result["pairs"], [])
                self.assertEqual(result["mapping"], {1: 1, 2: 2})
                # Separate classes may retain the exact same original point indices.
                self.assertEqual([group["point_indices"] for group in result["groups"]], [[0, 1, 2]] * 2)

    def test_thresholds_inclusive_and_both_required(self):
        records = [instance(1, range(10)), instance(2, range(7, 17), "CAM_BACK")]
        for shared, overlap, expected in ((3, .3, True), (4, .3, False),
                                           (3, math.nextafter(.3, 1), False),
                                           (3, math.nextafter(.3, 0), True), (1, 1, False)):
            with self.subTest(shared=shared, overlap=overlap):
                result = merge_instances_by_points(records, shared, overlap)
                self.assertEqual(result["pairs"][0]["eligible"], expected)
        exact = merge_instances_by_points([instance(1, [3]), instance(2, [3], "CAM_BACK")], 1, 1)
        self.assertTrue(exact["groups"][0]["merged"])

    def test_duplicate_points_deduplicated_before_thresholds_and_counts(self):
        result = merge_instances_by_points([
            instance(1, [1, 1, 2, 3, 3]), instance(2, [3, 3, 2, 1, 1], "CAM_BACK")])
        pair, group = result["pairs"][0], result["groups"][0]
        self.assertEqual((pair["size_a"], pair["size_b"], pair["shared_points"]), (3, 3, 3))
        self.assertEqual(group["point_indices"], [1, 2, 3])
        self.assertEqual((group["source_point_observations"], group["deduplicated_shared_observations"]), (6, 3))
        repeated = merge_instances_by_points([
            instance(1, [1, 1, 1]), instance(2, [1, 1, 1], "CAM_BACK")])
        self.assertFalse(repeated["pairs"][0]["eligible"])

    def test_empty_and_stuff_singletons_are_preserved(self):
        records = [instance(9), instance(17, range(3), "CAM_BACK"),
                   instance(25, range(3), "CAM_LEFT", 11),
                   instance(26, range(3), "CAM_RIGHT", 11),
                   instance(40, range(3), "CAM_FRONT", 16)]
        result = merge_instances_by_points(records)
        self.assertEqual(result["mapping"], {9: 9, 17: 17, 25: 25, 26: 26, 40: 40})
        self.assertEqual(result["pairs"], [])
        self.assertTrue(all(group["member_count"] == 1 and not group["merged"] for group in result["groups"]))
        self.assertEqual(result["metadata"]["input_foreground_instances"], 2)
        self.assertEqual(result["metadata"]["nonempty_foreground_instances"], 1)
        self.assertEqual(result["groups"][0]["point_indices"], [])

    def test_tie_break_ids_and_input_permutations_are_deterministic(self):
        records = [instance(30, range(3), "CAM_FRONT"), instance(10, range(3), "CAM_BACK"),
                   instance(20, range(3), "CAM_FRONT")]
        expected = merge_instances_by_points(records)
        self.assertEqual(expected["mapping"], {10: 10, 20: 10, 30: 30})
        self.assertEqual([pair["decision"] for pair in expected["pairs"]],
                         ["accepted", "rejected_camera_conflict"])
        for shuffled in permutations(records):
            self.assertEqual(merge_instances_by_points(list(shuffled)), expected)

    def test_candidate_order_priorities_before_ids(self):
        # Every case gives the higher-ID same-camera competitor the stronger edge.
        cases = [
            ([0, 1, 2, 3], [0, 1, 2], [0, 1, 2, 3], .8, .8),  # overlap
            (list(range(10)), list(range(20)), list(range(10)), .8, .8),  # IoU
            (list(range(10)), [0, 1, 2], list(range(5)), .8, .8),  # shared count
            ([0, 1, 2], [0, 1, 2], [0, 1, 2], .5, .9),  # minscore
        ]
        for anchor, weaker, stronger, low_score, high_score in cases:
            with self.subTest(anchor=anchor, weaker=weaker, stronger=stronger, scores=(low_score, high_score)):
                result = merge_instances_by_points([
                    instance(1, anchor, "CAM_BACK", score=1),
                    instance(2, weaker, "CAM_FRONT", score=low_score),
                    instance(3, stronger, "CAM_FRONT", score=high_score)])
                self.assertEqual(result["mapping"], {1: 1, 2: 2, 3: 1})

    def test_camera_conflict_between_existing_groups(self):
        records = [instance(1, range(5), "CAM_A"),
                   instance(2, range(6), "CAM_B"),
                   instance(3, range(3), "CAM_A")]
        result = merge_instances_by_points(records)
        self.assertEqual(result["mapping"], {1: 1, 2: 1, 3: 3})
        self.assertEqual(result["pairs"][1]["decision"], "rejected_camera_conflict")
        self.assertEqual(result["metadata"]["rejected_camera_conflict"], 1)
        for group in result["groups"]:
            self.assertEqual(len(group["source_cameras"]), group["member_count"])

    def test_chain_with_zero_endpoint_evidence_rejected_by_complete_link(self):
        result = merge_instances_by_points([
            instance(1, range(3), "CAM_A"), instance(2, range(6), "CAM_B"),
            instance(3, range(3, 6), "CAM_C")])
        self.assertEqual(result["mapping"], {1: 1, 2: 1, 3: 3})
        self.assertEqual([pair["decision"] for pair in result["pairs"]],
                         ["accepted", "rejected_threshold", "rejected_group_consistency"])
        self.assertEqual(result["metadata"]["eligible_pairs"], 2)
        self.assertEqual(result["metadata"]["rejected_group_consistency"], 1)

    def test_chain_with_nonzero_below_threshold_endpoint_evidence_rejected(self):
        result = merge_instances_by_points([
            instance(1, [0, 1, 2, 3], "CAM_A"),
            instance(2, [0, 1, 2, 3, 4, 5], "CAM_B"),
            instance(3, [2, 3, 4, 5], "CAM_C")])
        self.assertEqual(result["mapping"], {1: 1, 2: 1, 3: 3})
        self.assertEqual(result["pairs"][1]["shared_points"], 2)
        self.assertEqual(result["pairs"][2]["decision"], "rejected_group_consistency")

    def test_three_camera_clique_merges_and_cycle_logged(self):
        result = merge_instances_by_points([
            instance(90, [0, 1, 2, 3], "CAM_A", score=.1),
            instance(40, [0, 1, 2, 4], "CAM_B", score=.9),
            instance(70, [0, 1, 2, 5], "CAM_C", score=.6)])
        group = result["groups"][0]
        self.assertEqual(group["merged_instance_id"], 40)
        self.assertEqual(group["member_instance_ids"], [40, 70, 90])
        self.assertEqual(group["point_indices"], list(range(6)))
        self.assertEqual(group["score"], .9)
        self.assertEqual(group["source_point_observations"], 12)
        self.assertEqual(group["deduplicated_shared_observations"], 6)
        self.assertEqual(result["metadata"]["eligible_pairs"], 3)
        self.assertEqual(result["metadata"]["accepted_edges"], 2)
        self.assertEqual(result["metadata"]["cycle_redundant"], 1)
        self.assertEqual(result["metadata"]["reduction"], 2)

    def test_every_cross_group_member_pair_must_qualify(self):
        # A-B and C-D form first. Three of four cross pairs qualify, B-D does not.
        records = [instance(1, range(10), "CAM_A"),
                   instance(2, range(9), "CAM_B"),
                   instance(3, range(5, 15), "CAM_C"),
                   instance(4, range(6, 15), "CAM_D")]
        result = merge_instances_by_points(records, min_shared_points=4)
        self.assertEqual(result["mapping"], {1: 1, 2: 1, 3: 3, 4: 3})
        self.assertEqual(result["metadata"]["rejected_group_consistency"], 3)

    def test_all_input_points_and_instances_survive_without_mutation(self):
        records = [instance(101, [10, 11, 12, 10000], "CAM_A"),
                   instance(202, [10, 11, 12, 20000], "CAM_B"),
                   instance(303, [12], "CAM_C", 5), instance(404, [], "CAM_D"),
                   instance(505, [10, 88], "CAM_E", 12)]
        records[0].update(roi_point_indices=[], hull_xy=[[999, 999]],
                          sample_token="unused", extra={"nested": [1, 2]})
        before = copy.deepcopy(records)
        result = merge_instances_by_points(records)
        self.assertEqual(records, before)
        by_id = {group["merged_instance_id"]: group for group in result["groups"]}
        self.assertEqual(set(result["mapping"]), {item["global_instance_id"] for item in records})
        flattened = [identifier for group in result["groups"] for identifier in group["member_instance_ids"]]
        self.assertCountEqual(flattened, result["mapping"])
        for item in records:
            group = by_id[result["mapping"][item["global_instance_id"]]]
            self.assertTrue(set(item["point_indices"]).issubset(group["point_indices"]))
        self.assertEqual(by_id[101]["point_indices"], [10, 11, 12, 10000, 20000])
        self.assertEqual(len(result["groups"]), 4)
        self.assertFalse(result["metadata"]["uses_gt"])
        self.assertFalse(result["metadata"]["uses_geometry"])
        self.assertFalse(result["metadata"]["cross_frame_identity"])
        self.assertIn("complete-link", result["metadata"]["rule"])
        self.assertIn("no cross-frame", result["metadata"]["identity_scope"])
        encoded = json.loads(json.dumps(result))
        self.assertEqual(encoded["mapping"], {str(key): value for key, value in result["mapping"].items()})
        result["groups"][0]["point_indices"].append(999999)
        result["groups"][0]["member_instance_ids"].append(777)
        self.assertEqual(records, before)

    def test_empty_input_and_finite_scores_outside_probability_range(self):
        result = merge_instances_by_points([])
        self.assertEqual((result["groups"], result["pairs"], result["mapping"]), ([], [], {}))
        for key in ("input_instances", "output_groups", "merged_groups", "eligible_pairs", "accepted_edges"):
            self.assertEqual(result["metadata"][key], 0)
        result = merge_instances_by_points([
            instance(1, range(3), score=-5), instance(2, range(3), "CAM_BACK", score=7)])
        self.assertEqual(result["groups"][0]["score"], 7)


class ValidationTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for value in (0, -1, 1.2, 3., True, "3", None, float("nan")):
            with self.subTest(min_shared_points=value), self.assertRaises((TypeError, ValueError)):
                merge_instances_by_points([], min_shared_points=value)
        for value in (0, -.1, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.3", None):
            with self.subTest(min_overlap=value), self.assertRaises((TypeError, ValueError)):
                merge_instances_by_points([], min_overlap=value)

    def test_invalid_top_level_or_missing_fields(self):
        for value in (None, {}, (), [None], [1], [[]], [{}]):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                merge_instances_by_points(value)
        for key in instance():
            record = instance()
            del record[key]
            with self.subTest(missing=key), self.assertRaises((TypeError, ValueError)):
                merge_instances_by_points([record])

    def test_invalid_ids_labels_camera_names_scores_and_point_indices(self):
        cases = {
            "global_instance_id": (0, -2, True, 1., "1", None),
            "class_id": (0, 17, -1, True, 4., "4", None),
            "class_name": (None, 4, []),
            "camera": (None, "", 1, " ", "CAM FRONT", "..", ".", "CAM/A", "CAM\\A", "CAM\x00A"),
            "score": (True, "0.5", None, complex(1, 1), float("nan"), float("inf"), -float("inf")),
            "point_indices": (None, 3, "123", {"index": 0}, [-1], [True], [1.], ["1"], [[1]], [None]),
        }
        for key, values in cases.items():
            for value in values:
                record = instance()
                record[key] = value
                with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                    merge_instances_by_points([record])

    def test_duplicate_ids_including_empty_or_stuff_are_invalid(self):
        for second in (instance(), instance(class_id=16), instance(indices=[1])):
            with self.subTest(second=second), self.assertRaises(ValueError):
                merge_instances_by_points([instance(), second])

    def test_all_sixteen_class_ids_valid_foreground_boundaries(self):
        for class_id in range(1, 17):
            with self.subTest(class_id=class_id):
                result = merge_instances_by_points([
                    instance(1, range(3), "CAM_A", class_id),
                    instance(2, range(3), "CAM_B", class_id)])
                self.assertEqual(len(result["pairs"]), int(class_id <= 10))
                self.assertEqual(len(result["groups"]), 1 if class_id <= 10 else 2)


if __name__ == "__main__":
    unittest.main()
