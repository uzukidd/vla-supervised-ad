"""Synthetic exact point-filter tests; no models, scipy, sklearn or nuScenes."""

import argparse
import copy
import json
import unittest
from unittest import mock

import numpy as np

from tasks.nuscenes_sam3.python.instance_filter import (
    add_filter_args, clean_instances, clean_point_support, config_from_args,
    filter_instance_points, validate_filter,
)


def points(xs):
    return np.column_stack((np.asarray(xs, dtype=float), np.zeros((len(xs), 2))))


def record(indices, class_id=4, **extra):
    return dict(class_id=class_id, point_indices=list(indices), **extra)


def dbscan_config(**updates):
    return dict(algorithm="dbscan", dbscan_eps=.3, dbscan_min_samples=3,
                cluster_policy="largest", **updates)


class FilterConfigTests(unittest.TestCase):
    def test_defaults_and_fresh_normalized_values(self):
        expected = {"algorithm": "none", "sor_k": 16, "sor_std": 2.,
                    "dbscan_eps": 1., "dbscan_min_samples": 5,
                    "cluster_policy": "significant", "min_cluster_size": 10,
                    "cluster_ratio": .1}
        self.assertEqual(validate_filter(), expected)
        first = validate_filter()
        first["sor_k"] = 3
        self.assertEqual(validate_filter(), expected)
        normalized = validate_filter({"sor_k": np.int64(7), "dbscan_eps": np.float32(.5)})
        self.assertIs(type(normalized["sor_k"]), int)
        self.assertIs(type(normalized["dbscan_eps"]), float)

    def test_all_parameters_validated_when_disabled(self):
        invalid = [{"algorithm": "random"}, {"cluster_policy": "all"},
                   {"sor_k": 0}, {"sor_k": 1.5}, {"sor_k": True},
                   {"sor_std": 0}, {"sor_std": float("nan")},
                   {"dbscan_eps": -1}, {"dbscan_eps": float("inf")},
                   {"dbscan_min_samples": 0}, {"min_cluster_size": False},
                   {"cluster_ratio": 0}, {"cluster_ratio": 1.01}, {"typo": 3}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_filter(config)
        with self.assertRaises(ValueError):
            validate_filter("none")

    def test_argparse_round_trip(self):
        parser = add_filter_args(argparse.ArgumentParser())
        self.assertEqual(config_from_args(parser.parse_args([])), validate_filter())
        args = parser.parse_args(["--point-filter", "sor_dbscan", "--sor-k", "3",
                                  "--sor-std", "1.5", "--dbscan-eps", "2",
                                  "--dbscan-min-samples", "2", "--cluster-policy", "largest",
                                  "--min-cluster-size", "7", "--cluster-ratio", ".25"])
        self.assertEqual(config_from_args(args), {
            "algorithm": "sor_dbscan", "sor_k": 3, "sor_std": 1.5,
            "dbscan_eps": 2., "dbscan_min_samples": 2, "cluster_policy": "largest",
            "min_cluster_size": 7, "cluster_ratio": .25})
        self.assertEqual(config_from_args(argparse.Namespace()), validate_filter())


class InstanceFilterTests(unittest.TestCase):
    def test_none_preserves_unique_sorted_original_indices(self):
        xyz = points([0, 1, 2, 10])
        result = filter_instance_points(xyz, [3, 0, 3, 2])
        np.testing.assert_array_equal(result["kept_indices"], [0, 2, 3])
        self.assertEqual(result["kept_indices"].dtype, np.int64)
        self.assertEqual(result["removed_indices"].dtype, np.int64)
        self.assertEqual(result["cluster_labels"].dtype, np.int32)
        self.assertEqual(result["removed_indices"].tolist(), [])
        self.assertEqual(result["metadata"]["status"], "disabled")
        self.assertEqual(result["metadata"]["input"], 3)

    def test_empty_valid_for_all_algorithms(self):
        for algorithm in ("none", "sor", "dbscan", "sor_dbscan"):
            with self.subTest(algorithm=algorithm):
                result = filter_instance_points(np.empty((0, 3)), [], {"algorithm": algorithm})
                self.assertEqual(result["kept_indices"].tolist(), [])
                self.assertEqual(result["removed_indices"].tolist(), [])
                self.assertEqual(result["cluster_labels"].tolist(), [])
                self.assertEqual(result["metadata"]["status"],
                                 "disabled" if algorithm == "none" else "too_few")
                json.dumps(result["metadata"], allow_nan=False)

    def test_sor_preserves_one_and_two_points(self):
        for count in (1, 2):
            result = filter_instance_points(points([0, 100][:count]), range(count), {"algorithm": "sor"})
            self.assertEqual(result["kept_indices"].tolist(), list(range(count)))
            self.assertEqual(result["metadata"]["status"], "too_few")
            self.assertEqual(result["metadata"]["stages"][0]["k"], count - 1)

    def test_sor_removes_isolated_outlier(self):
        xyz = points([0, .01, .02, .03, .04, .05, .06, .07, .08, 100])
        result = filter_instance_points(xyz, range(10), {"algorithm": "sor", "sor_k": 3})
        self.assertEqual(result["removed_indices"].tolist(), [9])
        self.assertEqual(result["cluster_labels"].tolist(), [0] * 9 + [-2])

    def test_sor_exact_3d_knn_population_std_and_chunk_boundary(self):
        xyz = np.random.default_rng(41).normal(size=(280, 3))
        config = {"algorithm": "sor", "sor_k": 7, "sor_std": 1.3}
        squared = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(squared, np.inf)
        distances = np.sqrt(np.sort(squared, axis=1)[:, :7]).mean(axis=1)
        threshold = distances.mean() + 1.3 * distances.std(ddof=0)
        result = filter_instance_points(xyz, list(reversed(range(280))), config)
        np.testing.assert_array_equal(result["kept_indices"], np.flatnonzero(distances <= threshold))
        stage = result["metadata"]["stages"][0]
        self.assertAlmostEqual(stage["mean_distance"], float(distances.mean()))
        self.assertAlmostEqual(stage["std_distance"], float(distances.std(ddof=0)))
        self.assertAlmostEqual(stage["threshold"], float(threshold))

    def test_sor_clamps_k_and_excludes_self(self):
        xyz = points([0, 2, 5])
        result = filter_instance_points(xyz, [2, 0, 1], {"algorithm": "sor", "sor_k": 16})
        means = np.array([3.5, 2.5, 4.])
        stage = result["metadata"]["stages"][0]
        self.assertEqual(stage["k"], 2)
        self.assertAlmostEqual(stage["mean_distance"], means.mean())
        self.assertAlmostEqual(stage["std_distance"], means.std(ddof=0))

    def test_sor_duplicate_coordinates_have_zero_distance(self):
        xyz = np.full((7, 3), 1e10)
        result = filter_instance_points(xyz, [6, 1, 0, 4, 3, 2, 5, 1], {"algorithm": "sor"})
        self.assertEqual(result["kept_indices"].tolist(), list(range(7)))
        stage = result["metadata"]["stages"][0]
        self.assertEqual((stage["mean_distance"], stage["std_distance"], stage["threshold"]), (0., 0., 0.))

    def test_dbscan_cluster_and_noise(self):
        result = filter_instance_points(points([0, .1, .2, 4]), [3, 2, 0, 1], dbscan_config())
        self.assertEqual(result["kept_indices"].tolist(), [0, 1, 2])
        self.assertEqual(result["removed_indices"].tolist(), [3])
        self.assertEqual(result["cluster_labels"].tolist(), [0, 0, 0, -1])
        self.assertEqual(result["metadata"]["stages"][0]["noise_points"], 1)

    def test_dbscan_all_noise_is_removed_no_fallback(self):
        for xs in ([0, 5, 10], [0], [0, .1]):
            with self.subTest(xs=xs):
                result = filter_instance_points(points(xs), range(len(xs)), dbscan_config())
                self.assertEqual(result["kept_indices"].tolist(), [])
                self.assertEqual(result["removed_indices"].tolist(), list(range(len(xs))))
                self.assertEqual(result["cluster_labels"].tolist(), [-1] * len(xs))

    def test_dbscan_radius_exact_3d_and_negative_cells(self):
        xyz = np.array([[-1., 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1]])
        result = filter_instance_points(xyz, range(4), {
            "algorithm": "dbscan", "dbscan_eps": 1., "dbscan_min_samples": 3,
            "cluster_policy": "largest"})
        self.assertEqual(result["kept_indices"].tolist(), [0, 1, 2])
        self.assertEqual(result["cluster_labels"].tolist(), [0, 0, 0, -1])

    def test_dbscan_min_samples_includes_self_duplicates(self):
        xyz = np.zeros((5, 3))
        result = filter_instance_points(xyz, range(5), {"algorithm": "dbscan", "cluster_policy": "largest"})
        self.assertEqual(result["kept_indices"].tolist(), list(range(5)))
        too_many = filter_instance_points(xyz, range(5), {
            "algorithm": "dbscan", "dbscan_min_samples": 6, "cluster_policy": "largest"})
        self.assertEqual(too_many["kept_indices"].tolist(), [])

    def test_dbscan_previously_visited_noise_becomes_border(self):
        xyz = points([0, .2, .3, .4, .6])
        result = filter_instance_points(xyz, range(5), {
            "algorithm": "dbscan", "dbscan_eps": .25, "dbscan_min_samples": 4,
            "cluster_policy": "largest"})
        self.assertEqual(result["cluster_labels"].tolist(), [0] * 5)
        self.assertEqual(result["kept_indices"].tolist(), list(range(5)))

    def test_two_legitimate_groups_significant_vs_largest(self):
        xyz = points([0] * 20 + [5] * 10 + [10] * 3 + [100])
        significant = dict(algorithm="dbscan", dbscan_eps=.1, dbscan_min_samples=3,
                           cluster_policy="significant", min_cluster_size=10, cluster_ratio=.1)
        result = filter_instance_points(xyz, range(34), significant)
        self.assertEqual(result["kept_indices"].tolist(), list(range(30)))
        stage = result["metadata"]["stages"][0]
        self.assertEqual(stage["cluster_sizes"], [20, 10, 3])
        self.assertEqual(stage["kept_cluster_labels"], [0, 1])
        self.assertEqual(stage["discarded_cluster_points"], 3)
        self.assertTrue(stage["warning"])
        largest = filter_instance_points(xyz, range(34), dict(significant, cluster_policy="largest"))
        self.assertEqual(largest["kept_indices"].tolist(), list(range(20)))
        # Policy does not overwrite diagnostic cluster identities.
        self.assertEqual(largest["cluster_labels"][20:30].tolist(), [1] * 10)

    def test_significant_relative_threshold_and_minimum(self):
        xyz = points([0] * 20 + [5] * 10)
        config = {"algorithm": "dbscan", "min_cluster_size": 3, "cluster_ratio": .5}
        both = filter_instance_points(xyz, range(30), config)
        self.assertEqual(len(both["kept_indices"]), 30)
        one = filter_instance_points(xyz, range(30), dict(config, cluster_ratio=.51))
        self.assertEqual(len(one["kept_indices"]), 20)
        none = filter_instance_points(xyz, range(30), dict(config, min_cluster_size=21))
        self.assertEqual(len(none["kept_indices"]), 0)

    def test_largest_tie_uses_first_original_index(self):
        xyz = points([9, 9, 9, 0, 0, 0])
        first = filter_instance_points(xyz, range(6), dbscan_config())
        second = filter_instance_points(xyz, [5, 3, 4, 1, 0, 2, 2], dbscan_config())
        for key in ("kept_indices", "removed_indices", "cluster_labels"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["kept_indices"].tolist(), [0, 1, 2])

    def test_combined_sor_then_dbscan_labels(self):
        xyz = points([0] * 10 + [3] * 5 + [100])
        result = filter_instance_points(xyz, range(16), {
            "algorithm": "sor_dbscan", "sor_k": 3, "dbscan_eps": .5,
            "dbscan_min_samples": 3, "cluster_policy": "largest"})
        self.assertEqual(result["kept_indices"].tolist(), list(range(10)))
        self.assertEqual(result["cluster_labels"].tolist(), [0] * 10 + [1] * 5 + [-2])
        self.assertEqual([stage["stage"] for stage in result["metadata"]["stages"]], ["sor", "dbscan"])
        self.assertEqual([stage["input"] for stage in result["metadata"]["stages"]], [16, 15])

    def test_bounds_shape_and_coordinate_validation(self):
        xyz = points([0, 1, 2])
        for indices in ([-1], [3], [1.], [True], [[0]], [2**64 - 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                filter_instance_points(xyz, indices)
        for coordinates in (np.zeros((3, 2)), [0, 1, 2], [[0, 0, float("nan")]],
                            [[0, 0, float("inf")]], [["a", "b", "c"]]):
            with self.subTest(xyz=coordinates), self.assertRaises(ValueError):
                filter_instance_points(coordinates, [])


class CleanSupportTests(unittest.TestCase):
    def test_instances_are_independent_even_for_same_class(self):
        xyz = points([0, 0, 0, 10, 10, 10])
        records = [record(range(3)), record(range(3, 6))]
        result = clean_instances(xyz, records, dbscan_config())
        self.assertEqual([item["point_indices"] for item in result["instances"]],
                         [list(range(3)), list(range(3, 6))])
        self.assertEqual(result["stats"]["input_instances"], 2)
        self.assertEqual(result["stats"]["per_class"], [
            {"class_id": 4, "instances": 2, "input": 6, "kept": 6, "removed": 0}])

    def test_records_and_nested_metadata_json_safe_and_no_mutation(self):
        xyz = points([0, .1, .2, 9])
        records = [record([3, 0, 2, 1, 1], camera="CAM_FRONT",
                          score=np.float32(.9), extra={"array": np.array([1, 2])})]
        before = copy.deepcopy(records)
        xyz_before = xyz.copy()
        result = clean_instances(xyz, records, dbscan_config())
        item = result["instances"][0]
        self.assertEqual(item["original_point_indices"], [0, 1, 2, 3])
        self.assertEqual(item["point_indices"], [0, 1, 2])
        self.assertEqual(item["removed_point_indices"], [3])
        json.dumps(result, allow_nan=False)
        item["extra"]["array"][0] = 7
        self.assertEqual(records[0]["point_indices"], before[0]["point_indices"])
        np.testing.assert_array_equal(records[0]["extra"]["array"], before[0]["extra"]["array"])
        np.testing.assert_array_equal(xyz, xyz_before)

    def test_keep_union_across_cameras(self):
        xyz = points([0, .1, .2, 5, 5.1, 5.2, 100])
        prediction = np.full(7, 4, dtype=np.uint8)
        records = [record([0, 1, 2, 3, 6], camera="CAM_FRONT"),
                   record([2, 3, 4, 5, 6], camera="CAM_BACK")]
        result = clean_point_support(xyz, prediction, records, dbscan_config())
        np.testing.assert_array_equal(result["prediction"], [4, 4, 4, 4, 4, 4, 0])
        np.testing.assert_array_equal(result["keep_mask"], [True] * 6 + [False])
        self.assertEqual(result["removed_indices"].tolist(), [6])
        self.assertEqual([item["point_indices"] for item in result["instances"]],
                         [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(prediction.tolist(), [4] * 7)
        self.assertIsNot(result["prediction"], prediction)
        json.dumps(result["metadata"], allow_nan=False)

    def test_other_class_instance_cannot_rescue_prediction(self):
        xyz = points([0, .1, .2, 5, 5.1, 5.2])
        prediction = np.array([4, 4, 4, 4, 5, 5], dtype=np.uint8)
        records = [record([0, 1, 2, 3], 4), record([3, 4, 5], 5)]
        result = clean_point_support(xyz, prediction, records, dbscan_config())
        self.assertEqual(result["prediction"].tolist(), [4, 4, 4, 0, 5, 5])
        self.assertEqual(result["metadata"]["per_class"], [
            {"class_id": 4, "input": 4, "kept": 3, "removed": 1},
            {"class_id": 5, "input": 2, "kept": 2, "removed": 0}])

    def test_missing_membership_raises_enabled(self):
        xyz = points([0, 1, 2])
        prediction = np.full(3, 4, dtype=np.uint8)
        for records in ([], [record([0, 1])], [record(range(3), 5)]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, "same-class"):
                clean_point_support(xyz, prediction, records, {"algorithm": "sor"})

    def test_none_identity_allows_empty_records(self):
        prediction = np.array([4, 0, 2], dtype=np.uint8)
        result = clean_point_support(points([0, 1, 2]), prediction, [])
        np.testing.assert_array_equal(result["prediction"], prediction)
        self.assertTrue(result["keep_mask"].all())
        self.assertEqual(result["removed_indices"].tolist(), [])
        self.assertEqual(result["instances"], [])
        self.assertEqual(result["metadata"]["status"], "disabled")
        result["prediction"][0] = 0
        self.assertEqual(prediction[0], 4)

    def test_empty_records_background_and_empty_scan(self):
        xyz = points([0, 1, 2])
        result = clean_point_support(xyz, np.zeros(3, dtype=np.uint8), [], {"algorithm": "dbscan"})
        self.assertTrue(result["keep_mask"].all())
        self.assertEqual(result["metadata"]["removed"], 0)
        result = clean_point_support(np.empty((0, 3)), np.empty(0, dtype=np.uint8), [], {"algorithm": "sor"})
        self.assertEqual(result["prediction"].shape, (0,))
        records = clean_instances(xyz, [record([])], dbscan_config())
        self.assertEqual(records["stats"]["empty_instances"], 1)
        self.assertEqual(records["instances"][0]["point_indices"], [])

    def test_surface_sparse_outlier_and_disconnected_instances_bypass_all_filters(self):
        xs = [0] * 10 + [5] * 5 + [100, 200]
        xyz = points(xs)
        records = [record(range(16), 254), record([16], 2), record([], 2)]
        for algorithm in ("none", "sor", "dbscan", "sor_dbscan"):
            config = dict(dbscan_config(), algorithm=algorithm, sor_k=3)
            with self.subTest(algorithm=algorithm), mock.patch(
                    "tasks.nuscenes_sam3.python.instance_filter._filter_points",
                    side_effect=AssertionError("surface instances must bypass filtering")):
                result = clean_instances(xyz, records, config, surface_class_ids=(254, 2, 254))
            for original, cleaned in zip(records, result["instances"]):
                self.assertEqual(cleaned["point_indices"], original["point_indices"])
                self.assertEqual(cleaned["original_point_indices"], original["point_indices"])
                self.assertEqual(cleaned["removed_point_indices"], [])
                self.assertEqual(cleaned["filter"]["status"], "surface_exempt")
                self.assertEqual(cleaned["filter"]["effective_algorithm"], "none")
                self.assertEqual(cleaned["filter"]["config"], validate_filter(config))
                self.assertEqual(cleaned["filter"]["stages"], [])
                self.assertIn("surface", cleaned["filter"]["reason"])
            self.assertEqual(result["stats"]["surface_class_ids"], [2, 254])
            self.assertEqual(result["stats"]["exempt_surface_points"], 17)
            self.assertEqual(result["stats"]["exempt_surface_instances"], 3)
            self.assertEqual(result["stats"]["removed"], 0)
            json.dumps(result, allow_nan=False)

    def test_surface_predictions_retained_while_other_instances_still_filter(self):
        xyz = points(([0] * 10 + [5] * 5 + [100]) * 2)
        prediction = np.array([254] * 16 + [4] * 16, dtype=np.uint8)
        records = [record(range(16), 254), record(range(16, 32), 4)]
        for algorithm in ("sor", "dbscan", "sor_dbscan"):
            config = dict(dbscan_config(), algorithm=algorithm, sor_k=3)
            with self.subTest(algorithm=algorithm):
                result = clean_point_support(xyz, prediction, records, config,
                                             surface_class_ids=(254,))
                expected = filter_instance_points(xyz, range(16, 32), config)
                np.testing.assert_array_equal(result["prediction"][:16], [254] * 16)
                self.assertEqual(result["instances"][0]["point_indices"], list(range(16)))
                self.assertEqual(result["instances"][1]["point_indices"],
                                 expected["kept_indices"].tolist())
                np.testing.assert_array_equal(result["removed_indices"], expected["removed_indices"])
                self.assertGreater(len(result["removed_indices"]), 0)
                self.assertEqual(result["prediction"].shape, prediction.shape)
                self.assertEqual(result["keep_mask"].shape, prediction.shape)
                self.assertEqual(result["metadata"]["exempt_surface_points"], 16)
                self.assertEqual(result["metadata"]["exempt_surface_instances"], 1)
                self.assertIn("without instance coverage", result["metadata"]["ownership"])
                self.assertEqual(result["metadata"]["per_class"][-1],
                                 {"class_id": 254, "input": 16, "kept": 16, "removed": 0})

    def test_uncovered_surface_predictions_retained_and_non_surface_coverage_required(self):
        xyz = points([0, 100, 200, 300, 400])
        prediction = np.array([254, 254, 4, 4, 0], dtype=np.uint8)
        records = [record([0], 254), record([0], 254), record([2, 3], 4)]
        result = clean_point_support(xyz, prediction, records, dbscan_config(),
                                     surface_class_ids=(254,))
        self.assertEqual(result["prediction"].tolist(), [254, 254, 0, 0, 0])
        self.assertEqual(result["keep_mask"].tolist(), [True, True, False, False, True])
        self.assertEqual(result["metadata"]["surface_class_ids"], [254])
        self.assertEqual(result["metadata"]["exempt_surface_points"], 2)
        self.assertEqual(result["metadata"]["instance_stats"]["exempt_surface_points"], 2)
        self.assertEqual(result["metadata"]["exempt_surface_instances"], 2)
        self.assertEqual(result["metadata"]["removed"], 2)
        with self.assertRaisesRegex(ValueError, "1 nonzero prediction points"):
            clean_point_support(xyz, prediction, [record([2], 4)], dbscan_config(),
                                surface_class_ids=(254,))
        only_surfaces = clean_point_support(xyz[:2], prediction[:2], [], dbscan_config(),
                                            surface_class_ids=(254,))
        np.testing.assert_array_equal(only_surfaces["prediction"], prediction[:2])
        self.assertEqual(only_surfaces["metadata"]["exempt_surface_instances"], 0)

    def test_surface_observation_stats_distinct_from_prediction_point_stats(self):
        xyz = points([0, 100, 200])
        prediction = np.array([254, 254, 254], dtype=np.uint8)
        records = [record([0, 1], 254), record([0, 1], 254)]
        result = clean_point_support(xyz, prediction, records, dbscan_config(),
                                     surface_class_ids=(254,))
        self.assertEqual(result["metadata"]["exempt_surface_points"], 3)
        self.assertEqual(result["metadata"]["instance_stats"]["exempt_surface_points"], 4)
        self.assertEqual(result["metadata"]["instance_stats"]["per_class"], [
            {"class_id": 254, "instances": 2, "input": 4, "kept": 4, "removed": 0}])
        self.assertTrue(result["keep_mask"].all())

    def test_dynamic_non_surface_class_254_is_still_filtered(self):
        xyz = points([0, 100, 200])
        prediction = np.array([1, 254, 0], dtype=np.uint8)
        result = clean_point_support(xyz, prediction, [record([1], 254)], dbscan_config(),
                                     surface_class_ids=(1,))
        self.assertEqual(result["prediction"].tolist(), [1, 0, 0])
        self.assertEqual(result["removed_indices"].tolist(), [1])
        self.assertEqual(result["instances"][0]["filter"]["status"], "filtered")
        self.assertEqual(result["metadata"]["exempt_surface_points"], 1)
        self.assertEqual(result["metadata"]["exempt_surface_instances"], 0)

    def test_surface_ids_validated_for_both_public_entry_points(self):
        xyz = points([0])
        prediction = np.array([254], dtype=np.uint8)
        invalid = (None, 254, True, "254", {254: True}, (0,), (-1,), (255,),
                   (256,), (True,), (np.bool_(False),), (2.0,), ("2",), ([2],))
        for ids in invalid:
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "surface_class_ids"):
                    clean_instances(xyz, [], surface_class_ids=ids)
                with self.assertRaisesRegex(ValueError, "surface_class_ids"):
                    clean_point_support(xyz, prediction, [], surface_class_ids=ids)
        result = clean_instances(xyz, [record([0], 254)],
                                 surface_class_ids=np.array([254, 1, 254], dtype=np.int64))
        self.assertEqual(result["stats"]["surface_class_ids"], [1, 254])

    def test_surface_exemptions_do_not_mutate_inputs_or_share_nested_metadata(self):
        xyz = points([0, 100, 200])
        prediction = np.array([254, 254, 0], dtype=np.uint8)
        records = [record([1, 0, 1], 254, extra={"array": np.array([1, 2])})]
        config = dbscan_config()
        surface_ids = [254, 2, 254]
        before = copy.deepcopy((xyz, prediction, records, config, surface_ids))
        result = clean_point_support(xyz, prediction, records, config,
                                     surface_class_ids=surface_ids)
        self.assertEqual(result["instances"][0]["point_indices"], [0, 1])
        result["instances"][0]["extra"]["array"][0] = 99
        result["instances"][0]["filter"]["config"]["algorithm"] = "none"
        result["metadata"]["surface_class_ids"].append(9)
        result["prediction"][0] = 0
        np.testing.assert_array_equal(xyz, before[0])
        np.testing.assert_array_equal(prediction, before[1])
        self.assertEqual(records[0]["point_indices"], before[2][0]["point_indices"])
        np.testing.assert_array_equal(records[0]["extra"]["array"], before[2][0]["extra"]["array"])
        self.assertEqual(config, before[3])
        self.assertEqual(surface_ids, before[4])
        self.assertEqual(result["metadata"]["per_instance"][0]["config"]["algorithm"], "dbscan")
        self.assertEqual(result["metadata"]["instance_stats"]["surface_class_ids"], [2, 254])
        json.dumps(result["metadata"], allow_nan=False)

    def test_prediction_and_record_validation(self):
        xyz = points([0, 1])
        for prediction in (np.zeros(2, dtype=int), np.zeros(1, dtype=np.uint8),
                           np.zeros((2, 1), dtype=np.uint8)):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                clean_point_support(xyz, prediction, [])
        for records in (None, {}, [None], [{}], [record([], 0)], [record([], 256)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                clean_instances(xyz, records)


if __name__ == "__main__":
    unittest.main()
