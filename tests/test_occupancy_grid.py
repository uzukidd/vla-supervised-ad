"""Sparse occupancy grid voxelization, fusion, validation, and persistence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tasks.nuscenes_sam3.python.occupancy_grid import (
    build_occupancy_grid,
    flatten_surface_occupancy,
    save_flat_occupancy,
    save_occupancy_grid,
    voxelize_points,
)


class OccupancyGridTests(unittest.TestCase):
    def test_voxelize_uses_floor_and_counts_collisions(self):
        points = np.array([
            [-.01, 0., 0.], [-.5, 0., 0.], [-.51, 0., 0.],
            [0., 0., 0.], [.49, 0., 0.], [1., 0., 0.],
        ])
        indices, counts = voxelize_points(points, .5)
        np.testing.assert_array_equal(indices, [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(counts, [1, 2, 2, 1])
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(counts.dtype, np.uint64)

    def test_fused_counts_account_for_every_point_and_empty_frames(self):
        frames = [
            np.array([[-.1, 0, 0], [0., 0, 0], [0.1, 0, 0]]),
            np.array([[0.2, 0, 0], [2., 0, 0]]),
            np.empty((0, 3)),
        ]
        payload = build_occupancy_grid(frames, .5)
        self.assertEqual(payload["origin"], [0, 0, 0])
        self.assertEqual([frame["point_count"] for frame in payload["frames"]], [3, 2, 0])
        self.assertEqual(payload["fused"]["indices"], [[-1, 0, 0], [0, 0, 0], [4, 0, 0]])
        self.assertEqual(payload["fused"]["point_counts"], [1, 3, 1])
        self.assertEqual(sum(payload["fused"]["point_counts"]), 5)
        self.assertTrue(payload["metadata"]["occupied_only"])
        self.assertFalse(payload["metadata"]["free_space_raytracing"])
        json.dumps(payload)

    def test_empty_scene_is_valid(self):
        payload = build_occupancy_grid([], .5)
        self.assertEqual(payload["fused"]["indices"], [])
        self.assertEqual(payload["fused"]["point_counts"], [])
        self.assertEqual(payload["metadata"]["point_count"], 0)

    def test_semantic_voxels_exclude_unassigned_and_use_point_majority(self):
        xyz = np.array([[.1, 0, 0], [.2, 0, 0], [.3, 0, 0], [.4, 0, 0],
                        [1.1, 0, 0], [1.2, 0, 0]])
        labels = np.array([2, 1, 2, 1, 0, 254], dtype=np.uint8)
        payload = build_occupancy_grid([(xyz, labels)], .5, semantic=True)
        frame = payload["frames"][0]
        self.assertEqual(frame["indices"], [[0, 0, 0], [2, 0, 0]])
        self.assertEqual(frame["point_counts"], [4, 1])
        self.assertEqual(frame["class_ids"], [1, 254])
        self.assertEqual(payload["metadata"]["semantic_source"], "aligned_frame_labels_class_ids_1_254")
        self.assertEqual(payload["metadata"]["fusion_vote_source"], "raw_semantic_point_counts")

    def test_semantic_fusion_aggregates_raw_class_point_counts(self):
        frame1_xyz = np.zeros((5, 3))
        frame1_labels = np.array([1, 1, 1, 2, 2], dtype=np.uint8)
        frame2_xyz = np.zeros((2, 3))
        frame2_labels = np.array([2, 2], dtype=np.uint8)
        payload = build_occupancy_grid([(frame1_xyz, frame1_labels), (frame2_xyz, frame2_labels)], semantic=True)
        self.assertEqual([frame["class_ids"] for frame in payload["frames"]], [[1], [2]])
        self.assertEqual(payload["fused"]["class_ids"], [2])
        self.assertEqual(payload["fused"]["point_counts"], [7])
        self.assertEqual(payload["fused"]["vote_counts"], [[0, 1, 3], [0, 2, 4]])

    def test_semantic_empty_inputs_and_validation(self):
        empty = build_occupancy_grid([(np.empty((0, 3)), np.empty((0,), dtype=np.uint8))], semantic=True)
        self.assertEqual(empty["frames"][0]["indices"], [])
        self.assertEqual(empty["frames"][0]["class_ids"], [])
        self.assertEqual(empty["fused"]["vote_counts"], [])
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_occupancy_grid([(np.zeros((2, 3)), np.array([1]))], semantic=True)
        for labels in (np.array([1., 2.]), np.array([-1, 2]), np.array([1, 255])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                build_occupancy_grid([(np.zeros((2, 3)), labels)], semantic=True)
        with self.assertRaisesRegex(ValueError, "pairs"):
            build_occupancy_grid([np.zeros((1, 3))], semantic=True)

    def test_semantic_save_persists_class_ids_and_sparse_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            xyz = np.zeros((3, 3))
            payload = save_occupancy_grid(directory, [(xyz, np.array([1, 2, 2], dtype=np.uint8))], semantic=True)
            with np.load(Path(directory) / "occupancy_grid.npz", allow_pickle=False) as saved:
                self.assertIn("fused_class_ids", saved.files)
                self.assertIn("frame_0_class_ids", saved.files)
                self.assertEqual(saved["fused_class_ids"].dtype, np.uint8)
                np.testing.assert_array_equal(saved["fused_vote_counts"], [[0, 1, 1], [0, 2, 2]])
                np.testing.assert_array_equal(saved["frame_0_class_ids"], [2])
            self.assertEqual(payload["fused"]["class_ids"], [2])
            metadata = json.loads((Path(directory) / "occupancy_grid.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["metadata"]["semantic_filtering"], True)

    def test_named_unassigned_layer_preserves_independent_overlapping_voxels(self):
        frames = [
            (np.array([[.1, 0, 0], [.2, 0, 0], [1.1, 0, 0]]), np.array([0, 2, 0], dtype=np.uint8)),
            (np.array([[.3, 0, 0], [1.2, 0, 0]]), np.array([0, 1], dtype=np.uint8)),
            (np.empty((0, 3)), np.empty(0, dtype=np.uint8)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = save_occupancy_grid(root, iter(frames), .25, semantic=True)
            semantic_bytes = (root / "occupancy_grid.npz").read_bytes()
            unknown = save_occupancy_grid(root, (xyz[labels == 0] for xyz, labels in frames), .25,
                basename="unassigned_occupancy_grid",
                metadata={"semantic_source": "aligned_frame_filtered_labels_class_id_0", "source_class_ids": [0]})
            self.assertEqual((root / "occupancy_grid.npz").read_bytes(), semantic_bytes)
            self.assertEqual(unknown["resolution"], semantic["resolution"])
            self.assertEqual(unknown["origin"], semantic["origin"])
            self.assertEqual(unknown["frames"][0]["indices"], [[0, 0, 0], [4, 0, 0]])
            self.assertEqual(semantic["frames"][0]["indices"], [[0, 0, 0]])
            self.assertEqual(unknown["fused"]["point_count"], 3)
            self.assertEqual(unknown["frames"][2]["point_count"], 0)
            for (xyz, _), known, unassigned in zip(frames, semantic["frames"], unknown["frames"]):
                self.assertEqual(known["point_count"] + unassigned["point_count"], len(xyz))
                self.assertNotIn("class_ids", unassigned)
                self.assertNotIn("vote_counts", unassigned)
            self.assertEqual(semantic["fused"]["point_count"] + unknown["fused"]["point_count"], 5)
            with np.load(root / "unassigned_occupancy_grid.npz", allow_pickle=False) as saved:
                self.assertFalse(any("class_ids" in key or "vote_counts" in key for key in saved.files))
                np.testing.assert_array_equal(saved["frame_0_indices"], [[0, 0, 0], [4, 0, 0]])
                self.assertEqual(saved["frame_2_indices"].shape, (0, 3))
            metadata = json.loads((root / "unassigned_occupancy_grid.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["metadata"]["semantic_source"], "aligned_frame_filtered_labels_class_id_0")
            self.assertEqual(metadata["metadata"]["point_count"], 3)
            self.assertNotIn("class_ids", metadata["fused"])

    def test_all_unknown_and_all_semantic_layers_can_be_empty_independently(self):
        xyz = np.array([[.1, 0, 0], [1.1, 0, 0]])
        for class_id in (0, 2):
            labels = np.full(2, class_id, dtype=np.uint8)
            known = build_occupancy_grid([(xyz, labels)], semantic=True)
            unknown = build_occupancy_grid([xyz[labels == 0]])
            self.assertEqual(known["fused"]["point_count"], 0 if class_id == 0 else 2)
            self.assertEqual(unknown["fused"]["point_count"], 2 if class_id == 0 else 0)
            self.assertNotIn("class_ids", unknown["fused"])

    def test_named_save_rejects_paths_as_basenames(self):
        with tempfile.TemporaryDirectory() as directory:
            for basename in ("", ".", "..", "../outside", "nested/name", "nested\\name", None):
                with self.subTest(basename=basename), self.assertRaisesRegex(ValueError, "basename"):
                    save_occupancy_grid(directory, [], basename=basename)

    def test_rejects_invalid_resolution_and_point_clouds(self):
        for resolution in (0, -1, np.inf, np.nan, True):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                voxelize_points(np.empty((0, 3)), resolution)
        for points in (np.zeros((3, 2)), np.array([[0., np.nan, 0.]]), np.array([[np.inf, 0., 0.]])):
            with self.subTest(points=points), self.assertRaises(ValueError):
                voxelize_points(points)

    def test_saves_compact_json_and_numeric_npz_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = save_occupancy_grid(directory, [np.array([[0., 0., 0.], [0.1, 0, 0]]), np.empty((0, 3))])
            root = Path(directory)
            self.assertEqual({path.name for path in root.iterdir()}, {"occupancy_grid.npz", "occupancy_grid.json"})
            metadata = json.loads((root / "occupancy_grid.json").read_text(encoding="utf-8"))
            self.assertNotIn("indices", metadata["fused"])
            with np.load(root / "occupancy_grid.npz", allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {
                    "fused_indices", "fused_point_counts", "frame_0_indices", "frame_0_counts",
                    "frame_1_indices", "frame_1_counts",
                })
                np.testing.assert_array_equal(saved["fused_indices"], [[0, 0, 0]])
                np.testing.assert_array_equal(saved["fused_point_counts"], [2])
                self.assertEqual(saved["frame_0_indices"].shape, (1, 3))
                self.assertEqual(saved["frame_1_indices"].shape, (0, 3))
            self.assertEqual(payload["fused"]["point_count"], 2)


class FlatOccupancyTests(unittest.TestCase):
    @staticmethod
    def semantic_frames(frames, resolution=.5):
        """Build full point clouds from (XYZ, class, count) synthetic supports."""
        clouds = []
        for frame in frames:
            points, labels = [], []
            for xyz, class_id, count in frame:
                points.extend([xyz] * count)
                labels.extend([class_id] * count)
            clouds.append((np.asarray(points, dtype=float).reshape((-1, 3)),
                           np.asarray(labels, dtype=np.uint8)))
        return build_occupancy_grid(clouds, resolution, semantic=True)

    def test_surface_winners_only_support_counts_negative_xy_and_multiple_z(self):
        occupancy = self.semantic_frames([[
            ((-.1, -.1, -1), 19, 5), ((-.1, -.1, -1), 23, 4),
            ((-.1, -.1, 0), 23, 6),
            ((-.1, -.1, 1), 7, 9), ((-.1, -.1, 1), 19, 8),
            ((1., 0., 0.), 7, 3), ((1., 0., 0.), 23, 2),
            ((1., 2., 0.), 254, 1),
        ]])
        occupancy["origin"] = [10., -4., 8.]
        flat = flatten_surface_occupancy(occupancy, (254, 23, 19), plane_z_scene=-2.)
        self.assertEqual(set(flat), {"resolution", "origin", "plane_z_scene", "frames", "fused", "metadata"})
        self.assertEqual(flat["origin"], [10., -4.])
        self.assertEqual(flat["plane_z_scene"], -2.)
        self.assertEqual(flat["resolution"], .5)
        expected = {"indices": [[-1, -1], [2, 4]], "class_ids": [23, 254],
                    "point_counts": [11, 1], "cell_count": 2, "point_count": 12}
        self.assertEqual(flat["frames"][0], expected)
        self.assertEqual(flat["fused"], expected)
        self.assertEqual(flat["metadata"]["surface_class_ids"], [19, 23, 254])
        json.dumps(flat, allow_nan=False)

    def test_ties_choose_smaller_id_after_summing_support_across_z(self):
        occupancy = self.semantic_frames([[
            ((0, 0, -1), 23, 2), ((0, 0, 0), 23, 3), ((0, 0, 1), 19, 5),
        ]])
        flat = flatten_surface_occupancy(occupancy, (23, 19))
        self.assertEqual(flat["fused"]["class_ids"], [19])
        self.assertEqual(flat["fused"]["point_counts"], [10])

    def test_fused_source_is_authoritative_full_3d_occupancy(self):
        occupancy = self.semantic_frames([
            [((0, 0, 0), 19, 3), ((0, 0, 0), 23, 2),
             ((3, 0, 0), 19, 3), ((3, 0, 0), 7, 2)],
            [((0, 0, 0), 23, 2), ((3, 0, 0), 7, 2)],
        ])
        flat = flatten_surface_occupancy(occupancy, (19, 23))
        self.assertEqual(flat["frames"][0]["class_ids"], [19, 19])
        self.assertEqual(flat["frames"][1]["class_ids"], [23])
        # Re-voting frame winners would choose 19 at both cells. Full fused
        # 3D votes instead choose 23 at the first and a non-surface at the second.
        self.assertEqual(flat["fused"], {"indices": [[0, 0]], "class_ids": [23],
                                        "point_counts": [4], "cell_count": 1, "point_count": 4})

    def test_legacy_records_fall_back_only_when_vote_counts_absent(self):
        occupancy = self.semantic_frames([[
            ((0, 0, -1), 19, 3), ((0, 0, -1), 7, 2), ((0, 0, 0), 23, 4),
        ]])
        for record in [*occupancy["frames"], occupancy["fused"]]:
            del record["vote_counts"]
        flat = flatten_surface_occupancy(occupancy, (19, 23))
        self.assertEqual(flat["fused"]["point_counts"], [9])
        self.assertEqual(flat["fused"]["class_ids"], [19])

    def test_empty_frames_and_all_excluded_are_valid(self):
        for occupancy in (self.semantic_frames([]), self.semantic_frames([[]]),
                          self.semantic_frames([[((0, 0, 0), 7, 1)]])):
            flat = flatten_surface_occupancy(occupancy, (19,))
            self.assertEqual(flat["fused"], {"indices": [], "class_ids": [], "point_counts": [],
                                            "cell_count": 0, "point_count": 0})
            with tempfile.TemporaryDirectory() as directory:
                save_flat_occupancy(directory, flat)
                with np.load(Path(directory) / "occupancy_flat.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["fused_indices"].shape, (0, 2))
                    self.assertEqual(saved["fused_class_ids"].dtype, np.uint8)

    def test_validate_ids_and_calibration_metadata(self):
        occupancy = self.semantic_frames([[((0, 0, 0), 19, 1)]])
        for ids in (None, (), (0,), (255,), (-1,), (19, 19), (True,), (19.,), "19"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                flatten_surface_occupancy(occupancy, ids)
        for z in (np.nan, np.inf, True, None):
            with self.subTest(z=z), self.assertRaises(ValueError):
                flatten_surface_occupancy(occupancy, (19,), z)
        for key, value in (("resolution", 0), ("resolution", np.inf), ("resolution", True),
                           ("origin", [0, 0]), ("origin", [0, np.nan, 0]),
                           ("origin", [True, False, False]), ("metadata", None),
                           ("metadata", {"coordinate_frame": ""})):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                flatten_surface_occupancy({**occupancy, key: value}, (19,))

    def test_validate_semantic_records_and_sparse_votes(self):
        occupancy = self.semantic_frames([[((0, 0, 0), 19, 3), ((0, 0, 0), 7, 2)]])
        for key, value in (("indices", [[0., 0., 0.]]), ("indices", [[0, 0]]),
                           ("class_ids", [255]), ("class_ids", [19.]), ("point_counts", [-1]),
                           ("vote_counts", []), ("vote_counts", [[1, 19, 5]]),
                           ("vote_counts", [[0, 19, 3], [0, 19, 2]])):
            fused = {**occupancy["fused"], key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                flatten_surface_occupancy({**occupancy, "fused": fused}, (19,))
        with self.assertRaisesRegex(ValueError, "class_ids"):
            flatten_surface_occupancy(build_occupancy_grid([np.zeros((1, 3))]), (19,))

    def test_save_sorted_numeric_arrays_and_deterministic_metadata(self):
        occupancy = self.semantic_frames([[
            ((-1, -1, 0), 19, 2), ((1, 0, 0), 23, 3),
        ], []])
        flat = flatten_surface_occupancy(occupancy, (19, 23), 1.25)
        for record in (flat["frames"][0], flat["fused"]):
            for key in ("indices", "class_ids", "point_counts"):
                record[key].reverse()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIs(save_flat_occupancy(root, flat), flat)
            first_json = (root / "occupancy_flat.json").read_bytes()
            first_npz = (root / "occupancy_flat.npz").read_bytes()
            save_flat_occupancy(root, flat)
            self.assertEqual((root / "occupancy_flat.json").read_bytes(), first_json)
            self.assertEqual((root / "occupancy_flat.npz").read_bytes(), first_npz)
            self.assertEqual({path.name for path in root.iterdir()}, {"occupancy_flat.json", "occupancy_flat.npz"})
            metadata = json.loads(first_json)
            self.assertEqual(metadata["plane_z_scene"], 1.25)
            self.assertEqual(metadata["fused"], {"cell_count": 2, "point_count": 5})
            self.assertEqual(metadata["frames"][1], {"cell_count": 0, "point_count": 0})
            with np.load(root / "occupancy_flat.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["fused_indices"].dtype, np.int64)
                self.assertEqual(saved["fused_class_ids"].dtype, np.uint8)
                self.assertEqual(saved["fused_point_counts"].dtype, np.uint64)
                np.testing.assert_array_equal(saved["fused_indices"], [[-2, -2], [2, 0]])
                np.testing.assert_array_equal(saved["fused_class_ids"], [19, 23])
                np.testing.assert_array_equal(saved["fused_point_counts"], [2, 3])
                self.assertEqual(saved["frame_1_indices"].shape, (0, 2))
                for key in saved.files:
                    self.assertNotEqual(saved[key].dtype, object)


if __name__ == "__main__":
    unittest.main()
