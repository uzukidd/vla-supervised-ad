"""Shared SAM3 helper and checkpoint loading tests."""
from pathlib import Path
import tempfile
import unittest

import torch

from shared.python.sam3 import _load_local_sam3_checkpoint


class Sam3CheckpointTests(unittest.TestCase):
    def test_local_checkpoint_uses_mmap_assign_and_detector_key_mapping(self):
        expected_weight=torch.arange(12,dtype=torch.float32).reshape(3,4)
        expected_bias=torch.tensor([.25,.5,.75])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'sam3.pt'
            torch.save({'model':{'detector.projection.weight':expected_weight,
                                 'detector.projection.bias':expected_bias,
                                 'unrelated.weight':torch.ones(2)}},checkpoint)
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.projection=torch.nn.Linear(4,3);self.inst_interactive_predictor=None
            model=Model()
            _load_local_sam3_checkpoint(model,checkpoint,torch)
            torch.testing.assert_close(model.projection.weight,expected_weight)
            torch.testing.assert_close(model.projection.bias,expected_bias)
            self.assertTrue(model.projection.weight.is_contiguous())

    def test_local_checkpoint_loader_rejects_missing_file(self):
        with self.assertRaises((FileNotFoundError,RuntimeError)):
            _load_local_sam3_checkpoint(torch.nn.Linear(2,2),Path('/missing/sam3.pt'),torch)


if __name__=='__main__':unittest.main()
