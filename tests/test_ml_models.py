import pytest
import numpy as np

def test_ml_models_import_and_forward():
    try:
        import torch
        from ml.models.physnet import PhysNet
        from ml.models.deepphys import DeepPhys
        from ml.models.tscan import TSCAN

        # Test PhysNet
        physnet = PhysNet()
        physnet.eval()
        dummy_video = torch.randn(2, 3, 32, 36, 36)
        with torch.no_grad():
            out_physnet = physnet(dummy_video)
        assert out_physnet.shape == (2, 32), f"Unexpected PhysNet shape: {out_physnet.shape}"

        # Test DeepPhys
        deepphys = DeepPhys()
        deepphys.eval()
        dummy_app = torch.randn(2, 3, 36, 36)
        dummy_mot = torch.randn(2, 3, 36, 36)
        with torch.no_grad():
            out_deepphys = deepphys(dummy_app, dummy_mot)
        assert out_deepphys.shape == (2, 1), f"Unexpected DeepPhys shape: {out_deepphys.shape}"

        # Test TSCAN
        tscan = TSCAN(n_segment=4)
        tscan.eval()
        dummy_ts = torch.randn(8, 3, 36, 36) # 2 batches * 4 segments = 8
        with torch.no_grad():
            out_tscan = tscan(dummy_ts)
        assert out_tscan.shape == (8, 1), f"Unexpected TSCAN shape: {out_tscan.shape}"

    except ImportError:
        pytest.skip("PyTorch not installed in environment; skipping neural model forward tests.")
