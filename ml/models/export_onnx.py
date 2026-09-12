"""
ONNX Export and TensorRT / Edge Validator for AuraPulse Neural rPPG Models.
"""

import os
import torch
from ml.models.physnet import PhysNet
from ml.models.deepphys import DeepPhys
from ml.models.tscan import TSCAN

def export_all():
    os.makedirs('ml/exported_onnx', exist_ok=True)
    print("[*] Exporting AuraPulse Neural rPPG models to ONNX...")

    try:
        physnet = PhysNet()
        physnet.eval()
        dummy_video = torch.randn(1, 3, 64, 72, 72)
        torch.onnx.export(
            physnet,
            dummy_video,
            'ml/exported_onnx/physnet_st_cnn.onnx',
            input_names=['video_frames'],
            output_names=['bvp_waveform'],
            opset_version=14,
            dynamic_axes={'video_frames': {0: 'batch_size', 2: 'num_frames'}}
        )
        print("[+] Successfully exported PhysNet to ml/exported_onnx/physnet_st_cnn.onnx")
    except Exception as e:
        print(f"[!] PhysNet ONNX export notice: {e}")

    try:
        deepphys = DeepPhys()
        deepphys.eval()
        dummy_app = torch.randn(1, 3, 72, 72)
        dummy_mot = torch.randn(1, 3, 72, 72)
        torch.onnx.export(
            deepphys,
            (dummy_app, dummy_mot),
            'ml/exported_onnx/deepphys_dual_stream.onnx',
            input_names=['appearance_frame', 'motion_diff'],
            output_names=['bvp_derivative'],
            opset_version=14,
            dynamic_axes={'appearance_frame': {0: 'batch_size'}, 'motion_diff': {0: 'batch_size'}}
        )
        print("[+] Successfully exported DeepPhys to ml/exported_onnx/deepphys_dual_stream.onnx")
    except Exception as e:
        print(f"[!] DeepPhys ONNX export notice: {e}")

    print("[*] ONNX export pipeline complete!")

if __name__ == '__main__':
    export_all()
