# -*- coding: utf-8 -*-
"""
Convert model weights from PyTorch (.pt) to NumPy (.npz) format.

Run under Python 3 + PyTorch (e.g. ArcGIS Pro Python environment).
The output .npz file is used by the ArcMap version of the toolbox.

Usage:
    python convert_weights_to_npz.py [input.pt] [output.npz]
"""

import os
import sys
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Run in ArcGIS Pro Python environment.")
    sys.exit(1)


def convert(pt_path, npz_path):
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['scorer_net']

    arrays = {}
    for key in ('k_parcel', 'k_global'):
        if key in checkpoint:
            arrays[key] = np.array(checkpoint[key])

    weight_keys = sorted([k for k in state_dict.keys() if k.endswith('.weight')])
    arrays['n_layers'] = np.array(len(weight_keys))

    for i, wk in enumerate(weight_keys):
        bk = wk.replace('.weight', '.bias')
        arrays['weight_%d' % i] = state_dict[wk].numpy()
        arrays['bias_%d' % i] = state_dict[bk].numpy()

    np.savez(npz_path, **arrays)
    print("Converted: %s -> %s (%.1f KB)" % (
        pt_path, npz_path, os.path.getsize(npz_path) / 1024.0))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        script_dir, 'arcgis_toolbox_pro', 'models', 'scorer_weights_v7.pt')
    npz_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        script_dir, 'arcgis_toolbox_arcmap', 'models', 'scorer_weights_v7.npz')

    if not os.path.exists(pt_path):
        print("ERROR: Source not found: %s" % pt_path)
        sys.exit(1)

    convert(pt_path, npz_path)
