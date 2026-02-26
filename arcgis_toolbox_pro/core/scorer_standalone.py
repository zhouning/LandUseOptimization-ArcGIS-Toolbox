# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class ScorerNetwork:

    def __init__(self, weights_path):
        cp = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.k_parcel = cp.get('k_parcel', 6)
        self.k_global = cp.get('k_global', 8)
        _h = cp.get('scorer_hiddens', [128, 64])
        _layers = []
        _d = self.k_parcel + self.k_global
        for _s in _h:
            _layers.append(nn.Linear(_d, _s))
            _layers.append(nn.Tanh())
            _d = _s
        _layers.append(nn.Linear(_d, 1))
        self.scorer_net = nn.Sequential(*_layers)
        self.scorer_net.load_state_dict(cp['scorer_net'])
        self.scorer_net.eval()

    def score_parcels(self, per_parcel_features, global_features):
        with torch.no_grad():
            _n = per_parcel_features.shape[0]
            _p = torch.tensor(per_parcel_features, dtype=torch.float32)
            _g = torch.tensor(global_features, dtype=torch.float32)
            _ge = _g.unsqueeze(0).expand(_n, -1)
            _c = torch.cat([_p, _ge], dim=1)
            _l = self.scorer_net(_c).squeeze(-1)
            return _l.numpy()

    def select_action(self, per_parcel_features, global_features, action_mask):
        _l = self.score_parcels(per_parcel_features, global_features)
        _l[~action_mask] = -np.inf
        return int(np.argmax(_l))
