# -*- coding: utf-8 -*-
import numpy as np

OTHER = 0
FARMLAND = 1
FOREST = 2
_KP = 6
_KG = 8


class InferenceState:

    def __init__(self, slopes, areas, initial_types, adjacency, n_parcels):
        self.n_parcels = n_parcels
        self.slopes = slopes
        self.areas = areas
        self.adjacency = adjacency
        self._smin = float(slopes.min())
        self._smax = float(slopes.max())
        self._srng = self._smax - self._smin + 1e-8
        self._snorm = ((slopes - self._smin) / self._srng).astype(np.float32)
        _amin = float(areas.min())
        _amax = float(areas.max())
        _arng = _amax - _amin + 1e-8
        self._anorm = ((areas - _amin) / _arng).astype(np.float32)
        self._si = np.where((initial_types == FARMLAND) | (initial_types == FOREST))[0]
        self.n_swappable = len(self._si)
        self._tnc = np.array([len(adjacency[i]) for i in range(n_parcels)], dtype=np.float32)
        _idx = self._si
        self._f0 = self._snorm[_idx].copy()
        self._f4 = self._anorm[_idx].copy()
        _nas = np.zeros(n_parcels, dtype=np.float32)
        for i in range(n_parcels):
            _nb = adjacency[i]
            if len(_nb) > 0:
                _nas[i] = self._snorm[_nb].mean()
        self._f3 = _nas[_idx].copy()
        self.land_use = initial_types.copy()
        self.initial_types = initial_types.copy()
        self._recompute()
        self._i_slope = self._avg_s
        self._i_cont = self._cont
        self._i_nf = self._nf

    def _recompute(self):
        _fm = self.land_use == FARMLAND
        self._nf = int(_fm.sum())
        self._nr = int((self.land_use == FOREST).sum())
        self._tfs = float(self.slopes[_fm].sum())
        self._fnc = np.zeros(self.n_parcels, dtype=np.int32)
        for i in range(self.n_parcels):
            _nb = self.adjacency[i]
            if len(_nb) > 0:
                self._fnc[i] = int((self.land_use[_nb] == FARMLAND).sum())
        self._tfa = int(self._fnc[_fm].sum())

    @property
    def _avg_s(self):
        return self._tfs / max(self._nf, 1)

    @property
    def _cont(self):
        return self._tfa / max(self._nf, 1)

    @property
    def avg_farmland_slope(self):
        return self._avg_s

    @property
    def contiguity(self):
        return self._cont

    @property
    def n_farmland(self):
        return self._nf

    @property
    def n_forest(self):
        return self._nr

    @property
    def initial_avg_slope(self):
        return self._i_slope

    @property
    def initial_contiguity(self):
        return self._i_cont

    @property
    def initial_n_farmland(self):
        return self._i_nf

    @property
    def swappable_indices(self):
        return self._si

    def swap_to_forest(self, k):
        self._tfa -= self._fnc[k]
        self._tfs -= self.slopes[k]
        self.land_use[k] = FOREST
        self._nf -= 1
        self._nr += 1
        for j in self.adjacency[k]:
            self._fnc[j] -= 1
            if self.land_use[j] == FARMLAND:
                self._tfa -= 1

    def swap_to_farmland(self, k):
        self.land_use[k] = FARMLAND
        self._nf += 1
        self._nr -= 1
        self._tfs += self.slopes[k]
        self._tfa += self._fnc[k]
        for j in self.adjacency[k]:
            self._fnc[j] += 1
            if self.land_use[j] == FARMLAND:
                self._tfa += 1

    def get_obs(self, step_count, max_steps, phase):
        _idx = self._si
        _as = self._avg_s
        _f1 = (self.land_use[_idx] == FARMLAND).astype(np.float32)
        _f2 = (self._fnc[_idx].astype(np.float32) / np.maximum(self._tnc[_idx], 1.0))
        _f5 = ((self.slopes[_idx] - _as) / (abs(_as) + 1e-8)).astype(np.float32)
        _pp = np.column_stack([self._f0, _f1, _f2, self._f3, self._f4, _f5])
        _ct = self._cont
        _gf = np.array([
            (_as - self._smin) / self._srng,
            _ct / 10.0,
            float(phase),
            step_count / max_steps,
            self._nf / self.n_parcels,
            self._nr / self.n_parcels,
            (_as - self._i_slope) / (abs(self._i_slope) + 1e-8),
            (_ct - self._i_cont) / (abs(self._i_cont) + 1e-8),
        ], dtype=np.float32)
        return np.concatenate([_pp.ravel(), _gf])


def run_paired_inference(scorer, slopes, areas, initial_types, adjacency,
                         n_parcels, n_pairs=100, progress_callback=None):
    _st = InferenceState(slopes, areas, initial_types, adjacency, n_parcels)
    _si = _st.swappable_indices
    _ms = n_pairs * 2
    _sw = np.zeros(_st.n_swappable, dtype=bool)
    _sc = 0
    _cp = 0
    for _pi in range(n_pairs):
        _obs = _st.get_obs(_sc, _ms, phase=0)
        _fm = (_st.land_use[_si] == FARMLAND) & (~_sw)
        if not _fm.any():
            break
        _nsw = _st.n_swappable
        _pp = _obs[:_nsw * _KP].reshape(_nsw, _KP)
        _gf = _obs[_nsw * _KP:]
        _a = scorer.select_action(_pp, _gf, _fm)
        _st.swap_to_forest(_si[_a])
        _sw[_a] = True
        _sc += 1
        _obs = _st.get_obs(_sc, _ms, phase=1)
        _rm = (_st.land_use[_si] == FOREST) & (~_sw)
        if not _rm.any():
            break
        _pp = _obs[:_nsw * _KP].reshape(_nsw, _KP)
        _gf = _obs[_nsw * _KP:]
        _a = scorer.select_action(_pp, _gf, _rm)
        _st.swap_to_farmland(_si[_a])
        _sw[_a] = True
        _sc += 1
        _cp += 1
        if progress_callback:
            progress_callback(_pi + 1, n_pairs, _st.avg_farmland_slope, _st.contiguity)
    _dsl = _st.avg_farmland_slope - _st.initial_avg_slope
    _dct = _st.contiguity - _st.initial_contiguity
    _dfc = _st.n_farmland - _st.initial_n_farmland
    return {
        'final_types': _st.land_use.copy(),
        'slope_change': _dsl,
        'slope_change_pct': _dsl / _st.initial_avg_slope * 100,
        'cont_change': _dct,
        'farmland_change': _dfc,
        'completed_pairs': _cp,
        'initial_avg_slope': _st.initial_avg_slope,
        'initial_contiguity': _st.initial_contiguity,
        'final_avg_slope': _st.avg_farmland_slope,
        'final_contiguity': _st.contiguity,
    }
