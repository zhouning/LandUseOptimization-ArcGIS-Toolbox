# -*- coding: utf-8 -*-
import numpy as np


def build_adjacency(input_fc, n_parcels, oid_to_idx, messages=None):
    import arcpy
    if _chk_lic():
        return _build_pn(input_fc, n_parcels, oid_to_idx, messages)
    else:
        return _build_geom(input_fc, n_parcels, oid_to_idx, messages)


def _chk_lic():
    import arcpy
    return arcpy.ProductInfo() in ('ArcInfo',) or arcpy.CheckProduct('ArcInfo') == 'Available'


def _build_pn(input_fc, n_parcels, oid_to_idx, messages=None):
    import arcpy
    if messages:
        messages.addMessage("Building adjacency graph (PolygonNeighbors)...")
    desc = arcpy.Describe(input_fc)
    _of = desc.OIDFieldName
    _nt = r"in_memory\nbr_table"
    if arcpy.Exists(_nt):
        arcpy.management.Delete(_nt)
    arcpy.analysis.PolygonNeighbors(
        in_features=input_fc, out_table=_nt, in_fields=_of,
        area_overlap="NO_AREA_OVERLAP", both_sides="BOTH_SIDES",
        out_linear_units="METERS")
    _sc = f"src_{_of}"
    _nc = f"nbr_{_of}"
    adj = [[] for _ in range(n_parcels)]
    with arcpy.da.SearchCursor(_nt, [_sc, _nc]) as cur:
        for _s, _n in cur:
            _si = oid_to_idx.get(_s)
            _ni = oid_to_idx.get(_n)
            if _si is not None and _ni is not None:
                adj[_si].append(_ni)
    arcpy.management.Delete(_nt)
    adj = [np.array(a, dtype=np.intp) for a in adj]
    _av = np.mean([len(a) for a in adj])
    if messages:
        messages.addMessage(f"  Adjacency built: avg {_av:.1f} neighbors/parcel")
    return adj


def _build_geom(input_fc, n_parcels, oid_to_idx, messages=None):
    import arcpy
    if messages:
        messages.addMessage("Building adjacency graph (geometry fallback, may be slow)...")
    _gm = {}
    with arcpy.da.SearchCursor(input_fc, ["OID@", "SHAPE@"]) as cur:
        for _o, _s in cur:
            _ix = oid_to_idx.get(_o)
            if _ix is not None:
                _gm[_ix] = _s
    adj = [[] for _ in range(n_parcels)]
    _ids = sorted(_gm.keys())
    for _p, i in enumerate(_ids):
        if messages and _p % 500 == 0:
            messages.addMessage(f"  Processing parcel {_p}/{len(_ids)}...")
        _gi = _gm[i]
        for j in _ids[_p + 1:]:
            if not _gi.disjoint(_gm[j]):
                adj[i].append(j)
                adj[j].append(i)
    adj = [np.array(a, dtype=np.intp) for a in adj]
    _av = np.mean([len(a) for a in adj])
    if messages:
        messages.addMessage(f"  Adjacency built: avg {_av:.1f} neighbors/parcel")
    return adj
