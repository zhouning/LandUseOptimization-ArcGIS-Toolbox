# -*- coding: utf-8 -*-
import numpy as np

OTHER = 0
FARMLAND = 1
FOREST = 2


def read_feature_class(input_fc, dlmc_field, slope_field, farmland_types, forest_types, messages=None):
    import arcpy
    if messages:
        messages.addMessage("Reading input feature class...")
    desc = arcpy.Describe(input_fc)
    spatial_ref = desc.spatialReference
    _flds = ["OID@", dlmc_field, slope_field, "SHAPE@AREA"]
    _oids, _dlmc, _sl, _ar = [], [], [], []
    with arcpy.da.SearchCursor(input_fc, _flds) as cur:
        for r in cur:
            _oids.append(r[0])
            _dlmc.append(r[1] if r[1] else "")
            _sl.append(float(r[2]) if r[2] is not None else 0.0)
            _ar.append(float(r[3]) if r[3] is not None else 0.0)
    _n = len(_oids)
    slopes = np.array(_sl, dtype=np.float64)
    areas = np.array(_ar, dtype=np.float64)
    oid_to_idx = {o: i for i, o in enumerate(_oids)}
    idx_to_oid = {i: o for i, o in enumerate(_oids)}
    _fs = set(farmland_types)
    _rs = set(forest_types)
    _it = np.full(_n, OTHER, dtype=np.int8)
    for i, d in enumerate(_dlmc):
        if d in _fs:
            _it[i] = FARMLAND
        elif d in _rs:
            _it[i] = FOREST
    _nf = int((_it == FARMLAND).sum())
    _nr = int((_it == FOREST).sum())
    if messages:
        messages.addMessage(f"  Total parcels: {_n}")
        messages.addMessage(f"  Farmland: {_nf}, Forest: {_nr}, Other: {_n - _nf - _nr}")
    return {
        'slopes': slopes, 'areas': areas, 'initial_types': _it,
        'dlmc_values': _dlmc, 'oid_to_idx': oid_to_idx,
        'idx_to_oid': idx_to_oid, 'n_parcels': _n, 'spatial_ref': spatial_ref,
    }


def write_output_fc(input_fc, output_fc, initial_types, final_types,
                    dlmc_field, farmland_types, forest_types,
                    idx_to_oid, messages=None):
    import arcpy
    if messages:
        messages.addMessage("Writing output feature class...")
    arcpy.management.CopyFeatures(input_fc, output_fc)
    arcpy.management.AddField(output_fc, "OPT_DLMC", "TEXT", field_length=30)
    arcpy.management.AddField(output_fc, "OPT_TYPE", "SHORT")
    arcpy.management.AddField(output_fc, "CHG_FLAG", "SHORT")
    arcpy.management.AddField(output_fc, "ORIG_DLMC", "TEXT", field_length=30)
    _rf = farmland_types[0] if farmland_types else "farmland"
    _rr = forest_types[0] if forest_types else "forest"
    _o2i = {o: x for x, o in idx_to_oid.items()}
    _uf = ["OID@", dlmc_field, "OPT_DLMC", "OPT_TYPE", "CHG_FLAG", "ORIG_DLMC"]
    _c1, _c2 = 0, 0
    with arcpy.da.UpdateCursor(output_fc, _uf) as cur:
        for r in cur:
            _oid = r[0]
            _od = r[1] if r[1] else ""
            _ix = _o2i.get(_oid)
            if _ix is None:
                r[2], r[3], r[4], r[5] = _od, OTHER, 0, _od
            else:
                r[5] = _od
                r[3] = int(final_types[_ix])
                if initial_types[_ix] == FARMLAND and final_types[_ix] == FOREST:
                    r[2], r[4] = _rr, 1
                    _c1 += 1
                elif initial_types[_ix] == FOREST and final_types[_ix] == FARMLAND:
                    r[2], r[4] = _rf, 2
                    _c2 += 1
                else:
                    r[2], r[4] = _od, 0
            cur.updateRow(r)
    if messages:
        messages.addMessage(f"  Output written: {output_fc}")
        messages.addMessage(f"  Conversions: {_c1} farm->forest, {_c2} forest->farm")
        if _c1 == _c2:
            messages.addMessage(f"  Farmland count conserved (FC=0)")
        else:
            messages.addWarningMessage(
                f"  WARNING: Farmland count not conserved! "
                f"farm->forest={_c1}, forest->farm={_c2}")
