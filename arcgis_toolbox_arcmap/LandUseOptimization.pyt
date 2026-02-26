# -*- coding: utf-8 -*-
"""
ArcMap Python Toolbox: Farmland Layout Optimization using DRL.

Python 2.7 compatible version for ArcMap 10.x.
Uses a trained Maskable PPO scorer network (pure NumPy, no PyTorch)
to perform paired farmland-forest swaps that reduce average farmland
slope while improving spatial contiguity, with guaranteed farmland
count conservation.
"""

import os
import sys
import traceback


class Toolbox(object):
    def __init__(self):
        self.label = u"Farmland Layout Optimization (DRL)"
        self.alias = u"LandUseOpt"
        self.description = (
            u"Farmland layout optimization toolbox based on Deep Reinforcement Learning. "
            u"Uses a trained Maskable PPO scorer network (pure NumPy) to perform paired "
            u"farmland-forest land use swaps, reducing average farmland slope while improving "
            u"spatial contiguity, with guaranteed farmland count conservation (FC=0)."
        )
        self.tools = [OptimizeLandUseTool, CheckDependenciesTool]


# ======================================================================
# Tool 1: Check Dependencies
# ======================================================================

class CheckDependenciesTool(object):
    def __init__(self):
        self.label = u"Check Dependencies"
        self.description = (
            u"Verify that NumPy and arcpy are available in the current "
            u"ArcMap Python environment, and check model weights file.\n\n"
            u"This ArcMap version does NOT require PyTorch. The DRL model "
            u"inference uses pure NumPy with weights in .npz format.\n\n"
            u"It is recommended to run this tool before your first optimization."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        return []

    def isLicensed(self):
        return True

    def execute(self, parameters, messages):
        messages.addMessage("=== Dependency Check (ArcMap) ===")
        messages.addMessage("Python: %s" % sys.version)

        all_ok = True

        # Check numpy
        try:
            import numpy
            messages.addMessage("NumPy: %s (OK)" % numpy.__version__)
        except ImportError:
            messages.addErrorMessage("NumPy is NOT installed.")
            all_ok = False

        # Check arcpy
        try:
            import arcpy
            messages.addMessage("arcpy: %s (OK)" % arcpy.GetInstallInfo()['Version'])
            messages.addMessage("License: %s" % arcpy.ProductInfo())

            if arcpy.ProductInfo() in ('ArcInfo',):
                messages.addMessage("PolygonNeighbors: Available (Advanced license)")
            else:
                messages.addWarningMessage(
                    "PolygonNeighbors may not be available. "
                    "Adjacency computation will use slower fallback method."
                )
        except ImportError:
            messages.addErrorMessage("arcpy is NOT available.")
            all_ok = False

        # Check model weights (.npz)
        weights_path = os.path.join(os.path.dirname(__file__), "models", "scorer_weights_v7.npz")
        if os.path.exists(weights_path):
            size_kb = os.path.getsize(weights_path) / 1024.0
            messages.addMessage("Model weights: %s (%.0f KB, OK)" % (weights_path, size_kb))
        else:
            messages.addWarningMessage(
                "Default model weights not found at: %s\n"
                "Please ensure scorer_weights_v7.npz is in the models/ folder.\n"
                "Convert from .pt using convert_weights_to_npz.py (requires Python 3 + PyTorch)."
                % weights_path
            )
            all_ok = False

        # Check PyTorch (informational only)
        messages.addMessage("")
        messages.addMessage("Note: This ArcMap version uses pure NumPy for model inference.")
        messages.addMessage("PyTorch is NOT required.")

        if all_ok:
            messages.addMessage("\nAll dependencies are satisfied. Ready to use!")
        else:
            messages.addErrorMessage("\nSome dependencies are missing. See errors above.")


# ======================================================================
# Tool 2: Optimize Land Use
# ======================================================================

class OptimizeLandUseTool(object):
    def __init__(self):
        self.label = u"DRL Land Use Optimization"
        self.description = (
            u"Optimize farmland layout using a trained Deep Reinforcement "
            u"Learning (Maskable PPO) model. Performs paired farmland-forest "
            u"swaps to reduce average farmland slope while improving spatial "
            u"contiguity, with guaranteed farmland count conservation (FC=0).\n\n"
            u"This ArcMap version uses pure NumPy for model inference (no PyTorch required).\n\n"
            u"Processing stages:\n"
            u"  1. Load the trained DRL scorer model weights (.npz)\n"
            u"  2. Read input feature class and classify parcels\n"
            u"  3. Build a spatial adjacency graph between polygons\n"
            u"  4. Run paired inference (alternating farmland/forest swaps)\n"
            u"  5. Write the optimized results to a new feature class\n\n"
            u"Output fields added:\n"
            u"  OPT_DLMC  - Optimized land use classification name\n"
            u"  OPT_TYPE  - Type code (0=Other, 1=Farmland, 2=Forest)\n"
            u"  CHG_FLAG  - Change flag (0=Unchanged, 1=Farm->Forest, 2=Forest->Farm)\n"
            u"  ORIG_DLMC - Original land use classification (preserved)"
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        import arcpy

        # 0: Input Feature Class
        p_input = arcpy.Parameter(
            displayName=u"Input Feature Class",
            name="input_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input",
        )
        p_input.filter.list = ["Polygon"]

        # 1: DLMC Field
        p_dlmc = arcpy.Parameter(
            displayName=u"Land Use Classification Field (DLMC)",
            name="dlmc_field",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )
        p_dlmc.parameterDependencies = [p_input.name]
        p_dlmc.filter.list = ["Text"]

        # 2: Slope Field
        p_slope = arcpy.Parameter(
            displayName=u"Slope Field",
            name="slope_field",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )
        p_slope.parameterDependencies = [p_input.name]
        p_slope.filter.list = ["Double", "Single", "Long", "Short"]

        # 3: Output Feature Class
        p_output = arcpy.Parameter(
            displayName=u"Output Feature Class",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        # 4: Model Weights File (.npz)
        p_weights = arcpy.Parameter(
            displayName=u"Model Weights File (.npz)",
            name="model_weights",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )
        p_weights.filter.list = ["npz"]
        p_weights.value = os.path.join(
            os.path.dirname(__file__), "models", "scorer_weights_v7.npz"
        )

        # 5: Number of Conversion Pairs
        p_pairs = arcpy.Parameter(
            displayName=u"Number of Conversion Pairs",
            name="n_pairs",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        p_pairs.value = 100
        p_pairs.filter.type = "Range"
        p_pairs.filter.list = [1, 500]

        # 6: Farmland Types
        p_farm = arcpy.Parameter(
            displayName=u"Farmland Type Names",
            name="farmland_types",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        p_farm.filter.type = "ValueList"
        p_farm.filter.list = []
        p_farm.value = [u"\u65f1\u5730", u"\u6c34\u7530"]

        # 7: Forest Types
        p_forest = arcpy.Parameter(
            displayName=u"Forest Type Names",
            name="forest_types",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        p_forest.filter.type = "ValueList"
        p_forest.filter.list = []
        p_forest.value = [u"\u679c\u56ed", u"\u6709\u6797\u5730"]

        return [p_input, p_dlmc, p_slope, p_output, p_weights, p_pairs, p_farm, p_forest]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        import arcpy

        if parameters[0].altered and parameters[0].value:
            fc = parameters[0].valueAsText
            if arcpy.Exists(fc):
                desc = arcpy.Describe(fc)
                if desc.shapeType != "Polygon":
                    parameters[0].setErrorMessage("Input must be a Polygon feature class.")
        return

    def execute(self, parameters, messages):
        import arcpy

        # Add core directory to path
        toolbox_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.join(toolbox_dir, "core")
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        if toolbox_dir not in sys.path:
            sys.path.insert(0, toolbox_dir)

        try:
            import numpy as np
            from core.scorer_standalone import ScorerNetwork
            from core.adjacency import build_adjacency
            from core.data_io import read_feature_class, write_output_fc
            from core.paired_inference import run_paired_inference

            # Parse parameters
            input_fc = parameters[0].valueAsText
            dlmc_field = parameters[1].valueAsText
            slope_field = parameters[2].valueAsText
            output_fc = parameters[3].valueAsText
            weights_path = parameters[4].valueAsText
            n_pairs = parameters[5].value if parameters[5].value else 100
            farmland_types = parameters[6].values if parameters[6].values else [u"\u65f1\u5730", u"\u6c34\u7530"]
            forest_types = parameters[7].values if parameters[7].values else [u"\u679c\u56ed", u"\u6709\u6797\u5730"]

            # === Stage 1: Load model ===
            arcpy.SetProgressor("default", "Loading model weights...")
            messages.addMessage("Loading model weights: %s" % weights_path)
            scorer = ScorerNetwork(weights_path)
            messages.addMessage("  Scorer network: k_parcel=%d, k_global=%d" % (scorer.k_parcel, scorer.k_global))

            # === Stage 2: Read data ===
            arcpy.SetProgressorLabel("Reading input feature class...")
            data = read_feature_class(
                input_fc, dlmc_field, slope_field,
                set(farmland_types), set(forest_types), messages
            )

            n_farm = int((data['initial_types'] == 1).sum())
            n_forest = int((data['initial_types'] == 2).sum())
            if n_farm == 0:
                arcpy.AddError("No farmland parcels found. Check farmland type names.")
                return
            if n_forest == 0:
                arcpy.AddError("No forest parcels found. Check forest type names.")
                return

            effective_pairs = min(n_pairs, n_farm, n_forest)
            if effective_pairs < n_pairs:
                messages.addWarningMessage(
                    "Requested %d pairs but only %d possible "
                    "(farmland=%d, forest=%d). Using %d."
                    % (n_pairs, effective_pairs, n_farm, n_forest, effective_pairs)
                )

            # === Stage 3: Build adjacency ===
            arcpy.SetProgressorLabel("Building spatial adjacency graph...")
            adjacency = build_adjacency(
                input_fc, data['n_parcels'], data['oid_to_idx'], messages
            )

            # === Stage 4: Run paired inference ===
            arcpy.SetProgressor("step", "Running DRL paired inference...",
                                0, effective_pairs, 1)

            def progress_cb(pair_i, total, slope, cont):
                arcpy.SetProgressorLabel(
                    "Pair %d/%d | Slope: %.4f | Contiguity: %.4f"
                    % (pair_i, total, slope, cont)
                )
                arcpy.SetProgressorPosition(pair_i)

            results = run_paired_inference(
                scorer=scorer,
                slopes=data['slopes'],
                areas=data['areas'],
                initial_types=data['initial_types'],
                adjacency=adjacency,
                n_parcels=data['n_parcels'],
                n_pairs=effective_pairs,
                progress_callback=progress_cb,
            )

            # === Stage 5: Write output ===
            arcpy.SetProgressor("default", "Writing output feature class...")
            write_output_fc(
                input_fc=input_fc,
                output_fc=output_fc,
                initial_types=data['initial_types'],
                final_types=results['final_types'],
                dlmc_field=dlmc_field,
                farmland_types=list(farmland_types),
                forest_types=list(forest_types),
                idx_to_oid=data['idx_to_oid'],
                messages=messages,
            )

            # === Summary ===
            arcpy.ResetProgressor()
            messages.addMessage("")
            messages.addMessage("=== Optimization Results ===")
            messages.addMessage("  Completed pairs: %d" % results['completed_pairs'])
            messages.addMessage("  Slope: %.4f -> %.4f (change: %.4f, %.2f%%)"
                                % (results['initial_avg_slope'], results['final_avg_slope'],
                                   results['slope_change'], results['slope_change_pct']))
            messages.addMessage("  Contiguity: %.4f -> %.4f (change: %+.4f)"
                                % (results['initial_contiguity'], results['final_contiguity'],
                                   results['cont_change']))
            messages.addMessage("  Farmland count change: %d" % results['farmland_change'])
            messages.addMessage("")
            messages.addMessage("Output: %s" % output_fc)

        except Exception as e:
            arcpy.AddError("Optimization failed: %s" % str(e))
            arcpy.AddError(traceback.format_exc())
