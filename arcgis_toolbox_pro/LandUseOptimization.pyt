# -*- coding: utf-8 -*-
"""
ArcGIS Pro Python Toolbox: Farmland Layout Optimization using DRL.

Uses a trained Maskable PPO scorer network (pure PyTorch) to perform
paired farmland-forest swaps that reduce average farmland slope while
improving spatial contiguity, with guaranteed farmland count conservation.
"""

import os
import sys
import traceback


class Toolbox(object):
    def __init__(self):
        self.label = "Farmland Layout Optimization (DRL)"
        self.alias = "LandUseOpt"
        self.description = (
            "Farmland layout optimization toolbox based on Deep Reinforcement Learning (DRL). "
            "Uses a trained Maskable PPO scorer network to perform paired farmland-forest land use "
            "swaps, reducing average farmland slope while improving spatial contiguity, with "
            "guaranteed farmland count conservation (FC=0). "
            "Developed for the study of farmland spatial layout optimization. "
            "\n\n"
            "Includes two tools:\n"
            "  1. DRL Land Use Optimization - Main optimization tool\n"
            "  2. Check Dependencies - Verify environment and dependencies"
        )
        self.tools = [OptimizeLandUseTool, CheckDependenciesTool]


# ======================================================================
# Tool 1: Check Dependencies
# ======================================================================

class CheckDependenciesTool(object):
    def __init__(self):
        self.label = "Check Dependencies"
        self.description = (
            "Verify that PyTorch and other required packages are available "
            "in the current ArcGIS Pro Python environment.\n\n"
            "This tool checks the following items:\n"
            "  - Python version\n"
            "  - PyTorch (CPU version) installation status\n"
            "  - NumPy installation status\n"
            "  - arcpy version and license level\n"
            "  - PolygonNeighbors availability (Advanced license)\n"
            "  - Default model weights file existence\n\n"
            "It is recommended to run this tool before your first optimization "
            "to ensure the environment is correctly configured. If PyTorch is not "
            "installed, follow the error message instructions to install it via "
            "the ArcGIS Pro Python Command Prompt."
        )
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        return []

    def isLicensed(self):
        return True

    def execute(self, parameters, messages):
        messages.addMessage("=== Dependency Check ===")
        messages.addMessage(f"Python: {sys.version}")

        all_ok = True

        # Check torch
        try:
            import torch
            messages.addMessage(f"PyTorch: {torch.__version__} (OK)")
        except ImportError:
            messages.addErrorMessage(
                "PyTorch is NOT installed. "
                "Install via ArcGIS Pro Python Command Prompt: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
            all_ok = False

        # Check numpy
        try:
            import numpy
            messages.addMessage(f"NumPy: {numpy.__version__} (OK)")
        except ImportError:
            messages.addErrorMessage("NumPy is NOT installed.")
            all_ok = False

        # Check arcpy
        try:
            import arcpy
            messages.addMessage(f"arcpy: {arcpy.GetInstallInfo()['Version']} (OK)")
            messages.addMessage(f"License: {arcpy.ProductInfo()}")

            # Check PolygonNeighbors
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

        # Check model weights
        weights_path = os.path.join(os.path.dirname(__file__), "models", "scorer_weights_v7.pt")
        if os.path.exists(weights_path):
            size_kb = os.path.getsize(weights_path) / 1024
            messages.addMessage(f"Model weights: {weights_path} ({size_kb:.0f} KB, OK)")
        else:
            messages.addWarningMessage(f"Default model weights not found at: {weights_path}")

        if all_ok:
            messages.addMessage("\nAll dependencies are satisfied. Ready to use!")
        else:
            messages.addErrorMessage("\nSome dependencies are missing. See errors above.")


# ======================================================================
# Tool 2: Optimize Land Use
# ======================================================================

class OptimizeLandUseTool(object):
    def __init__(self):
        self.label = "DRL Land Use Optimization"
        self.description = (
            "Optimize farmland layout using a trained Deep Reinforcement "
            "Learning (Maskable PPO) model. Performs paired farmland-forest "
            "swaps to reduce average farmland slope while improving spatial "
            "contiguity, with guaranteed farmland count conservation (FC=0).\n\n"
            "Processing stages:\n"
            "  1. Load the trained DRL scorer model weights\n"
            "  2. Read input feature class and classify parcels\n"
            "  3. Build a spatial adjacency graph between polygons\n"
            "  4. Run paired inference (alternating farmland->forest, forest->farmland swaps)\n"
            "  5. Write the optimized results to a new feature class\n\n"
            "Output fields added to the result:\n"
            "  - OPT_DLMC: Optimized land use classification name\n"
            "  - OPT_TYPE: Optimized type code (0=Other, 1=Farmland, 2=Forest)\n"
            "  - CHG_FLAG: Change flag (0=Unchanged, 1=Farmland->Forest, 2=Forest->Farmland)\n"
            "  - ORIG_DLMC: Original land use classification name (preserved)\n\n"
            "Note: The input data must contain polygon features with land use "
            "classification (text) and slope (numeric) fields."
        )
        self.canRunInBackground = True
        self.category = "Optimization"

    def getParameterInfo(self):
        import arcpy

        # 0: Input Feature Class
        p_input = arcpy.Parameter(
            displayName="Input Feature Class",
            name="input_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input",
            description=(
                "Input polygon feature class containing land use parcels. "
                "Must contain a land use classification text field and a slope numeric field. "
                "Supported formats: Shapefile, File Geodatabase Feature Class. "
                "Coordinate system should be a projected coordinate system (units in meters)."
            ),
        )
        p_input.filter.list = ["Polygon"]

        # 1: DLMC Field
        p_dlmc = arcpy.Parameter(
            displayName="Land Use Classification Field (DLMC)",
            name="dlmc_field",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            description=(
                "Text field containing the land use classification name for each parcel. "
                "The tool uses this field to identify farmland and forest parcels. "
                "For example, the field may contain values such as 'dry land', 'paddy', "
                "'orchard', 'forest', etc. Only parcels matching the specified farmland "
                "or forest type names will participate in the optimization."
            ),
        )
        p_dlmc.parameterDependencies = [p_input.name]
        p_dlmc.filter.list = ["Text"]

        # 2: Slope Field
        p_slope = arcpy.Parameter(
            displayName="Slope Field",
            name="slope_field",
            datatype="Field",
            parameterType="Required",
            direction="Input",
            description=(
                "Numeric field containing the slope value for each parcel (unit: degrees). "
                "This is the key optimization target - the tool aims to reduce the average "
                "slope of all farmland parcels. Typically this is the maximum or mean slope "
                "calculated via DEM zonal statistics."
            ),
        )
        p_slope.parameterDependencies = [p_input.name]
        p_slope.filter.list = ["Double", "Single", "Long", "Short"]

        # 3: Output Feature Class
        p_output = arcpy.Parameter(
            displayName="Output Feature Class",
            name="output_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
            description=(
                "Output polygon feature class to store the optimization results. "
                "The output will contain all original fields plus four new fields: "
                "OPT_DLMC (optimized land use name), OPT_TYPE (type code), "
                "CHG_FLAG (change flag: 0=Unchanged, 1=Farmland->Forest, 2=Forest->Farmland), "
                "and ORIG_DLMC (original land use name preserved for comparison)."
            ),
        )

        # 4: Model Weights File
        p_weights = arcpy.Parameter(
            displayName="Model Weights File (.pt)",
            name="model_weights",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            description=(
                "Path to the trained DRL model weights file (.pt format). "
                "Default: the built-in scorer_weights_v7.pt located in the models/ folder. "
                "The model is dimension-agnostic - the same weights work for any number of "
                "parcels, enabling cross-dataset transfer without retraining."
            ),
        )
        p_weights.filter.list = ["pt"]
        p_weights.value = os.path.join(
            os.path.dirname(__file__), "models", "scorer_weights_v7.pt"
        )

        # 5: Number of Conversion Pairs
        p_pairs = arcpy.Parameter(
            displayName="Number of Conversion Pairs",
            name="n_pairs",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            description=(
                "Number of farmland-forest swap pairs to perform (range: 1-500, default: 100). "
                "Each pair consists of one farmland->forest conversion and one forest->farmland "
                "conversion, ensuring the total farmland count remains unchanged. "
                "A larger value allows more extensive optimization but takes longer. "
                "The actual number of pairs will be limited by the smaller count of "
                "available farmland or forest parcels."
            ),
        )
        p_pairs.value = 100
        p_pairs.filter.type = "Range"
        p_pairs.filter.list = [1, 500]

        # 6: Farmland Types
        p_farm = arcpy.Parameter(
            displayName="Farmland Type Names",
            name="farmland_types",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            description=(
                "Land use classification names that represent farmland. "
                "Default: 'dry land' and 'paddy'. "
                "These must exactly match the text values in the Land Use Classification Field. "
                "Only parcels with these types will be considered as farmland candidates "
                "for the optimization."
            ),
        )
        p_farm.filter.type = "ValueList"
        p_farm.filter.list = []
        p_farm.value = ["\u65f1\u5730", "\u6c34\u7530"]  # Han Di, Shui Tian

        # 7: Forest Types
        p_forest = arcpy.Parameter(
            displayName="Forest Type Names",
            name="forest_types",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            description=(
                "Land use classification names that represent forest land. "
                "Default: 'orchard' and 'forest'. "
                "These must exactly match the text values in the Land Use Classification Field. "
                "Only parcels with these types will be considered as forest candidates "
                "for the optimization."
            ),
        )
        p_forest.filter.type = "ValueList"
        p_forest.filter.list = []
        p_forest.value = ["\u679c\u56ed", "\u6709\u6797\u5730"]  # Guo Yuan, You Lin Di

        return [p_input, p_dlmc, p_slope, p_output, p_weights, p_pairs, p_farm, p_forest]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        import arcpy

        # Validate input exists and has polygons
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
            # Check torch
            try:
                import torch
            except ImportError:
                arcpy.AddError(
                    "PyTorch is not installed. Please run 'Check Dependencies' tool "
                    "or install via: pip install torch --index-url https://download.pytorch.org/whl/cpu"
                )
                return

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
            farmland_types = parameters[6].values if parameters[6].values else ["\u65f1\u5730", "\u6c34\u7530"]
            forest_types = parameters[7].values if parameters[7].values else ["\u679c\u56ed", "\u6709\u6797\u5730"]

            # === Stage 1: Load model ===
            arcpy.SetProgressor("default", "Loading model weights...")
            messages.addMessage(f"Loading model weights: {weights_path}")
            scorer = ScorerNetwork(weights_path)
            messages.addMessage(f"  Scorer network: k_parcel={scorer.k_parcel}, k_global={scorer.k_global}")

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
                    f"Requested {n_pairs} pairs but only {effective_pairs} possible "
                    f"(farmland={n_farm}, forest={n_forest}). Using {effective_pairs}."
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
                    f"Pair {pair_i}/{total} | "
                    f"Slope: {slope:.4f} | Contiguity: {cont:.4f}"
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
            messages.addMessage("\n=== Optimization Results ===")
            messages.addMessage(f"  Completed pairs: {results['completed_pairs']}")
            messages.addMessage(f"  Slope: {results['initial_avg_slope']:.4f} -> {results['final_avg_slope']:.4f} "
                                f"(change: {results['slope_change']:.4f}, {results['slope_change_pct']:.2f}%)")
            messages.addMessage(f"  Contiguity: {results['initial_contiguity']:.4f} -> {results['final_contiguity']:.4f} "
                                f"(change: {results['cont_change']:+.4f})")
            messages.addMessage(f"  Farmland count change: {results['farmland_change']}")
            messages.addMessage(f"\nOutput: {output_fc}")

        except Exception as e:
            arcpy.AddError(f"Optimization failed: {str(e)}")
            arcpy.AddError(traceback.format_exc())
