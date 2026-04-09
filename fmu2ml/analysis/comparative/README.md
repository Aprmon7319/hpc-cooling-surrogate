# Comparative Cooling Model Analysis

This module provides tools for comparing cooling system behaviors across different HPC data center configurations (e.g., Summit, Marconi100, Frontier).

## Analysis Levels

### 1. System-Level Comparison (Original)
Compares aggregate system metrics across configurations.

### 2. CDU-Level Comparison (NEW)
Per-CDU comparative analysis for neural operator development.

---

## CDU-Level Analysis (NEW)

The CDU-level analysis framework answers key questions for neural operator design:
- **Q1**: Given identical inputs, how do outputs differ across models?
- **Q2**: How responsive is each model to input changes (sensitivity)?
- **Q3**: What are the transfer function characteristics (gains, coupling)?
- **Q4**: Where are the operating regime differences (efficiency, constraints)?
- **Q5**: What are the dynamic response differences (time constants)?

### Quick Start (CDU-Level)

```python
from fmu2ml.analysis.comparative import run_analysis

# Run complete CDU-level analysis
results = run_analysis(
    models=["marconi100", "summit", "lassen"],
    cdu_ids=[1, 10, 50],  # Specific CDUs to analyze
    output_dir="analysis_results/cdu_comparison"
)
```

### Command Line (CDU-Level)

```bash
python -m fmu2ml.analysis.comparative.runners.cdu_comparative_analysis \
    --models marconi100 summit lassen \
    --cdus 1 10 50 \
    --output analysis_results/cdu_comparison \
    --verbose
```

### Analysis Workflow

| Phase | Description |
|-------|-------------|
| 1. Data Generation | Standardized inputs for all models |
| 2. Static Analysis | Response surfaces, sensitivity, correlations |
| 3. Dynamic Analysis | Step response, time constants, delays |
| 4. Transfer Function | DC gains, coupling (RGA), stability |
| 5. Operating Regime | Efficiency, constraints, PUE |
| 6. Cross-Model Comparison | Side-by-side metrics |
| 7. Visualization | Heatmaps, surfaces, dashboards |
| 8. Report Generation | Markdown summary |

### CDU-Level Module Structure

```
fmu2ml/analysis/comparative/
├── analyzers/
│   ├── cdu_response_analyzer.py        # Per-CDU I/O analysis
│   ├── sensitivity_analyzer.py         # Jacobian/Sobol sensitivity
│   ├── dynamic_response_analyzer.py    # Step/impulse response
│   ├── transfer_function_analyzer.py   # Gain/coupling analysis
│   └── operating_regime_analyzer.py    # Efficiency/constraints
├── data_generators/
│   ├── standardized_input_generator.py # Identical inputs
│   └── test_sequence_generator.py      # Step, ramp, sweep
├── visualizers/
│   ├── cdu_comparison_visualizer.py    # CDU comparison plots
│   ├── response_surface_visualizer.py  # 2D/3D surfaces
│   └── model_comparison_dashboard.py   # Summary dashboards
└── runners/
    └── cdu_comparative_analysis.py     # Main orchestrator
```

### Key Metrics for Neural Operators

| Analysis Output | Relevance |
|-----------------|-----------|
| Sensitivity rankings | Feature importance |
| Time constants | Temporal resolution |
| Nonlinearity degree | Model complexity |
| Coupling structure | Architecture design |
| Operating envelope | Training distribution |
| Model differences | Transfer learning |

---

## System-Level Analysis (Original)

The comparative analysis framework enables:
- **Multi-system comparison** of cooling model behaviors
- **Normalized metrics** for fair comparison across different scales
- **Comprehensive visualizations** including efficiency, thermal, and scaling analysis
- **Automated report generation** in PDF format

## Quick Start

### Compare Two Systems (Quickest)
```bash
cd scripts/analysis
./compare_cooling_models.sh --systems marconi100 summit --quick
```

### Full Analysis with Data Generation
```bash
./compare_cooling_models.sh --systems marconi100 summit frontier --generate --n-samples 500
```

### View System Profiles Only
```bash
./compare_cooling_models.sh --profile-only --systems summit frontier marconi100
```

## Available Systems

| System | CDUs | Location | Description |
|--------|------|----------|-------------|
| `marconi100` | 49 | Italy | CINECA supercomputer |
| `summit` | 257 | USA (ORNL) | Oak Ridge supercomputer |
| `frontier` | 25 | USA (ORNL) | AMD-based exascale system |
| `lassen` | - | USA (LLNL) | Lawrence Livermore system |
| `fugaku` | - | Japan | Arm-based supercomputer |
| `setonix` | - | Australia | GPU/CPU partitioned |

## Usage Options

### Shell Script (Recommended)
```bash
./compare_cooling_models.sh [OPTIONS]

Options:
  --systems SYSTEM1 [SYSTEM2 ...]   Systems to compare
  --generate                         Generate simulation data
  --n-samples N                      Number of samples (default: 500)
  --quick                            Quick mode (100 samples)
  --profile-only                     Show profiles only
  --output-dir PATH                  Output directory
  --skip-viz                         Skip visualizations
  --no-pdf                           Skip PDF report
  -v, --verbose                      Verbose output
```

### Python Script (Advanced)
```bash
python fmu2ml/scripts/data_analysis/compare_cooling_models.py \
    --systems marconi100 summit frontier \
    --generate-data \
    --n-samples 500 \
    --output-dir results/my_comparison/
```

### SLURM Job (HPC Clusters)
```bash
# Default settings
sbatch scripts/analysis/compare_cooling_models_job.sh

# Custom systems
sbatch --export=SYSTEMS="summit frontier",N_SAMPLES=1000 \
    scripts/analysis/compare_cooling_models_job.sh
```

## Output Structure

```
analysis_results/comparative_analysis/comparison_TIMESTAMP/
├── comparison_efficiency_TIMESTAMP.csv
├── comparison_thermal_TIMESTAMP.csv
├── comparison_flow_TIMESTAMP.csv
├── comparison_dynamic_TIMESTAMP.csv
├── comparison_sensitivity_TIMESTAMP.csv
├── comparison_normalized_TIMESTAMP.csv
├── system_metrics_TIMESTAMP.json
├── profile_marconi100_TIMESTAMP.json
├── profile_summit_TIMESTAMP.json
├── ...
├── 01_efficiency_comparison_TIMESTAMP.png
├── 02_thermal_comparison_TIMESTAMP.png
├── 03_power_profile_TIMESTAMP.png
├── 04_scaling_analysis_TIMESTAMP.png
├── 05_radar_comparison_TIMESTAMP.png
├── 06_flow_comparison_TIMESTAMP.png
├── 07_dynamic_comparison_TIMESTAMP.png
├── cooling_model_comparison_report_TIMESTAMP.pdf
├── comparison_summary_TIMESTAMP.txt
└── comparative_analysis.log
```

## Metrics Computed

### Efficiency Metrics
- Total CDUP power consumption
- Power per CDU (normalized)
- Cooling power ratio (CDUP/Heat Load)
- Heat load per CDU

### Thermal Metrics
- Rack return/supply temperatures
- Temperature delta (ΔT)
- Temperature variability
- Facility temperatures

### Flow Metrics
- Total flow rates (primary/secondary)
- Flow per CDU (normalized)
- Flow variability

### Dynamic Metrics
- Temperature rate of change
- Thermal time constant
- Response characteristics

### Sensitivity Metrics
- Q_flow to CDUP correlation
- Q_flow to temperature correlation

## Configuration

Edit `fmu2ml/config/defaults/comparative_analysis.yaml`:

```yaml
# Systems to compare
systems:
  - marconi100
  - summit
  - frontier

# Data generation settings
data_generation:
  enabled: true
  n_samples: 500
  input_ranges:
    Q_flow: [12.0, 40.0]      # kW
    T_Air: [288.15, 308.15]   # K (15-35°C)
    T_ext: [283.15, 313.15]   # K (10-40°C)

# Visualization settings
visualization:
  create_plots: true
  create_pdf_report: true
```

## Python API

```python
from fmu2ml.analysis.comparative import CoolingModelComparator

# Initialize comparator
comparator = CoolingModelComparator(
    system_names=['marconi100', 'summit', 'frontier'],
    n_workers=4
)

# Generate or load data
comparator.generate_data(n_samples=500)
# or: comparator.load_data({'marconi100': 'path/to/data.parquet', ...})

# Compute metrics
comparator.compute_metrics()

# Run comparisons
efficiency_df = comparator.compare_efficiency()
thermal_df = comparator.compare_thermal_performance()
normalized_df = comparator.compute_normalized_comparison()

# Save results
comparator.save_results('output/comparison/')

# Generate report
print(comparator.generate_summary_report())
```

## Visualizations

1. **Efficiency Comparison**: Bar charts of power consumption and efficiency
2. **Thermal Comparison**: Temperature profiles and deltas
3. **Power Profile**: CDUP power breakdown and ranking
4. **Scaling Analysis**: How metrics scale with system size
5. **Radar Chart**: Multi-dimensional normalized comparison
6. **Flow Comparison**: Coolant flow rates
7. **Dynamic Response**: Temporal behavior characteristics

## Interpreting Results

### Normalized Metrics (Key for Fair Comparison)
When comparing systems of different scales, use normalized (per-CDU) metrics:
- `cdup_power_per_cdu_kw`: Lower is more efficient
- `cooling_power_ratio`: Lower means less cooling overhead per heat removed
- `sec_flow_per_cdu_gpm`: Flow requirements per CDU

### Scaling Behavior
The scaling analysis plots show how metrics change with system size, helping identify:
- Linear scaling (expected)
- Sub-linear scaling (efficiency gains)
- Super-linear scaling (potential issues)

---

## CDU-Level Python API Reference

### AnalysisConfig

```python
from fmu2ml.analysis.comparative import AnalysisConfig

config = AnalysisConfig(
    models=["marconi100", "summit", "lassen", "frontier"],
    cdu_ids=[1, 10, 50, 100],  # Specific CDU indices
    output_dir="./results",
    n_samples=500,
    n_workers=4,
    
    # Analysis toggles
    run_static_analysis=True,
    run_dynamic_analysis=True,
    run_transfer_function_analysis=True,
    run_operating_regime_analysis=True,
    
    # Visualization toggles
    create_visualizations=True,
    create_dashboard=True,
    generate_report=True,
    
    # Verbosity
    verbose=True
)
```

### CDUComparativeAnalysis (Runner)

```python
from fmu2ml.analysis.comparative import CDUComparativeAnalysis

# Create runner
runner = CDUComparativeAnalysis(config)

# Run complete workflow
all_results = runner.run()

# Or run individual phases
runner.generate_standardized_data()
static_results = runner.run_static_analysis()
dynamic_results = runner.run_dynamic_analysis()
transfer_results = runner.run_transfer_function_analysis()
regime_results = runner.run_operating_regime_analysis()
comparison = runner.run_cross_model_comparison()
runner.create_visualizations()
runner.generate_final_report()
```

### Individual Analyzers

```python
from fmu2ml.analysis.comparative.analyzers import (
    CDUResponseAnalyzer,
    SensitivityAnalyzer,
    DynamicResponseAnalyzer,
    TransferFunctionAnalyzer,
    OperatingRegimeAnalyzer
)

# Sensitivity analysis
sens = SensitivityAnalyzer(df)
result = sens.run_full_analysis()
print(result.sensitivity_rankings)

# Dynamic analysis
dyn = DynamicResponseAnalyzer(df)
step_result = dyn.analyze_step_response("output_col")
print(f"Time constant: {step_result.time_constant:.2f}s")

# Transfer function analysis
tf = TransferFunctionAnalyzer(df)
coupling = tf.compute_relative_gain_array()

# Operating regime analysis
regime = OperatingRegimeAnalyzer(df)
envelope = regime.compute_operating_envelope()
pue = regime.compute_pue()
```

### Visualizers

```python
from fmu2ml.analysis.comparative.visualizers import (
    CDUComparisonVisualizer,
    ResponseSurfaceVisualizer,
    ModelComparisonDashboard
)

# Response surface plots
rsv = ResponseSurfaceVisualizer()
rsv.plot_response_surface_3d(data, "Q_flow", "T_Air", "power", output_dir)
rsv.plot_response_comparison(data_dict, "Q_flow", "T_Air", "power", output_dir)

# Executive dashboard
dashboard = ModelComparisonDashboard()
dashboard.create_executive_summary(results_dict, output_dir)
```

---

## Troubleshooting

### Import Errors
Ensure the project is in your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/fmu2ml-exadigit"
```

### Memory Issues
Reduce samples or workers:
```bash
./compare_cooling_models.sh --n-samples 100 --n-workers 2
```

### Missing FMU Files
Ensure FMU files exist in the paths specified in `config/<system>/cooling.json`

## Contributing

To add a new system for comparison:
1. Create config files in `config/<system_name>/`
2. Ensure FMU model is available
3. Add system to `SYSTEM_COLORS` in visualizers (optional)
