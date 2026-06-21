# mock-patient-profile

## Description

Mock patient-aware morphology profiling workflow using BBBC021, CytoTable, CytoDataFrame, coSMicQC, and pycytominer.

## Goal

Prototype the computational architecture for a future patient-derived fibroblast Cell Painting project using fully open datasets and tooling.

The mock intentionally excludes vendor-specific tooling (e.g., SImA) until real exports are available.

## Architecture

BBBC021 Images / Features
↓
CellProfiler-style outputs
↓
CytoTable
↓
Parquet
↓
CytoDataFrame
↓
coSMicQC
↓
pycytominer
↓
Patient-aware profiles
↓
Mock clinical + snRNA-seq integration

## Repository Scaffold

Start from:
https://github.com/CU-DBMI/template-uv-python-research-software/

Project owner:

- Dave Bunten

Repository:

- mock-patient-profile

## Milestone 1: Dataset Ingestion

- Acquire BBBC021
- Create reproducible download workflow
- Validate metadata
- Convert to Parquet
- Establish canonical schema

## Milestone 2: CytoTable Integration

- Build ingestion wrappers
- Generate Parquet feature store
- Validate schema consistency

## Milestone 3: CytoDataFrame

Represent:

- Patient
- Sample
- Plate
- Well
- Cell
- Profile

Test patient-aware operations.

## Milestone 4: coSMicQC

Generate:

- Cell counts
- Missingness
- Feature drift
- Plate effects
- Outlier detection

## Milestone 5: pycytominer

Benchmark:

- Normalization
- Feature selection
- Aggregation
- Consensus profiling

## Milestone 6: Mock Patient Layer

Generate synthetic:

- PatientID
- DiseaseGroup
- FailureType
- Age
- Sex

Disease groups:

- Healthy
- Stable SV
- Fontan Failure
- Systolic Failure

## Milestone 7: Batch Correction

Compare:

- None
- Robust Z-score
- Sphering
- Harmony
- ComBat

Evaluate:

- Replicate correlation
- Profile stability
- Batch leakage
- Disease preservation

## Milestone 8: Multi-omic Integration

Use lightweight mock tables:

- patient.parquet
- clinical.parquet
- snrna_summary.parquet
- morphology_profile.parquet

Focus on joins and schema design rather than full single-cell workflows.

## Success Criteria

Demonstrate:

- End-to-end open workflow
- Patient-aware aggregation
- Reproducible QC
- Batch-correction benchmarking
- Future compatibility with real project data
