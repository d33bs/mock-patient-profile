# Analysis review & study sheet

A working guide for leading the analysis components of `mock-patient-profile`.
For each step of the workflow it lists **what it does**, **where the code is**,
**what to understand / research**, and **what to read**. Use it as a checklist
and a reading map, not a tutorial.

## Mental model (read this first)

This repo is a **plumbing harness with a tunable synthetic data generator**, not
yet a method benchmark. The single-cell features are *planted* with known
disease / treatment / batch signal, so any analysis "recovers" what was put in.
That makes it perfect for validating architecture and for *controlled*
experiments (you know ground truth), but it means conclusions about analysis
*methods* are only as realistic as the generator
([`synthetic.py`](../src/mock_patient_profile/synthetic.py)). Treat the generator
as an experimental knob you co-evolve with the metrics.

Orientation order: run it (`uv run mock-patient-profile run`), read the outputs
(`data/reports/qc_report.md`, the parquets), then read
[`schema.py`](../src/mock_patient_profile/schema.py) →
[`synthetic.py`](../src/mock_patient_profile/synthetic.py) →
[`pipeline.py`](../src/mock_patient_profile/pipeline.py).

**Three must-reads that cover ~80% of the concepts below:**

- Caicedo et al. 2017, *Data-analysis strategies for image-based cell
  profiling*, Nature Methods — the canonical end-to-end methods reference.
- Bray et al. 2016, *Cell Painting, a high-content image-based assay for
  morphological profiling…*, Nature Protocols — the assay itself.
- Chandrasekaran et al. 2021, *Image-based profiling for drug discovery: due for
  a machine-learning upgrade?*, Nature Reviews Drug Discovery — the landscape.

## Workflow steps

| #   | Step                         | Module                                                                                                 |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| 0   | Assay & dataset background   | —                                                                                                      |
| 1   | Canonical schema & storage   | [`schema`](../src/mock_patient_profile/schema.py), [`bbbc021`](../src/mock_patient_profile/bbbc021.py) |
| 2   | Single-cell features         | [`synthetic`](../src/mock_patient_profile/synthetic.py)                                                |
| 3   | CytoTable compartment join   | [`cytotable_io`](../src/mock_patient_profile/cytotable_io.py)                                          |
| 4   | CytoDataFrame representation | [`cytodataframe_io`](../src/mock_patient_profile/cytodataframe_io.py)                                  |
| 5   | Single-cell QC               | [`qc`](../src/mock_patient_profile/qc.py)                                                              |
| 6   | Aggregation & consensus      | [`profiling`](../src/mock_patient_profile/profiling.py)                                                |
| 7   | Normalization                | [`profiling`](../src/mock_patient_profile/profiling.py)                                                |
| 8   | Feature selection            | [`profiling`](../src/mock_patient_profile/profiling.py)                                                |
| 9   | Batch effects & correction   | *(future)* + [`evaluation`](../src/mock_patient_profile/evaluation.py)                                 |
| 10  | Evaluation metrics           | [`evaluation`](../src/mock_patient_profile/evaluation.py)                                              |
| 11  | Patient design & confounding | [`patients`](../src/mock_patient_profile/patients.py)                                                  |
| 12  | Multi-omic integration       | [`multiomics`](../src/mock_patient_profile/multiomics.py)                                              |

### 0. Assay & dataset background

- **Understand:** what the Cell Painting channels stain (DNA, ER, RNA, Golgi/
  membrane, mitochondria; BBBC021 is a 3-channel ancestor: DNA, tubulin, actin),
  what a "plate / well / site / object" is, and what BBBC021's MoA benchmark is.
- **Read:** Bray et al. 2016 (assay); Cimini et al. 2023, *Optimizing the Cell
  Painting assay…*, Nature Protocols; Ljosa et al. 2013, *Comparison of methods
  for image-based profiling…*, J Biomol Screen (the BBBC021 MoA benchmark); the
  [BBBC021 page](https://bbbc.broadinstitute.org/BBBC021).

### 1. Canonical schema & storage

- **What:** the `Metadata_` column convention, PyArrow schemas, Parquet as the
  canonical format; BBBC021 metadata download + dev subset.
- **Understand / research:** why the `Metadata_` vs feature split is the de-facto
  contract across the cytomining tools; Parquet vs CSV trade-offs; columnar/Arrow
  and why DuckDB/Polars are a fit.
- **Read:** Apache Arrow & Parquet docs; pycytominer's "CP feature" conventions
  ([cytomining/pycytominer](https://github.com/cytomining/pycytominer)).

### 2. Single-cell features

- **What:** the synthetic generator emits CellProfiler-style features
  (AreaShape / Intensity / Texture across compartments × channels) with planted
  signal; `SignalConfig` controls strength + feature correlation.
- **Understand / research:** what real CellProfiler features mean (Zernike,
  Haralick texture, radial distribution, granularity); how many there are
  (~1000s) and how correlated they are; how the planted linear/Gaussian model
  differs from real heavy-tailed, nonlinear, correlated data.
- **Read:** CellProfiler measurement docs; Caicedo et al. 2017 (feature
  extraction section); Haralick et al. 1973 (texture features) for background.

### 3. CytoTable compartment join

- **What:** joins per-compartment tables into one single-cell table via DuckDB
  SQL; this repo re-attaches rich metadata by `Metadata_ImageNumber`.
- **Understand / research:** the join keys (`Parent_Cells` / `Parent_Nuclei` /
  object numbers); why single-cell merges are non-trivial at scale; the
  cytominer-database lineage.
- **Read:** [CytoTable docs](https://cytomining.github.io/CytoTable/);
  DuckDB docs.

### 4. CytoDataFrame representation

- **What:** a pandas subclass adding image-aware display + patient-aware
  selection/aggregation helpers over the Patient→Sample→Plate→Well→Cell
  hierarchy.
- **Understand / research:** the entity hierarchy and how identity columns key
  every downstream join; when single-cell vs well vs consensus is the right unit.
- **Read:** [CytoDataFrame](https://github.com/WayScience/CytoDataFrame) README.

### 5. Single-cell QC

- **What:** cell counts, missingness, feature drift, plate effects, and coSMicQC
  z-score outlier detection.
- **Understand / research:** which QC failures matter (debris, clumps, empty
  wells, focus/illumination artifacts); one-sided vs multivariate outlier
  detection; that the default thresholds are arbitrary and need tuning;
  per-plate vs global thresholding.
- **Read:** [coSMicQC](https://github.com/WayScience/coSMicQC) docs; Caicedo
  et al. 2017 (quality control section); illumination-correction discussion in
  the Cell Painting protocols.

### 6. Aggregation & consensus

- **What:** single cells → per-well median profiles → per-patient consensus.
- **Understand / research:** mean vs median vs **MODZ** (moderated z-score)
  consensus; why aggregation discards single-cell distribution/subpopulation
  information; **caveat in this repo:** the per-patient consensus medians across
  *different compounds*, conflating treatment with patient identity — decide the
  right aggregation grain (e.g. per patient × treatment, or DMSO-only).
- **Read:** pycytominer `aggregate`/`consensus` docs; Way et al. 2022,
  *Morphology and gene expression profiling provide complementary information…*,
  Cell Systems (consensus + percent replicating in practice).

### 7. Normalization

- **What:** pycytominer `normalize` — `standardize` (z-score), `mad_robustize`
  (robust), `spherize` (whitening).
- **Understand / research:** **control-based** normalization (normalize to DMSO
  negative controls, per plate) vs whole-plate; why this repo's default
  (`samples="all"`, global) does *not* remove plate effects; Typical Variation
  Normalization (TVN / sphering).
- **Read:** Caicedo et al. 2017 (normalization section); Ando et al. 2017,
  *Improving Phenotypic Measurements in High-Content Imaging Screens* (TVN);
  pycytominer `normalize` docs.

### 8. Feature selection

- **What:** variance threshold, correlation threshold, drop-NA (pycytominer).
- **Understand / research:** why ~1000s of correlated features need pruning;
  blocklisted features; the difference between unsupervised selection and
  leakage-prone supervised selection.
- **Read:** pycytominer `feature_select` docs + the CP feature blocklist;
  Caicedo et al. 2017 (feature selection).

### 9. Batch effects & correction *(the main analysis milestone — not yet built)*

- **What:** detect and remove technical (plate/batch/day) variation while
  preserving biology. The repo provides the *signal* (batch knobs) and the
  *scoring* (evaluation), but no correction methods yet.
- **Understand / research:** robust z-score, sphering, **Harmony**, **ComBat**;
  how each works and what assumptions it makes; the confounding problem (you
  cannot remove batch if it is perfectly confounded with biology).
- **Read (most relevant):** **Arevalo et al. 2024, *Evaluating batch correction
  methods for image-based cell profiling*, Nature Communications** — directly on
  point. Also: Korsunsky et al. 2019 (Harmony, Nature Methods); Johnson et al.
  2007 (ComBat empirical Bayes, Biostatistics); Tran et al. 2020 benchmark of
  scRNA-seq batch correction (Genome Biology). Tools:
  [harmonypy](https://github.com/slowkow/harmonypy),
  `scanpy.pp.combat`.

### 10. Evaluation metrics

- **What:** [`evaluation.benchmark`](../src/mock_patient_profile/evaluation.py)
  scorecard — replicate reproducibility (mAP), disease separation (silhouette +
  classifier), batch leakage (classifier + kNN mixing).
- **Understand / research:** mean average precision for profiling; percent
  replicating / mp-value / grit; kBET and LISI (iLISI/cLISI) for batch mixing;
  why a high disease classifier accuracy can be *batch leakage* under
  confounding; production tools that supersede these reference implementations.
- **Read:** Kalinin et al. 2025, *A versatile information-retrieval framework for
  evaluating profile strength and similarity*, Nature Communications (mAP /
  [copairs](https://github.com/cytomining/copairs));
  [cytominer-eval](https://github.com/cytomining/cytominer-eval) (percent
  replicating, grit, mp-value); Büttner et al. 2019 (kBET, Nature Methods);
  Luecken et al. 2022, *Benchmarking atlas-level data integration…*, Nature
  Methods (scIB metrics, [scib](https://github.com/theislab/scib)); Rousseeuw
  1987 (silhouette).

### 11. Patient design & confounding

- **What:** [`patients.assign_patients`](../src/mock_patient_profile/patients.py)
  maps wells→patients and exposes `disease_plate_confounding`.
- **Understand / research:** experimental design for patient screens — plate
  layout randomization, balancing disease across batches, replicate/control
  placement; why confounding is *the* central threat in patient cohorts; power
  (this mock's n=8 is for plumbing, not inference).
- **Read:** Caicedo et al. 2017 (experimental design); Lin & Boutros 2020,
  *Optimization of confounding factors…* / general design-of-experiments
  references; Nature Methods "Points of significance" columns on blocking and
  confounding.

### 12. Multi-omic integration

- **What:** [`multiomics`](../src/mock_patient_profile/multiomics.py) builds mock
  clinical + snRNA-seq **summary** tables and joins everything per patient with
  DuckDB. Integration here is structural (a join), not analytical.
- **Understand / research:** how morphology relates to transcriptomics; methods
  for genuine multi-omic integration (CCA, MOFA+, WNN) for when you go beyond
  joins; what a real snRNA-seq summary should contain.
- **Read:** Argelaguet et al. 2020 (MOFA+, Genome Biology); Hao et al. 2021
  (Seurat WNN, Cell); Way et al. 2022 (morphology + expression complementarity).

## Known gaps (recap) and what to read for each

| Gap                                    | Why it matters                            | Read / do                                                      |
| -------------------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| Planted, independent, linear signal    | Validation is circular; methods untested  | Evolve `SignalConfig`; Arevalo 2024 for realistic batch models |
| Confounding off by default             | The central patient-screen risk is hidden | Use `disease_plate_confounding`; Caicedo 2017 design section   |
| No batch correction implemented        | The main scientific payload               | Arevalo 2024; Harmony/ComBat docs                              |
| Disease accuracy can read batch        | Misleading under confounding              | Add within-batch disease metric; LISI/kBET reading             |
| Consensus mixes treatments             | Conflates treatment vs patient            | pycytominer consensus; decide aggregation grain                |
| Tiny n (8 patients)                    | No statistical power                      | Scale up; "Points of significance" on power                    |
| Toy 36-feature space                   | Hides dimensionality issues               | Real CP feature docs; Caicedo 2017                             |
| Production metrics are reference impls | Inspectable but not battle-tested         | Swap in copairs / cytominer-eval                               |

## Suggested first hands-on experiments

1. Run easy vs. hard and diff the scorecard:
   `uv run mock-patient-profile run --data_dir data_easy` vs.
   `… --data_dir data_hard --disease_plate_confounding 1.0 --batch_weight 3.0 --feature_correlation 0.5`,
   then `evaluation.benchmark_from_paths(...)` on each.
1. Switch normalization to control-based
   (`normalize_profiles(..., samples="Metadata_MoA == 'DMSO'")`) and re-score.
1. Implement one batch-correction method (start with ComBat via `scanpy`),
   score it with `evaluation.benchmark`, and confirm batch leakage drops while
   disease separation holds.
1. Add a leakage-aware disease metric (disease separation computed *within* each
   batch) and watch it diverge from the naive one under confounding.
