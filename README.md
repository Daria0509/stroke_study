# stroke_study
Lesion impact scoring (AAL + hub scores) and network-efficiency regression from DTI connectomes.

# Lesion Impact Scoring & Network Regression (DeepISLES + ExploreDTI)

<sub><b>Purpose:</b> helper scripts to (1) generate ADC/DWI from DTI, (2) compute atlas + hub-weighted lesion impact scores, and (3) test their association with structural network metrics (global efficiency / betweenness).</sub>

> **This repo includes only 4 scripts.** You must set up and run **DeepISLES** (lesion segmentation) and **ExploreDTI** (connectivity matrices) separately.
> 
> ## External tools (not included in this repo)

You must install/run these tools separately to generate the intermediate files used by the scripts here:

- **DeepISLES (lesion segmentation):** https://github.com/ezequieldlrosa/DeepIsles
- **ExploreDTI (tractography + connectivity matrices):**
  - Official site: http://www.exploredti.com/
  - Manual (PDF in this repo): `docs/Manual_ExploreDTI.pdf`

---

## What’s inside

| File | What it does | Main output |
|------|--------------|-------------|
| **`get_adc_dwi.py`** | Fits a DTI tensor (DIPY) and exports **ADC** + **synthetic DWI (b=1000)**, reoriented to RAS | `adc_<PATIENT_ID>.nii.gz`, `dwi_<PATIENT_ID>.nii.gz` |
| **`single_lesion_scores.py`** | Computes **per-region** lesion involvement & hub-weighted lesion scores for **one** patient | `lesion_scores_<PATIENT_ID>.xlsx` |
| **`all_lesion_scores.py`** | Computes **cohort** lesion scores (one row per patient): max hub-weighted regional score + volumes | `all_lesion_scores.xlsx` |
| **`regression.m`** | Computes graph metrics from ExploreDTI matrices and runs correlations/regressions vs lesion score | figures (optional) + model output in MATLAB |

---

## Pipeline (high level)

1. **DTI → ADC/DWI**  
   Use `get_adc_dwi.py` to create the diffusion-derived inputs needed by DeepISLES.
2. **DeepISLES (external)**  
   Run DeepISLES using **FLAIR + ADC + DWI** to obtain lesion masks in MNI space.
3. **(Recommended) QA / manual correction**  
   Inspect lesions (e.g., ITK-SNAP) and export final masks as `lesion_mask_<PATIENT_ID>.nii(.gz)`.
4. **Lesion scoring (Python)**  
   Use AAL (SPM12) + hub-score table to compute lesion impact scores.
5. **ExploreDTI (external)**  
   Run ExploreDTI to export **AAL90 connectivity matrices** (`*_FA_END.mat`).
6. **Regression (MATLAB)**  
   Run `regression.m` to compute network metrics and test association with lesion impact score.

---

## Data you need (inputs & formats)

### Per patient: imaging inputs
**A) FLAIR (for DeepISLES)**
- `flair_<PATIENT_ID>.nii.gz`  *(DeepISLES expects gzipped NIfTI)*

**B) DTI series + gradients (for ADC/DWI generation)**
- `dti_<PATIENT_ID>.nii` *(4D)*
- `dti_<PATIENT_ID>.bval`
- `dti_<PATIENT_ID>.bvec`

### Required intermediate data (from DeepISLES / manual correction)
**Final lesion masks used by the scoring scripts**
- `lesion_mask_<PATIENT_ID>.nii` or `lesion_mask_<PATIENT_ID>.nii.gz`  
  (non-zero voxels are treated as lesion)

**Optional but useful (for covariates)**
- a brain mask in MNI space (e.g. DeepISLES skull-strip output or HD-BET output)  
  If missing, `brain_volume_ml` will be reported as `NaN` by the cohort script.

### Tables
**Hub scores (Excel)**
- An `.xlsx` with at least:
  - a region/node name column (e.g. `Node`)
  - a hub-score column (e.g. `Hub-score`)

**Lesion score sheet (for regression.m)**
- At minimum:
  - `patient_id`
  - `lesion_score`
- Optional covariates (if present, used automatically in covariate models):
  - `infarct_volume_ml`, `brain_volume_ml`, `Age`, `Sex`

### ExploreDTI outputs (required for regression.m)
- One file per patient:
  - `<PATIENT_ID>_FA_END.mat`
- Each `.mat` must contain:
  - `CM` = **90 × 90** weighted connectivity matrix (AAL90)

---

## How lesion impact score is computed

For each patient:
1. The AAL atlas (SPM12) is resampled into lesion mask space (nearest-neighbor).
2. For each AAL region (excluding cerebellum/background):
   - `pct_affected = overlap(lesion, region) / volume(region)`
   - `region_score = pct_affected × hub_score(region)`
3. The patient’s `lesion_score` is the **maximum** `region_score` across regions.

---

## ID matching (important)

Your `PATIENT_ID` must match **exactly** across:
- lesion masks: `lesion_mask_<PATIENT_ID>.nii(.gz)`
- connectivity matrices: `<PATIENT_ID>_FA_END.mat`
- Excel sheets: `patient_id` column

If IDs don’t match, subjects will be dropped during joins.

---

## Quick run examples (edit paths in scripts first)

### Python
```bash
python3 get_adc_dwi.py
python3 all_lesion_scores.py
python3 single_lesion_scores.py
