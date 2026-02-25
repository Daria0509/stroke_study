#!/usr/bin/env python3

"""
Per-region lesion scoring using the AAL atlas + hub-score table (multi-patient)

This script computes lesion involvement per brain region for multiple patients
and exports a single Excel summary table. For each patient, the final reported
lesion score is the maximum hub-weighted regional lesion score across all
(non-ignored) AAL regions.

Idea
----
Given:
- binary lesion masks in MNI space (1 mm grid in this dataset)
- the AAL atlas (SPM12; distributed at ~2 mm)
- an external hub score table 

We:
1) Iterate over all lesion masks in MASK_DIR (lesion_mask_<PATIENT>.nii[.gz]).
2) For each patient:
   a) Load the lesion mask and resample the AAL atlas to the lesion grid (1 mm)
      using nearest-neighbor interpolation to preserve integer labels.
   b) For each AAL region (excluding cerebellum/background):
        - compute fraction of the region overlapped by lesion:
              pct_affected = (# voxels where lesion AND region) / (# voxels in region)
        - map the AAL label name to the hub-score naming convention (AAL_TO_HUB)
        - retrieve the hub score from the Excel table (if available; otherwise 0)
        - compute the hub-weighted lesion score:
              region_lesion_score = pct_affected * hub_score
   c) Keep the **maximum** region_lesion_score as the patient's lesion_score.
   d) Compute infarct volume (mL) from the lesion mask and (optionally) brain
      volume (mL) from a provided brain mask if available.
3) Save one Excel file with one row per patient.

Inputs
------
- MASK_DIR:
    Directory containing lesion masks named:
      lesion_mask_<PATIENT>.nii  or  lesion_mask_<PATIENT>.nii.gz
    Non-zero voxels are treated as lesion.
- HUB_XLSX:
    Excel file with hub scores (region name → hub score). Column names may vary;
    the script detects common variants for the region and hub-score columns.
- AAL atlas:
    Automatically downloaded via nilearn.datasets.fetch_atlas_aal(version="SPM12")
- (Optional) brain mask per patient:
    PATIENT_BASE/<PATIENT>/<BRAIN_MASK_PATTERN>
    If missing, brain volume is reported as NaN.

Outputs
-------
- Excel summary table (one row per patient):
    OUT_DIR/all_lesion_scores.xlsx

Excel columns include:
- patient_id
- lesion_score          (max over regions of pct_affected * hub_score)
- brain_volume_ml       (if brain mask found; otherwise NaN)
- infarct_volume_ml

Notes
-----
- Atlas: lesion resampling uses nearest neighbor to avoid mixing labels.
- Regions without a mapping (AAL_TO_HUB) or excluded in IGNORE are skipped.
"""

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import image, datasets

# -------------------- paths --------------------
MASK_DIR     = Path(".../lesions_masks") # lesion_mask_<PATIENT>.nii
HUB_XLSX     = Path(".../hub-scores.xlsx") # hub scores 
OUT_DIR      = Path(".../lesion_score")
OUT_XLSX     = OUT_DIR / "all_lesion_scores.xlsx"

PATIENT_BASE  = Path(".../patient_files")
BRAIN_MASK_PATTERN = "segmentation/brain_msk-mni.nii.gz" # relative to patient folder

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- atlas & labels --------------------
aal = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nib.load(aal['maps'])
aal_labels  = [str(x) for x in aal['labels']]
aal_indices = [int(i) for i in aal['indices']]
AAL_INDEX_TO_NAME = dict(zip(aal_indices, aal_labels))

# print(len(aal_labels)) # 117 labels in total, only 90 used afterwards

# -------------------- hub scores --------------------
hub_df = pd.read_excel(HUB_XLSX)
hub_df.columns = [c.strip() for c in hub_df.columns]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

COL_NODE     = pick_col(hub_df, ["Node", "node", "Region", "region"])
COL_HUBSCORE = pick_col(hub_df, ["Hub-score", "Hub Score", "HubScore", "hub_score"])
hub_lookup   = hub_df.set_index(COL_NODE)

# -------------------- mapping --------------------
IGNORE = {
    'Background',
    'Cerebelum_Crus1_L','Cerebelum_Crus1_R','Cerebelum_Crus2_L','Cerebelum_Crus2_R',
    'Cerebelum_3_L','Cerebelum_3_R','Cerebelum_4_5_L','Cerebelum_4_5_R',
    'Cerebelum_6_L','Cerebelum_6_R','Cerebelum_7b_L','Cerebelum_7b_R',
    'Cerebelum_8_L','Cerebelum_8_R','Cerebelum_9_L','Cerebelum_9_R',
    'Cerebelum_10_L','Cerebelum_10_R',
    'Vermis_1_2','Vermis_3','Vermis_4_5','Vermis_6','Vermis_7','Vermis_8','Vermis_9','Vermis_10',
}

AAL_TO_HUB = {
    "Precentral_L": "Left Precentral", "Precentral_R": "Right Precentral",
    "Frontal_Sup_L": "Left Superior Frontal", "Frontal_Sup_R": "Right Superior Frontal",
    "Frontal_Sup_Orb_L": "Left Superior Frontal Orbital", "Frontal_Sup_Orb_R": "Right Superior Frontal Orbital",
    "Frontal_Mid_L": "Left Middle Frontal", "Frontal_Mid_R": "Right Middle Frontal",
    "Frontal_Mid_Orb_L": "Left Middle Frontal Orbital", "Frontal_Mid_Orb_R": "Right Middle Frontal Orbital",
    "Frontal_Inf_Oper_L": "Left Inferior Frontal Operculum", "Frontal_Inf_Oper_R": "Right Inferior Frontal Operculum",
    "Frontal_Inf_Tri_L": "Left Inferior Frontal", "Frontal_Inf_Tri_R": "Right Inferior Frontal",
    "Frontal_Inf_Orb_L": "Left Inferior Frontal Orbital", "Frontal_Inf_Orb_R": "Right Inferior Frontal Orbital",
    "Rolandic_Oper_L": "Left Rolandic Operculum", "Rolandic_Oper_R": "Right Rolandic Operculum",
    "Supp_Motor_Area_L": "Left Superior Motor", "Supp_Motor_Area_R": "Right Superior Motor",
    "Olfactory_L": "Left Olfactory", "Olfactory_R": "Right Olfactory",
    "Frontal_Sup_Medial_L": "Left Superior Medial Frontal", "Frontal_Sup_Medial_R": "Right Superior Medial Frontal",
    "Frontal_Med_Orb_L": "Left Medial Frontal Orbital", "Frontal_Med_Orb_R": "Right Medial Frontal Orbital",
    "Rectus_L": "Left Rectus", "Rectus_R": "Right Rectus",
    "Insula_L": "Left Insula", "Insula_R": "Right Insula",
    "Cingulum_Ant_L": "Left Anterior Cingulum", "Cingulum_Ant_R": "Right Anterior Cingulum",
    "Cingulum_Mid_L": "Left Middle Cingulum", "Cingulum_Mid_R": "Right Middle Cingulum",
    "Cingulum_Post_L": "Left Cingulum", "Cingulum_Post_R": "Right Cingulum",
    "Hippocampus_L": "Left Hippocampus", "Hippocampus_R": "Right Hippocampus",
    "ParaHippocampal_L": "Left Parahippocampus", "ParaHippocampal_R": "Right Parahippocampus",
    "Amygdala_L": "Left Amygdala", "Amygdala_R": "Right Amygdala",
    "Calcarine_L": "Left Calcarine", "Calcarine_R": "Right Calcarine",
    "Cuneus_L": "Left Cuneus", "Cuneus_R": "Right Cuneus",
    "Lingual_L": "Left Lingual", "Lingual_R": "Right Lingual",
    "Occipital_Sup_L": "Left Superior Occipital", "Occipital_Sup_R": "Right Superior Occipital",
    "Occipital_Mid_L": "Left Middle Occipital", "Occipital_Mid_R": "Right Middle Occipital",
    "Occipital_Inf_L": "Left Inferior Occipital", "Occipital_Inf_R": "Right Inferior Occipital",
    "Fusiform_L": "Left Fusiform", "Fusiform_R": "Right Fusiform",
    "Postcentral_L": "Left Postcentral", "Postcentral_R": "Right Postcentral",
    "Parietal_Sup_L": "Left Superior Parietal", "Parietal_Sup_R": "Right Superior Parietal",
    "Parietal_Inf_L": "Left Inferior Parietal", "Parietal_Inf_R": "Right Inferior Parietal",
    "SupraMarginal_L": "Left Supramarginal", "SupraMarginal_R": "Right Supramarginal",
    "Angular_L": "Left Angular", "Angular_R": "Right Angular",
    "Precuneus_L": "Left Precuneus", "Precuneus_R": "Right Precuneus",
    "Paracentral_Lobule_L": "Left Central Paracentral Lobule", "Paracentral_Lobule_R": "Right Central Paracentral Lobule",
    "Caudate_L": "Left Caudate", "Caudate_R": "Right Caudate",
    "Putamen_L": "Left Putamen", "Putamen_R": "Right Putamen",
    "Pallidum_L": "Left Pallidum", "Pallidum_R": "Right Pallidum",
    "Thalamus_L": "Left Thalamus", "Thalamus_R": "Right Thalamus",
    "Heschl_L": "Left Heschl", "Heschl_R": "Right Heschl",
    "Temporal_Sup_L": "Left Superior Temporal", "Temporal_Sup_R": "Right Superior Temporal",
    "Temporal_Pole_Sup_L": "Left Superior Temporal Pole", "Temporal_Pole_Sup_R": "Right Superior Temporal Pole",
    "Temporal_Mid_L": "Left Middle Temporal", "Temporal_Mid_R": "Right Middle Temporal",
    "Temporal_Pole_Mid_L": "Left Middle Temporal Pole", "Temporal_Pole_Mid_R": "Right Middle Temporal Pole",
    "Temporal_Inf_L": "Left Inferior Temporal", "Temporal_Inf_R": "Right Inferior Temporal",
}

def aal_to_hub(aal_name: str):
    if aal_name in IGNORE:
        return None
    return AAL_TO_HUB.get(aal_name)

# -------------------- helpers --------------------
def get_patient_id_from_mask(path: Path) -> str:
    m = re.match(r"lesion_mask_(.+)\.nii(\.gz)?$", path.name)
    return m.group(1) if m else path.stem.replace("lesion_mask_", "")

def brain_mask_path_for_patient(pid: str) -> Path:
    return PATIENT_BASE / pid / BRAIN_MASK_PATTERN.format(pid=pid)

def voxel_volume_mm3(nii: nib.Nifti1Image) -> float:
    zooms = nii.header.get_zooms()[:3] # (1.0, 1.0, 1.0) for both flair and lesion 
    if all(z > 0 for z in zooms):
        return float(np.prod(zooms)) # 1.0 mm3 for both flair and lesion 
    det = abs(np.linalg.det(nii.affine[:3, :3]))
    if det <= 0:
        raise ValueError("Cannot determine voxel volume.")
    return float(det)

def lesion_volume_from_mask(lesion_img: nib.Nifti1Image) -> float:
    data = lesion_img.get_fdata()
    vox_mm3 = voxel_volume_mm3(lesion_img) # 1.0 mm3
    n_vox = np.count_nonzero(data > 0)
    vol_mm3 = n_vox * vox_mm3
    vol_ml = vol_mm3 / 1000.0
    return vol_ml

def brain_volume_from_brain_mask(flair_img: nib.Nifti1Image) -> float:
    data = flair_img.get_fdata()
    vox_mm3 = voxel_volume_mm3(flair_img) # 1.0 mm3
    n_vox = np.count_nonzero(data > 0)
    vol_mm3 = n_vox * vox_mm3
    vol_ml = vol_mm3 / 1000.0
    return vol_ml

def compute_patient_max(lesion_path: Path) -> dict:
    patient = get_patient_id_from_mask(lesion_path)

    lesion_img  = nib.load(str(lesion_path))
    lesion_mask = lesion_img.get_fdata() > 0

    # resample atlas to lesion space (nearest)
    aal_up   = image.resample_to_img(aal_img, lesion_img, interpolation='nearest', force_resample=True, copy_header=True)
    aal_data = aal_up.get_fdata().astype(int)

    # volumes
    infarct_ml = lesion_volume_from_mask(lesion_img)
    flair_path = brain_mask_path_for_patient(patient)
    if flair_path.is_file():
        brain_ml = brain_volume_from_brain_mask(nib.load(str(flair_path)))
    else:
        brain_ml = np.nan

    best = {
        "patient_id": patient,
        "Max_Lesion_Score": 0.0,
        "Brain_Volume_ml": float(brain_ml) if np.isfinite(brain_ml) else np.nan,
        "Infarct_Volume_ml": float(infarct_ml),
    }

    # scan regions, compute %affected * hub-score, keep max
    for idx, aal_region in zip(aal_indices, aal_labels):
        hub_name = aal_to_hub(aal_region)
        if hub_name is None:
            continue

        region_mask = (aal_data == idx)
        if not region_mask.any():
            continue

        overlap = np.logical_and(lesion_mask, region_mask).sum()
        region_vol = int(region_mask.sum())
        pct = (overlap / region_vol) if region_vol else 0.0

        hub_score = float(hub_lookup.loc[hub_name, COL_HUBSCORE]) if hub_name in hub_lookup.index else 0.0
        lesion_score = pct * hub_score

        if lesion_score > best["Max_Lesion_Score"]:
            best["Max_Lesion_Score"] = float(lesion_score)

    return best

# -------------------- main --------------------
def is_nifti(p: Path) -> bool:
    if p.name.startswith("._"):   # ignore AppleDouble
        return False
    if p.suffix == ".nii":
        return True
    return len(p.suffixes) >= 2 and p.suffixes[-2:] == [".nii", ".gz"]

mask_files = sorted([p for p in MASK_DIR.iterdir() if p.name.startswith("lesion_mask_") and is_nifti(p)])
if not mask_files:
    raise SystemExit(f"No masks found in {MASK_DIR}")

rows = []
for i, mfile in enumerate(mask_files, 1):
    try:
        row = compute_patient_max(mfile)
        rows.append(row)
        print(f"[{i}/{len(mask_files)}] {row['patient_id']}: max={row['Max_Lesion_Score']:.6f} | "
              f"infarct={row['Infarct_Volume_ml']:.2f} mL | "
              f"brain={row['Brain_Volume_ml'] if np.isfinite(row['Brain_Volume_ml']) else np.nan:.2f} mL")
    except Exception as e:
        print(f"[{i}/{len(mask_files)}] {mfile.name}: ERROR - {e}")

summary_df = pd.DataFrame(rows)

# --- Only keep requested columns  ---
summary_df = summary_df.rename(columns={
    "Max_Lesion_Score": "lesion_score",
    "Brain_Volume_ml":  "brain_volume_ml",
    "Infarct_Volume_ml":"infarct_volume_ml",
})

summary_df = summary_df[["patient_id", "lesion_score", "brain_volume_ml", "infarct_volume_ml"]] \
                       .sort_values("patient_id").reset_index(drop=True)

summary_df.to_excel(OUT_XLSX, index=False)
print(f"\nSaved summary to: {OUT_XLSX}")
