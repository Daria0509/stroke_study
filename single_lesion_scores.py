#!/usr/bin/env python3

"""
Per-region lesion scoring using the AAL atlas + hub-score table (single patient)

This script computes lesion involvement per brain region for ONE patient and
exports a ranked Excel table of hub-weighted lesion scores.

Idea
-------
Given:
- a binary lesion mask in MNI space (1 mm grid in this dataset)
- the AAL atlas (SPM12; distributed at ~2 mm)
- an external hub score table

We:
1) Resample the AAL atlas from its native resolution to the lesion mask grid (1 mm)
   using nearest-neighbor interpolation to preserve integer labels.
2) For each AAL region (excluding cerebellum/background):
      - compute fraction of region overlapped by lesion:
            pct_affected = (# voxels where lesion AND region) / (# voxels in region)
      - map the AAL label name to the hub-score naming convention (AAL_TO_HUB)
      - retrieve the hub score from the Excel table (if available)
      - compute the hub-weighted lesion score:
            lesion_score = pct_affected * hub_score
3) Keep only regions with lesion_score > 0 and sort descending.
4) Save the per-region table to Excel and mark the highest-scoring region.

Inputs 
------------------------------------
- LESION_PATH:
    lesion_mask_<PATIENT>.nii (or .nii.gz)
    Non-zero voxels are treated as lesion.
- HUB_XLSX:
    Excel file with hub scores.
- AAL atlas:
    Automatically downloaded via nilearn.datasets.fetch_atlas_aal(version="SPM12")

Outputs
-------    
- Excel table of per-region results:
    OUT_DIR/lesion_scores_<PATIENT>.xlsx

Excel columns include:
- AAL Region, AAL Index
- Hub Name, Hub Index (if present in table)
- Percentage Affected
- Hub Score
- Lesion Score (= Percentage Affected * Hub Score)
- Highlight (marks the highest lesion score)

Notes 
-------------------
- Atlas: lesion resampling uses nearest neighbor to avoid mixing labels.
- Regions without a mapping (AAL_TO_HUB) or excluded in IGNORE are skipped.
- This script is intended for inspection (single patient). For batch
  processing across many patients, use the multi-patient version that aggregates
  max lesion score per patient.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import datasets, image

# -------------------- paths --------------------
PATIENT       = "07_CG_s1"
MASK_DIR      = Path(".../lesion_score/lesion_masks") 
LESION_PATH   = MASK_DIR / f"lesion_mask_{PATIENT}.nii" # lesion_mask_<PATIENT>.nii
HUB_XLSX      = Path(".../hub-scores.xlsx") # hub scores from Aben
OUT_DIR       = Path(".../lesion_score")
OUT_XLSX      = OUT_DIR / f"lesion_scores_{PATIENT}.xlsx"


OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- atlas --------------------
aal = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nib.load(aal['maps'])
aal_labels  = [str(x) for x in aal['labels']]
aal_indices = [int(i) for i in aal['indices']]

# -------------------- hub scores --------------------
hub_df = pd.read_excel(HUB_XLSX)
hub_df.columns = [c.strip() for c in hub_df.columns]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

COL_NODE       = pick_col(hub_df, ["Node", "node", "Region", "region"])
COL_HUBSCORE   = pick_col(hub_df, ["Hub-score", "Hub Score", "HubScore", "hub_score"])
COL_NODE_INDEX = next((c for c in ["Node index", "Node Index", "Index", "idx"] if c in hub_df.columns), None)
hub_lookup     = hub_df.set_index(COL_NODE)

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

# -------------------- load lesion & resample atlas --------------------
lesion_img = nib.load(str(LESION_PATH))
lesion_mask = lesion_img.get_fdata() > 0

aal_up = image.resample_to_img(
    aal_img,                 # source = 2 mm atlas
    lesion_img,              # target = 1 mm lesion grid
    interpolation='nearest', # preserve integer labels
    force_resample=True,     # new default — set explicitly to suppress warning
    copy_header=True         # new default — set explicitly to suppress warning
)
aal_up_path = OUT_DIR / "AAL_upsampled_1mm.nii.gz"
aal_up.to_filename(str(aal_up_path))

aal_data_1mm = aal_up.get_fdata().astype(int)

'''
print("Lesion (1 mm) shape: ", lesion_img.shape)
print("Lesion affine: \n", lesion_img.affine)
print("AAL atlas (2 mm) shape: ", aal_img.shape)
print("AAL affine: \n", aal_img.affine)
print("Atlas ↑ to 1 mm shape: ", atlas_upsampled.shape)
print("Atlas ↑ to 1 mm affine: \n", atlas_upsampled.affine)
'''

'''
print("\nAAL region labels:")
for label in aal['labels']:
    print("  ", label)
'''

# -------------------- compute lesion scores --------------------
results = []
for aal_region, idx in zip(aal_labels, aal_indices):
    hub = aal_to_hub(aal_region)
    if hub is None:
        continue

    region_mask = (aal_data_1mm == idx)
    if not region_mask.any():
        continue

    overlap = np.logical_and(lesion_mask, region_mask).sum()
    region_vol = int(region_mask.sum())
    pct_aff   = (overlap / region_vol) if region_vol else 0.0

    if hub in hub_lookup.index:
        hub_score = float(hub_lookup.loc[hub, COL_HUBSCORE])
        hub_index = int(hub_lookup.loc[hub, COL_NODE_INDEX]) if COL_NODE_INDEX and pd.notna(hub_lookup.loc[hub, COL_NODE_INDEX]) else None
    else:
        hub_score = 0.0
        hub_index = None

    lesion_score = pct_aff * hub_score
    if lesion_score > 0:
        results.append({
            "AAL Region": aal_region,
            "AAL Index": idx,
            "Hub Name": hub,
            "Hub Index": hub_index,
            "Percentage Affected": pct_aff,
            "Hub Score": hub_score,
            "Lesion Score": lesion_score,
        })

df = pd.DataFrame(results)
df_sorted = df.sort_values("Lesion Score", ascending=False).reset_index(drop=True)

if not df_sorted.empty:
    df_sorted["Highlight"] = ""
    df_sorted.loc[df_sorted["Lesion Score"].idxmax(), "Highlight"] = "<-- HIGHEST"
    print("Highest lesion score:", df_sorted.loc[0, "Lesion Score"])
else:
    print("No regions with non-zero Lesion Score.")

df_sorted.to_excel(OUT_XLSX, index=False)
print(f"Saved per-region scores to: {OUT_XLSX}")
