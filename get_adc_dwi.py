"""
This script takes a raw diffusion MRI (DTI) series plus its gradient tables
(.bval/.bvec), fits a diffusion tensor model with DIPY, and exports:

1) ADC map (apparent diffusion coefficient)
2) synthetic DWI at b=1000 s/mm² computed from ADC and the b=0 image

Both outputs are saved as NIfTI (.nii.gz) and reoriented to RAS
(Right–Anterior–Superior) so downstream tools that expect a standard orientation
(e.g., Deep learning pipelines) can consume consistent volumes.

Inputs
------
- DTI NIfTI:   .../patient_files/<PATIENT>/dti_data/dti_<PATIENT>.nii
- b-values:    .../patient_files/<PATIENT>/dti_data/dti_<PATIENT>.bval
- b-vectors:   .../patient_files/<PATIENT>/dti_data/dti_<PATIENT>.bvec

Outputs
-------
Saved in the patient folder:
- adc_<PATIENT>.nii.gz   (ADC scaled by 1000 before saving: often used as “x10^-3 mm²/s” units)
- dwi_<PATIENT>.nii.gz   (synthetic DWI at b=1000)

Notes
-------------------
- Assumes there is at least one b=0 volume (bvals == 0); the first is used as S0.
- ADC clipping upper bound (0.009) is a heuristic to suppress non-physiological outliers.
- If your pipeline requires rotated b-vectors after reorientation: this script reorients
  only the final derived volumes (ADC/DWI), not the original DTI + gradients.
"""

import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
from nibabel.orientations import (io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff)

def reorient_to_ras(img):
    """
    Force a Nifti1Image into right-anterior-superior (RAS)
    orientation with a positive‐determinant affine.
    """
    data   = img.get_fdata()
    affine = img.affine
    shape  = img.shape[:3]

    # 1) find current orientation
    orig_ornt = io_orientation(affine)
    # 2) calculate transform to RAS
    ras_ornt  = axcodes2ornt(('R','A','S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    # 3) apply it to the data array
    new_data  = apply_orientation(data, transform)
    # 4) compute the new affine
    new_affine = affine.dot(inv_ornt_aff(transform, shape))

    # carry over header (including zooms) but replace affine
    header = img.header.copy()
    return nib.Nifti1Image(new_data, new_affine, header)

# === Load raw diffusion + gradients ===
patient    = "09_CG_s1"
base_path  = f'.../patient_files/{patient}'
dti_path   = f'{base_path}/dti_data/dti_{patient}.nii'
bval_path  = f'{base_path}/dti_data/dti_{patient}.bval'
bvec_path  = f'{base_path}/dti_data/dti_{patient}.bvec'

data, affine = load_nifti(dti_path)
bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
gtab         = gradient_table(bvals, bvecs=bvecs)

# === Mask & fit tensor ===
data_masked, mask = median_otsu(data, vol_idx=range(data.shape[-1]),
                                autocrop=False, dilate=1)
# print(f"Masked data shape: {data_masked.shape}") # same as data.shape if autocrop=False

# vol_idx=range(0, data.shape[-1]): uses all diffusion volumes (b=0 + DWIs) to compute a mean baseline image for masking.
# Applies a median filter and Otsu thresholding to estimate the brain vs. background.
# dilate=1: expands the mask outward by 1 voxel to include a bit more of the brain (useful when skull-stripping is aggressive).

tenmodel = TensorModel(gtab)
tenfit   = tenmodel.fit(data_masked)

# === Compute ADC ===
trace    = np.sum(tenfit.evals, axis=-1) # tenfit.evals: array containing the 3 eigenvalues of the tensor (trace = sum of the 3 eigenvalues per voxel)
adc_map  = np.clip(trace / 3, 0, 0.009) # remove outliers
adc_map  = np.nan_to_num(adc_map)

# === Compute synthetic DWI ===
b0_idx   = np.where(bvals == 0)[0][0] # usually the first one (0)
S0       = data_masked[..., b0_idx] # this volume corresponds to b=0
b = 1000
dwi_map  = S0 * np.exp(-b * adc_map)
dwi_map  = np.nan_to_num(dwi_map)

# === Wrap in RAS and save ===
for suffix, arr in [('adc', adc_map*1000), ('dwi', dwi_map)]: # scale ADC for saving (right input for the algorithm DeepISLES)
    img      = nib.Nifti1Image(arr, affine)
    img_ras  = reorient_to_ras(img)
    out_path = f"{base_path}/{suffix}_{patient}.nii.gz"
    nib.save(img_ras, out_path)
    print(f"Saved {suffix.upper()} → {out_path}")
