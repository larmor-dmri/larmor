% Makes ../data/mfod/mfod_merge_ref.mat, MRIToolkit's merged mFOD FODs on the CFIN
% block of ../data/dki: the two FODs and the fractions as MRTTrack.SaveOutputToNii
% writes them, and the single FOD that
% EDTI_Library.WholeBrainTracking_mDRL_fast_exe makes from them before tracking,
% with its 'linear' (mFOD-WS) and its 'majority' interpolation.
%
% The fit is the one of make_mfod_ref.m: DKI white matter [1.7 0.2 0.2]e-3 mm^2/s
% with K = 0, NODDI grey matter [0.4 1.7e-9 1 0 3e-9 1], isotropic CSF 3e-3
% mm^2/s, 300 directions, inner shell weight 1, L2LSQ.
%
% ref holds, as MATLAB reads the files (MRIToolkit's voxel order and SH basis):
%   fod1, fod2: the DKI and NODDI FODs, SH coefficients, (X, Y, Z, C).
%   fractions: the DKI, NODDI and CSF fractions, (X, Y, Z, 3).
%   linear, majority: the merged FOD of each interpolation, (X, Y, Z, C).
%
% Run from inside tests/matlab_scripts, with MRIToolkit and the NODDI toolbox
% (SynthMeasWatsonSHStickTortIsoV_B0) on the MATLAB path.

global MRIToolkit;

data_dir = fullfile(pwd, '../data/dki');
out_dir = fullfile(pwd, '../data/mfod');
work_dir = fullfile(tempdir, 'larmor_mfod_merge_ref');  % intermediate, not committed
if exist(work_dir, 'dir')
    rmdir(work_dir, 's');
end
mkdir(work_dir);
assert(~isempty(which('SynthMeasWatsonSHStickTortIsoV_B0')), 'NODDI toolbox not on the path');

dwi_file = gunzip(fullfile(data_dir, 'dwi.nii.gz'), work_dir);
dwi_file = dwi_file{1};
mask_file = gunzip(fullfile(data_dir, 'mask.nii.gz'), work_dir);
mask_file = mask_file{1};
bmat_file = fullfile(work_dir, 'bmat.txt');
mat_file = fullfile(work_dir, 'dki.mat');

EDTI.b_Matrix_from_bval_bvec( ...
    'bval_file', fullfile(data_dir, 'bvals'), ...
    'bvec_file', fullfile(data_dir, 'bvecs'), ...
    'output', bmat_file ...
);
EDTI.PerformDTI_DKIFit( ...
    'nii_file', dwi_file, ...
    'txt_file', bmat_file, ...
    'mask_file', mask_file, ...
    'dki', 1, ...
    'grad_perm', 2, ...
    'grad_flip', 1, ...
    'output', mat_file ...
);
mrt_data = MRTQuant.EDTI_Data_2_MRIToolkit('mat_file', mat_file);

% The NODDI kernel and the merge write files in the current folder; keep them
% out of the repository.
here = pwd;
cd(work_dir);
SD = MRTTrack('data', mrt_data);
SD.nDirections = 300;
SD.AddAnisotropicRF_DKI([1.7e-3 0.2e-3 0.2e-3], 0);
SD.AddAnisotropicRF_NODDI({[0.4 1.7e-9 1 0.0 3e-9 1]});
SD.AddIsotropicRF(3e-3);
SD.setInnerShellWeighting(1.0);
SD.setDeconvMethod('L2LSQ');
out = SD.PerformDeconv(0);

prefix = fullfile(work_dir, 'mfod');
SD.SaveOutputToNii(out, prefix);

% The tracking parameters do not change the merged FOD, which is written before
% the tracking starts; the tracking of this small block may fail, so only the
% merged file is required.
p = struct('SeedPointRes', [2 2 2], 'StepSize', 1, 'blob_T', 0.1, ...
    'AngleThresh', 30, 'FiberLengthRange', [30 500], 'SeedMask', [], 'randp', 0);
merged = struct('linear', 'linterp', 'majority', 'majority');
ref = struct();
for mode = {'linear', 'majority'}
    try
        EDTI_Library.WholeBrainTracking_mDRL_fast_exe(mat_file, prefix, ...
            fullfile(work_dir, ['tracts_' mode{1} '.mat']), p, mode{1});
    catch err
        fprintf('Tracking with %s stopped: %s\n', mode{1}, err.message);
    end
    file = [prefix '_CSD_FOD_' merged.(mode{1}) '.nii'];
    assert(exist(file, 'file') == 2, ['MRIToolkit did not write ' file]);
    ref.(mode{1}) = double(EDTI_Library.E_DTI_read_nifti_file(file));
end
cd(here);

ref.fod1 = double(EDTI_Library.E_DTI_read_nifti_file([prefix '_CSD_FOD_1.nii']));
ref.fod2 = double(EDTI_Library.E_DTI_read_nifti_file([prefix '_CSD_FOD_2.nii']));
ref.fractions = double(EDTI_Library.E_DTI_read_nifti_file([prefix '_fractions.nii']));
ref.matlab = version;
ref.mritoolkit_version = MRIToolkit.version;

save(fullfile(out_dir, 'mfod_merge_ref.mat'), 'ref', '-v7');
rmdir(work_dir, 's');

fprintf('FODs %s, fractions %s\n', mat2str(size(ref.fod1)), mat2str(size(ref.fractions)));
