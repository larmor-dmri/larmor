% Makes the MRIToolkit result files in ../data/mritoolkit, for Larmor's readers
% of MRIToolkit's .mat files, from the oblique block made by
% ../data/mritoolkit/make_mritoolkit_input.py:
%   dti_fit.mat, dki_fit.mat: EDTI.PerformDTI_DKIFit without and with DKI
%     ('grad_perm' 2, 'grad_flip' 1, as for the CFIN data the block comes from),
%     as MRIToolkit writes them (-v7.3).
%   grl.mat: GRL_Results, the struct MRTTrack's PerformDeconv returns (GRL with
%     a DKI response and two isotropic ones, dRL, 300 directions), saved with
%     -v7.3 as a user's script would; grl_v7.mat the same with -v7.
%   grl_CSD_FOD.nii.gz, grl_fractions.nii.gz: SaveOutputToNii of that result.
%   tracts.mat: MRTTrack.PerformDTIBased_FiberTracking on dti_fit.mat, which
%     writes a version 5 file; tracts_v73.mat the same variables saved with
%     -v7.3, dti_fit_v7.mat the DTI fit's saved with -v7.
%
% Run from inside tests/matlab_scripts, with MRIToolkit on the MATLAB path.

global MRIToolkit;
EDTI.EnforceJSON(false);

data_dir = fullfile(pwd, '../data/mritoolkit');
work_dir = fullfile(tempdir, 'larmor_mritoolkit_ref');  % intermediate, not committed
if exist(work_dir, 'dir')
    rmdir(work_dir, 's');
end
mkdir(work_dir);

dwi_file = gunzip(fullfile(data_dir, 'dwi.nii.gz'), work_dir);
dwi_file = dwi_file{1};
mask_file = gunzip(fullfile(data_dir, 'mask.nii.gz'), work_dir);
mask_file = mask_file{1};
bmat_file = fullfile(work_dir, 'bmat.txt');

EDTI.b_Matrix_from_bval_bvec( ...
    'bval_file', fullfile(data_dir, 'bvals'), ...
    'bvec_file', fullfile(data_dir, 'bvecs'), ...
    'output', bmat_file ...
);
fits = {'dti_fit', 0; 'dki_fit', 1};
for f = 1:size(fits, 1)
    EDTI.PerformDTI_DKIFit( ...
        'nii_file', dwi_file, ...
        'txt_file', bmat_file, ...
        'mask_file', mask_file, ...
        'dki', fits{f, 2}, ...
        'grad_perm', 2, ...
        'grad_flip', 1, ...
        'output', fullfile(data_dir, [fits{f, 1} '.mat']) ...
    );
end
dti_file = fullfile(data_dir, 'dti_fit.mat');

mrt_data = MRTQuant.EDTI_Data_2_MRIToolkit('mat_file', fullfile(data_dir, 'dki_fit.mat'));
here = pwd;
cd(work_dir);
SD = MRTTrack('data', mrt_data);
SD.nDirections = 300;
SD.drl_iters = 600;
SD.AddAnisotropicRF_DKI([1.7e-3 0.2e-3 0.2e-3], 0);
SD.AddIsotropicRF(0.7e-3);
SD.AddIsotropicRF(3e-3);
SD.setDeconvMethod('dRL');
SD.setInnerShellWeighting(0.2);
SD.AutomaticDRLDamping();
GRL_Results = SD.PerformDeconv();
SaveOutputToNii(SD, GRL_Results, fullfile(work_dir, 'grl'));
cd(here);
save(fullfile(data_dir, 'grl.mat'), 'GRL_Results', '-v7.3');
save(fullfile(data_dir, 'grl_v7.mat'), 'GRL_Results', '-v7');
exports = dir(fullfile(work_dir, 'grl*.nii'));
for e = 1:numel(exports)
    gzip(fullfile(work_dir, exports(e).name), data_dir);
end

tracts_file = fullfile(data_dir, 'tracts.mat');
MRTTrack.PerformDTIBased_FiberTracking( ...
    'mat_file', dti_file, ...
    'output', tracts_file, ...
    'SeedPointRes', [2 2 2], ...
    'StepSize', 0.5, ...
    'FAThresh', 0.1, ...
    'AngleThresh', 45, ...
    'FiberLengthRange', [3 100] ...
);

copies = {'tracts', '_v73', '-v7.3'; 'dti_fit', '_v7', '-v7'};
for c = 1:size(copies, 1)
    S = load(fullfile(data_dir, [copies{c, 1} '.mat']));
    save(fullfile(data_dir, [copies{c, 1} copies{c, 2} '.mat']), '-struct', 'S', copies{c, 3});
end
