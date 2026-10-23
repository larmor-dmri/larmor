% Makes ../data/grl_obj.mat and ../data/deconv_obj.mat, MRIToolkit's GRL (dRL)
% on the 6x6x6 ISMRM block in ../data/grl, and ../data/bmat.txt, MRIToolkit's
% b-matrix of its gradient files ../data/bvals.txt and ../data/bvecs.txt.
%
% Run from inside tests/matlab_scripts, with MRIToolkit on the MATLAB path.

global MRIToolkit;

EDTI.EnforceNiiGz(true);
EDTI.EnforceJSON(false);
data_dir = fullfile(pwd, '../data');
work_dir = fullfile(tempdir, 'larmor_grl_ref');  % intermediate, not committed
if exist(work_dir, 'dir')
    rmdir(work_dir, 's');
end
mkdir(work_dir);

dwi_file_path = fullfile(data_dir, 'grl', 'block_dwi.nii');
mask_file_path = fullfile(data_dir, 'grl', 'block_dwi_brain_mask.nii');
bval_file_path = fullfile(data_dir, 'bvals.txt');
bvec_file_path = fullfile(data_dir, 'bvecs.txt');
bmat_file_path = fullfile(data_dir, 'bmat.txt');
mat_file_path = fullfile(work_dir, 'dti_fit.mat');

EDTI.b_Matrix_from_bval_bvec( ...
    'bval_file', bval_file_path, ...
    'bvec_file', bvec_file_path, ...
    'output', bmat_file_path ...
);

EDTI.PerformDTI_DKIFit( ...
    'nii_file', dwi_file_path, ...
    'txt_file', bmat_file_path, ...
    'mask_file', mask_file_path, ...
    'output', mat_file_path ...
);

data = MRTQuant.EDTI_Data_2_MRIToolkit('mat_file', mat_file_path);
GRL = MRTTrack('data', data);
GRL.nDirections = 300;
GRL.drl_iters = 600;
GRL.AddAnisotropicRF_DKI([1.7e-3 0.2e-3 0.2e-3], 0);
GRL.AddIsotropicRF(0.7e-3);
GRL.AddIsotropicRF(3e-3);
GRL.setDeconvMethod('dRL');
GRL.setInnerShellWeighting(.2);
GRL.AutomaticDRLDamping();
deconv = GRL.PerformDeconv();

grl_obj.bvals = GRL.data.bvals;
grl_obj.bvecs = GRL.data.bvecs;
grl_obj.img = GRL.data.img;
grl_obj.mask = GRL.data.mask;
grl_obj.shell_weight = GRL.shell_weight;
grl_obj.LRKernel = GRL.LRKernel;
grl_obj.HRKernel = GRL.HRKernel;
grl_obj.NN_L = GRL.NN_L;
grl_obj.NN_H = GRL.NN_H;
save(fullfile(data_dir, 'grl_obj.mat'), 'grl_obj', '-v7');

deconv_obj.fod = deconv.FOD;
deconv_obj.fractions = deconv.fractions;
deconv_obj.RSS = deconv.RSS;
save(fullfile(data_dir, 'deconv_obj.mat'), 'deconv_obj', '-v7');

rmdir(work_dir, 's');
fprintf('MATLAB %s, MRIToolkit %s\n', version, MRIToolkit.version);
