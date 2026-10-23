% Makes ../data/mfod/mfod_kernel.mat and ../data/mfod/mfod_ref.mat, MRIToolkit's
% mFOD (MRTTrack.PerformDeconv with 'L2LSQ') on the CFIN block of ../data/dki.
%
% The setup is the one of mrtd_deconv_fod -method mfod and of
% GettingStarted/Diffusion/3_SphericalDeconv/Example2_mFOD_GRL.m, with fixed
% response functions so the reference does not depend on response estimation:
% DKI white matter [1.7 0.2 0.2]e-3 mm^2/s with K = 0, NODDI grey matter
% [0.4 1.7e-9 1 0 3e-9 1], isotropic CSF 3e-3 mm^2/s, 300 directions, inner
% shell weight 1, L2LSQ_reg 0.1. The data struct comes from
% MRTQuant.EDTI_Data_2_MRIToolkit (b-values rounded to 100, PIS clean-up of the
% b0s), as mrtd_deconv_fod builds it.
%
% kernel (mfod_kernel.mat) holds what the kernels saw and the kernels:
%   bvals, bvecs: SD.data.bvals and SD.data.bvecs (MRIToolkit's gradient frame;
%     AddAnisotropicRF_DKI negates the z of its copy, the NODDI kernel does not).
%   shell_weight, L2LSQ_reg, rf_indices_lr, rf_indices_hr (1-based columns).
%   LRKernel, HRKernel: single, columns DKI block, NODDI block, CSF.
%   recon_dirs: SD.stdReconDirections, the 300 HR directions of both blocks.
%   dki_eigenvalues, dki_k, noddi_parameters, iso_d: the response parameters.
% ref (mfod_ref.mat) holds the result on the mask voxels only:
%   mask: the brain mask on the 8x8x8 grid in MRIToolkit's voxel order (x and y
%     swapped and both reversed from the NIfTI order).
%   voxel: the mask voxels' linear indices (1-based) in that grid, in the order
%     PerformDeconv visits them; ijk: the same voxels as 0-based NIfTI indices.
%   signal: their signals as PerformDeconv saw them (single, after the PIS
%     clean-up), one row per voxel, volumes in SD.data.bvals order.
%   fod: out.FOD (single, 600 columns: 300 DKI then 300 NODDI), fractions:
%     out.fractions (DKI, NODDI, CSF, normalised to sum 1), rss: out.RSS.
%
% Run from inside tests/matlab_scripts, with MRIToolkit and the NODDI toolbox
% (SynthMeasWatsonSHStickTortIsoV_B0) on the MATLAB path.

global MRIToolkit;

data_dir = fullfile(pwd, '../data/dki');
out_dir = fullfile(pwd, '../data/mfod');
work_dir = fullfile(tempdir, 'larmor_mfod_ref');  % intermediate, not committed
if exist(work_dir, 'dir')
    rmdir(work_dir, 's');
end
mkdir(work_dir);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
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
assert(isequal(unique(mrt_data.bvals)', [0 600 1000 1400 2000]));

dki_eigenvalues = [1.7e-3 0.2e-3 0.2e-3];
dki_k = 0;
noddi_parameters = [0.4 1.7e-9 1 0.0 3e-9 1];
iso_d = 3e-3;

% The NODDI kernel writes and deletes temporary gradient files in the current
% folder; keep them out of the repository.
here = pwd;
cd(work_dir);
SD = MRTTrack('data', mrt_data);
SD.nDirections = 300;
SD.AddAnisotropicRF_DKI(dki_eigenvalues, dki_k);
SD.AddAnisotropicRF_NODDI({noddi_parameters});
SD.AddIsotropicRF(iso_d);
SD.setInnerShellWeighting(1.0);
SD.setDeconvMethod('L2LSQ');
cd(here);

% Which frame each anisotropic block uses: compare its HR columns with the
% noise-free signals of fibres along the reconstruction directions, with the
% stored bvecs as they are and with their z negated.
dirs = SD.stdReconDirections;
g = SD.data.bvecs;
gz = g .* [1 1 -1];
b = SD.data.bvals;
dti = @(gg) exp(-b .* (dki_eigenvalues(2) + (dki_eigenvalues(1) - dki_eigenvalues(2)) * (gg * dirs').^2));
protocol = noddi_protocol(b, g, work_dir);
protocol_z = noddi_protocol(b, gz, work_dir);
noddi = zeros(numel(b), size(dirs, 1));
noddi_z = noddi;
for c = 1:size(dirs, 1)
    noddi(:, c) = SynthMeasWatsonSHStickTortIsoV_B0(noddi_parameters, protocol, dirs(c, :)');
    noddi_z(:, c) = SynthMeasWatsonSHStickTortIsoV_B0(noddi_parameters, protocol_z, dirs(c, :)');
end
hr = double(SD.HRKernel);
hr_dki = hr(:, SD.rf_indices_hr{1});
hr_noddi = hr(:, SD.rf_indices_hr{2});
fprintf('DKI block vs bvecs: %.2e, vs z-negated bvecs: %.2e\n', ...
    max(abs(hr_dki - dti(g)), [], 'all'), max(abs(hr_dki - dti(gz)), [], 'all'));
fprintf('NODDI block vs bvecs: %.2e, vs z-negated bvecs: %.2e\n', ...
    max(abs(hr_noddi - noddi), [], 'all'), max(abs(hr_noddi - noddi_z), [], 'all'));

kernel = struct();
kernel.bvals = SD.data.bvals;
kernel.bvecs = SD.data.bvecs;
kernel.shell_weight = SD.shell_weight;
kernel.L2LSQ_reg = SD.L2LSQ_reg;
kernel.rf_indices_lr = SD.rf_indices_lr;
kernel.rf_indices_hr = SD.rf_indices_hr;
kernel.LRKernel = single(SD.LRKernel);
kernel.HRKernel = single(SD.HRKernel);
kernel.recon_dirs = SD.stdReconDirections;
kernel.dki_eigenvalues = dki_eigenvalues;
kernel.dki_k = dki_k;
kernel.noddi_parameters = noddi_parameters;
kernel.iso_d = iso_d;
kernel.matlab = version;
kernel.mritoolkit_version = MRIToolkit.version;

img = mrt_data.img;
siz = size(img);
mask = mrt_data.mask > 0;
voxel = find(mask);

% Zero workers: parfor runs in this session, in order, without a pool.
t = tic;
out = SD.PerformDeconv(0);
run_time = toc(t);

fod = reshape(out.FOD, [], size(out.FOD, 4));
fractions = reshape(out.fractions, [], size(out.fractions, 4));
assert(all(fod(~mask(:), :) == 0, 'all') && all(fractions(~mask(:), :) == 0, 'all'));

% NIfTI index of each MRIToolkit voxel: undo the swap of x and y and the
% reversal of both.
swap = @(a) permute(flip(flip(a, 1), 2), [2 1 3]);
nifti_of = swap(reshape(1:prod(siz(1:3)), siz(1:3)));
[~, at] = ismember(voxel, nifti_of(:));
[i, j, k] = ind2sub(siz(1:3), at);
signal = reshape(img, [], siz(4));

ref = struct();
ref.mask = double(mask);
ref.voxel = voxel;
ref.ijk = [i j k] - 1;
ref.signal = signal(voxel, :);
ref.fod = single(fod(voxel, :));
ref.fractions = fractions(voxel, :);
ref.rss = out.RSS(voxel);
ref.matlab = version;
ref.mritoolkit_version = MRIToolkit.version;

save(fullfile(out_dir, 'mfod_kernel.mat'), 'kernel', '-v7');
save(fullfile(out_dir, 'mfod_ref.mat'), 'ref', '-v7');
rmdir(work_dir, 's');

fprintf('PerformDeconv: %.1f s on %d mask voxels\n', run_time, numel(voxel));
fprintf('LRKernel %dx%d (LR columns per RF: %s), HRKernel %dx%d\n', size(SD.LRKernel), ...
    mat2str(cellfun(@numel, SD.rf_indices_lr)), size(SD.HRKernel));
fprintf('fractions all zero per column: %s\n', mat2str(all(ref.fractions == 0, 1)));
fprintf('fractions min %s, max %s, mean %s\n', mat2str(min(ref.fractions), 3), ...
    mat2str(max(ref.fractions), 3), mat2str(mean(ref.fractions), 3));


function protocol = noddi_protocol(bvals, bvecs, work_dir)
% The NODDI protocol MRTTrack's kernel builds, through FSL text files.
bval_file = fullfile(work_dir, 'check.bval');
bvec_file = fullfile(work_dir, 'check.bvec');
bv = bvals';
bg = bvecs';
save(bval_file, 'bv', '-ascii');
save(bvec_file, 'bg', '-ascii');
protocol = FSL2Protocol(bval_file, bvec_file);
end
