% Makes ../data/wm_rf/wm_rf_ref.mat, MRIToolkit's white matter response estimate
% (MRTTrack.Eigenval_IsoK_WM_FromData) on the CFIN block of ../data/dki.
%
% The data struct comes from MRTQuant.EDTI_Data_2_MRIToolkit, as in MRIToolkit's
% fod pipeline, which rounds the b-values to 100 and applies the clean-up of
% physically implausible b0s (PIS). Its 'no_preproc' option is not usable here:
% Eigenval_IsoK_WM_FromData pairs the rows of its design matrix, grouped by
% unique b-value, with the signal in volume order, which only agree when the
% b-values are sorted, and the unrounded ones are not. ref holds:
%   mask: the brain mask, on the 8x8x8 grid in NIfTI voxel order.
%   pis: the masked voxels whose b0s the PIS clean-up changed.
%   voxel: on that grid, the per-voxel fit of Eigenval_IsoK_WM_FromData's loop
%     (E_DTI_WLLS_WW on single-precision signal, design with the isotropic
%     kurtosis column):
%     e_vals (3 eigenvalues of the fitted tensor, D2Dtensor, descending),
%     k (1e-6 * p_K / mean(lambda^2)), fa_mritoolkit (the FA MRIToolkit selects
%     on, from E_DTI_eigensystem_analytic on the raw parameters, whose
%     off-diagonal elements are doubled), fitted (voxel fitted with finite
%     parameters). Voxels not fitted have 0 in e_vals and k.
%   correct: the aggregate from voxel.e_vals and voxel.k: FA from those
%     eigenvalues, voxels with FA > 0.7 in the mask, per-column medians, the
%     second and third averaged; K the mean over those voxels of k in (0, 4).
%     lambdas, K, n (voxels selected), n_k (voxels in the mean of K).
%   mritoolkit: what Eigenval_IsoK_WM_FromData returns (lambdas, K), and n, the
%     voxels its selection picks.
%   single.correct, single.mritoolkit: the same two aggregates on the b0 and b = 1000
%     volumes only (one shell: no kurtosis column, K = 0).
%
% Run from inside tests/matlab_scripts, with MRIToolkit on the MATLAB path.

global MRIToolkit;

data_dir = '../data/dki';
out_dir = '../data/wm_rf';
work_dir = fullfile(tempdir, 'larmor_wm_rf_ref');  % intermediate, not committed
if exist(work_dir, 'dir')
    rmdir(work_dir, 's');
end
mkdir(work_dir);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

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

dwi = double(niftiread(dwi_file));  % (x, y, z, volume), NIfTI order
mask = niftiread(mask_file) > 0;

mat_file = fullfile(work_dir, 'dti.mat');
EDTI.PerformDTI_DKIFit( ...
    'nii_file', dwi_file, ...
    'txt_file', bmat_file, ...
    'mask_file', mask_file, ...
    'dki', 1, ...
    'fit_mode', 'wls', ...
    'grad_perm', 2, ...
    'grad_flip', 1, ...
    'output', mat_file ...
);

% MRIToolkit's reader swaps x and y and then reverses both; undone here.
swap = @(a) permute(flip(flip(double(a), 1), 2), [2 1 3 4]);

data = MRTQuant.EDTI_Data_2_MRIToolkit('mat_file', mat_file);
assert(isequal(unique(data.bvals)', [0 600 1000 1400 2000]));
n_b0 = sum(data.bvals == 0);
% Axis-order self-check: the .mat is sorted by b-value (stably) and the b0s come
% first in both, so its first volume, before the PIS clean-up, is the file's.
fit = load(mat_file, 'DWI');
assert(isequal(swap(fit.DWI{1}), dwi(:, :, :, 1)));
pis = any(swap(data.img(:, :, :, 1:n_b0)) ~= dwi(:, :, :, 1:n_b0), 4);
assert(all(~pis(~mask)));
assert(isequal(swap(data.mask), double(mask)));

ref = struct();
ref.mask = double(mask);
ref.pis = double(pis);

v = voxel_fit(data);
assert(v.fit_dki);
ref.voxel = struct('e_vals', swap(v.e_vals), 'k', swap(v.k), ...
    'fa_mritoolkit', swap(v.fa_mritoolkit), 'fitted', swap(v.fitted));
[ref.correct, ref.mritoolkit] = aggregates(data, v);

single_shell = data;
keep = data.bvals == 0 | data.bvals == 1000;
single_shell.img = data.img(:, :, :, keep);
single_shell.bvals = data.bvals(keep);
single_shell.bvecs = data.bvecs(keep, :);
v = voxel_fit(single_shell);
assert(~v.fit_dki);
[ref.single.correct, ref.single.mritoolkit] = aggregates(single_shell, v);

ref.matlab = version;
ref.mritoolkit_version = MRIToolkit.version;
save(fullfile(out_dir, 'wm_rf_ref.mat'), 'ref', '-v7');
rmdir(work_dir, 's');

fprintf('%d masked voxels, %d touched by the PIS clean-up\n', sum(mask(:)), sum(pis(:)));
names = {'multi-shell', 'single shell'};
pairs = {ref, ref.single};
for i = 1:2
    c = pairs{i}.correct;
    m = pairs{i}.mritoolkit;
    fprintf('%s: correct [%g %g %g] K %g (%d voxels, %d in K)\n', names{i}, ...
        c.lambdas, c.K, c.n, c.n_k);
    fprintf('%s: mritoolkit [%g %g %g] K %g (%d voxels)\n', names{i}, ...
        m.lambdas, m.K, m.n);
end


function v = voxel_fit(data)
% Eigenval_IsoK_WM_FromData's per-voxel loop, unchanged except that it is serial
% and keeps the per-voxel values.
img = single(data.img);
[sx, sy, sz, st] = size(img);
signal_stack = reshape(img, sx * sy * sz, st);
v.fit_dki = sum(unique(round(data.bvals)) > 500) > 1;
[G, good] = build_dt_mat(data, unique(data.bvals));
% The loop pairs the rows of G (grouped by b-value) with the signal in volume
% order; they agree only because the volumes are sorted by b-value.
assert(isequal(good(:), (1:st)'));
if v.fit_dki
    G = [G(:, 1:6) 1/6 * (data.bvals * 1e-3).^2 G(:, end)];
end
np = size(G, 2);
msk = sum(signal_stack == 0, 2) == 0;
n = size(signal_stack, 1);
p = zeros(n, np);
k = zeros(n, 1);
autovals = zeros(n, 3);
fitted = false(n, 1);
for ij = 1:n
    if ~msk(ij)
        continue
    end
    S = signal_stack(ij, :)';
    p1 = EDTI_Library.E_DTI_WLLS_WW(S, G);
    p(ij, :) = p1;
    if any(~isfinite(p(ij, :)))
        continue
    end
    d = p(ij, :);
    autoval = eig([d(1) d(2)/2 d(3)/2; d(2)/2 d(4) d(5)/2; d(3)/2 d(5)/2 d(6)]);
    autovals(ij, :) = sort(autoval, 'descend');
    k(ij) = 1e-6 * p1(end-1) / mean(autoval.^2);
    fitted(ij) = true;
end
p = reshape(p, sx, sy, sz, np);
DT = EDTI_Library.E_DTI_DWI_mat2cell(p(:, :, :, 1:6));
[~, m_FA, ~, ~, m_lambdas] = EDTI_Library.E_DTI_eigensystem_analytic(DT);
v.e_vals = reshape(autovals, sx, sy, sz, 3);
v.k = reshape(real(k), sx, sy, sz);
v.fa_mritoolkit = m_FA / sqrt(3);
v.lambdas_mritoolkit = reshape(real(m_lambdas), sx * sy * sz, 3);
v.fitted = reshape(fitted, sx, sy, sz);
end


function [correct, mrt] = aggregates(data, v)
% The correct aggregate from the fitted tensors, and MRIToolkit's, checked
% against what Eigenval_IsoK_WM_FromData returns.
n = numel(v.k);
e = reshape(v.e_vals, n, 3);
k = v.k(:);
md = mean(e, 2);
num = sqrt(1.5 * sum((e - md).^2, 2));
den = sqrt(sum(e.^2, 2));
fa = zeros(n, 1);
fa(den > 0) = num(den > 0) ./ den(den > 0);
sel = fa > 0.7 & data.mask(:) > 0 & v.fitted(:);
lambdas = median(e(sel, :), 1);
lambdas(2:3) = mean(lambdas(2:3));
correct.lambdas = lambdas;
correct.n = sum(sel);
if v.fit_dki
    in_k = sel & k > 0 & k < 4;
    correct.K = mean(k(in_k));
    correct.n_k = sum(in_k);
else
    correct.K = 0;
    correct.n_k = 0;
end

[mrt.lambdas, mrt.K] = MRTTrack.Eigenval_IsoK_WM_FromData(data, data.mask, double(v.fit_dki));
u = v.fa_mritoolkit(:) > 0.7 & data.mask(:) > 0;
mrt.n = sum(u);
own = median(v.lambdas_mritoolkit(u, :), 1);
own(2:3) = mean(own(2:3));
assert(isequal(single(own), single(mrt.lambdas)));
end


function [G, good] = build_dt_mat(data, bvalues)
% DW_BuildDTMat with fit_S0 = 1 and DW_GetFittedIndexes, private functions of
% MRTTrack.m (MRT_Library's copy cannot reach DW_GetFittedIndexes).
good = [];
for j = 1:length(bvalues)
    good = [good; find(data.bvals(:) == bvalues(j))]; %#ok<AGROW>
end
g = data.bvecs(good, :);
G = [g(:, 1).^2 g(:, 1).*g(:, 2) g(:, 1).*g(:, 3) g(:, 2).^2 g(:, 2).*g(:, 3) g(:, 3).^2];
for j = 1:6
    G(:, j) = -G(:, j) .* data.bvals(good);
end
G = [G ones(size(G, 1), 1)];
end
