% Makes ../data/noddi/noddi_ref.mat: the NODDI Matlab Toolbox fit of the
% phantom that ../data/make_noddi_phantom.py writes, for the noise-free and the
% noisy signal.
%
% The model is WatsonSHStickTortIsoV_B0, fitted with batch_fitting_single.
% For each of noise_free and noisy the file holds the maps (X, Y, Z) ficvf
% (v_ic), kappa, odi, fiso (v_iso), b0 and error_code, and fibredirs
% (X, Y, Z, 3). The toolbox keeps kappa divided by 10: odi is computed from
% 10 * kappa, as SaveParamsAsNIfTI does. fibredirs is in the frame of the
% bvecs file, so along the voxel axes with x flipped, since the phantom's
% affine has a positive determinant.
%
% The images are read with niftiread and the ROI file is written here in the
% layout of CreateROI, since the toolbox's own reader needs SPM's compiled
% files.
%
% Run from inside tests/matlab_scripts, with the NODDI Matlab Toolbox (v1.05)
% and the Image Processing Toolbox on the MATLAB path (MATLAB R2025a).

phantom_dir = fullfile(pwd, '../data/noddi');
work = tempname;
mkdir(work);
cleanup = onCleanup(@() rmdir(work, 's'));

% load reads a file without an extension as a .mat file.
copyfile(fullfile(phantom_dir, 'bvals'), fullfile(work, 'bvals.txt'));
copyfile(fullfile(phantom_dir, 'bvecs'), fullfile(work, 'bvecs.txt'));
protocol = FSL2Protocol(fullfile(work, 'bvals.txt'), fullfile(work, 'bvecs.txt'));
model = MakeModel('WatsonSHStickTortIsoV_B0');
column = @(name) find(strcmp(model.paramsStr, name));

in_mask = niftiread(fullfile(phantom_dir, 'mask.nii.gz')) > 0;
[xsize, ysize, zsize] = size(in_mask);

names = {'noise_free', 'noisy'};
inputs = {'dwi.nii.gz', 'dwi_noisy.nii.gz'};
for n = 1:numel(names)
    dwi = double(niftiread(fullfile(phantom_dir, inputs{n})));

    % CreateROI's layout: voxels in mask, x slowest, z fastest.
    [k, j, i] = ndgrid(1:zsize, 1:ysize, 1:xsize);
    idx = [i(:) j(:) k(:)];
    idx = idx(in_mask(sub2ind(size(in_mask), idx(:, 1), idx(:, 2), idx(:, 3))), :);
    flat = reshape(dwi, [], size(dwi, 4));
    roi = flat(sub2ind(size(in_mask), idx(:, 1), idx(:, 2), idx(:, 3)), :);
    mask = zeros(size(in_mask));
    mask(sub2ind(size(in_mask), idx(:, 1), idx(:, 2), idx(:, 3))) = 1:size(idx, 1);
    roi_file = fullfile(work, [names{n} '_roi.mat']);
    save(roi_file, 'roi', 'mask', 'idx');

    params_file = fullfile(work, [names{n} '_params.mat']);
    batch_fitting_single(roi_file, protocol, model, params_file);
    params = load(params_file);
    mlps = params.mlps;

    voxels = sub2ind(size(in_mask), idx(:, 1), idx(:, 2), idx(:, 3));
    as_map = @(values) accumarray(voxels, values, [numel(in_mask) 1]);
    fit = struct();
    for name = {'ficvf', 'kappa', 'fiso', 'b0'}
        fit.(name{1}) = reshape(as_map(mlps(:, column(name{1}))), size(in_mask));
    end
    odi = atan2(1, mlps(:, column('kappa')) * 10) * 2 / pi;
    fit.odi = reshape(as_map(odi), size(in_mask));
    fit.error_code = reshape(as_map(double(params.error_code(:))), size(in_mask));
    dirs = GetFibreOrientation(model.name, mlps)';
    fit.fibredirs = zeros(xsize, ysize, zsize, 3);
    for a = 1:3
        fit.fibredirs(:, :, :, a) = reshape(as_map(dirs(:, a)), size(in_mask));
    end
    ref.(names{n}) = fit;
end

save(fullfile(phantom_dir, 'noddi_ref.mat'), 'ref', '-v7');
