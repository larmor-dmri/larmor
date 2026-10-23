% GRL kernel with a non-zero kurtosis on three shells, for test_matlab_match.py.
% MRIToolkit's AddAnisotropicRF_DKI builds the kernel; its kurtosis term is
% (1/6) b^2 MD^2 K (MRTTrack.m, create_signal_multi_tensor_dki).
% Run from tests/matlab_scripts with MRIToolkit on the path (MATLAB R2025a).
bvecs = load('../data/bvecs.txt')';    % (32, 3), as the single-shell block
dw = bvecs(sum(bvecs.^2, 2) > 0.5, :);
bvals = [0; repmat(1000, size(dw, 1), 1); repmat(2000, size(dw, 1), 1); repmat(3000, size(dw, 1), 1)];
data.bvecs = [0 0 0; dw; dw; dw];
data.bvals = bvals;
data.img = ones(2, 2, 2, length(bvals));
data.mask = ones(2, 2, 2);
data.hdr = [];
GRL = MRTTrack('data', data);
GRL.nDirections = 300;
GRL.AddAnisotropicRF_DKI([1.7e-3 0.2e-3 0.2e-3], 0.8);
grl_kurtosis.bvals = GRL.data.bvals;
grl_kurtosis.bvecs = GRL.data.bvecs;
grl_kurtosis.eigen_vals = [1.7e-3 0.2e-3 0.2e-3];
grl_kurtosis.K = 0.8;
grl_kurtosis.HRKernel = GRL.HRKernel;
save('../data/grl_kurtosis_kernel.mat', 'grl_kurtosis', '-v7');
