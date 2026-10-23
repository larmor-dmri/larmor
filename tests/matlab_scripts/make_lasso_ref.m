% Makes ../data/lasso/lasso_problems.mat: small fixed elastic-net problems of
% FOD shape and MATLAB's lasso on them, with the settings of MRIToolkit's
% LASSO deconvolution (Alpha 0.5, default Standardize and intercept).
%
% Each problem has a design matrix X (stored as single; the solutions are of
% double(X)) and a right-hand side y (double):
%   hr_dki60: X = HRKernel columns 1:5:300 (60 of the DKI block) of
%     ../data/mfod/mfod_kernel.mat; y = the signal of the mfod_ref voxel with
%     the largest DKI fraction, divided by the mean of its b <= min(bvals)
%     volumes.
%   hr_mixed120: X = HRKernel DKI columns 1:5:300 and NODDI columns
%     300 + (1:5:300); y = the voxel with the largest NODDI fraction, normalised
%     the same way.
%   lr135: X = LRKernel without its CSF column (135 columns, more rows than
%     columns); y = max(s - LRKernel(:, 136) * f_csf, 0) for the voxel with the
%     largest DKI fraction, s its normalised signal and f_csf its CSF fraction
%     in mfod_ref.
%   hr600: X = HRKernel without its CSF column (600 columns); y the same
%     formula for the voxel with the largest CSF fraction.
%   random_tall: rng(0, 'twister'), then X = single(rand(198, 100)),
%     w = zeros(100, 1) with w([3 17 42 77]) = [0.8 0.5 0.3 0.1],
%     y = double(X) * w + 0.01 * randn(198, 1).
%   random_wide: after random_tall, X = single(randn(198, 250)),
%     y = randn(198, 1).
%
% For each problem: lambda = [0.1 0.01 0.001], alpha = 0.5, lambda_max as
% lasso.m computes it (X0 = (X - mean(X)) ./ std(X, 1), y0 = y - mean(y),
% max(abs(X0' * y0)) / (N * alpha)), and lasso(double(X), y, 'Alpha', 0.5,
% 'Lambda', lambda(i)) with the default RelTol 1e-4 (B_default,
% intercept_default), with RelTol 1e-12 and MaxIter 1e7 (B_tight,
% intercept_tight; the default MaxIter 1e5 stops short of 1e-12 on the kernel
% problems), and on single(X), single(y) with the default RelTol (B_single,
% intercept_single), the precision MRTTrack.PerformDeconv runs it in.
% all_zero_default marks the lambdas whose default solution is all zero;
% kkt_default and kkt_tight are the largest violations of the optimality
% conditions of the standardised problem (see kkt below).
%
% Run from inside tests/matlab_scripts, after make_mfod_ref.m.

kernel_file = fullfile(pwd, '../data/mfod/mfod_kernel.mat');
ref_file = fullfile(pwd, '../data/mfod/mfod_ref.mat');
out_dir = fullfile(pwd, '../data/lasso');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
k = load(kernel_file);
k = k.kernel;
r = load(ref_file);
r = r.ref;

hr = k.HRKernel;
lr = k.LRKernel;
b0 = k.bvals <= min(k.bvals);
normalised = @(v) double(r.signal(v, :))' / mean(double(r.signal(v, b0)));
[~, v_dki] = max(r.fractions(:, 1));
[~, v_noddi] = max(r.fractions(:, 2));
[~, v_csf] = max(r.fractions(:, 3));
no_csf = @(v, csf) max(normalised(v) - double(csf) * double(r.fractions(v, 3)), 0);

names = {'hr_dki60', 'hr_mixed120', 'lr135', 'hr600', 'random_tall', 'random_wide'};
Xs = cell(1, 6);
ys = cell(1, 6);
Xs{1} = hr(:, 1:5:300);
ys{1} = normalised(v_dki);
Xs{2} = hr(:, [1:5:300, 300 + (1:5:300)]);
ys{2} = normalised(v_noddi);
Xs{3} = lr(:, 1:end - 1);
ys{3} = no_csf(v_dki, lr(:, end));
Xs{4} = hr(:, 1:end - 1);
ys{4} = no_csf(v_csf, hr(:, end));
rng(0, 'twister');
Xs{5} = single(rand(198, 100));
w = zeros(100, 1);
w([3 17 42 77]) = [0.8 0.5 0.3 0.1];
ys{5} = double(Xs{5}) * w + 0.01 * randn(198, 1);
Xs{6} = single(randn(198, 250));
ys{6} = randn(198, 1);

lambda = [0.1 0.01 0.001];
alpha = 0.5;
max_iter_tight = 1e7;
problems = struct([]);
t = tic;
for p = 1:numel(names)
    X = double(Xs{p});
    y = ys{p};
    [n, d] = size(X);
    X0 = (X - mean(X)) ./ std(X, 1);
    y0 = y - mean(y);
    pr = struct();
    pr.name = names{p};
    pr.X = Xs{p};
    pr.y = y;
    pr.alpha = alpha;
    pr.lambda = lambda;
    pr.lambda_max = max(abs(X0' * y0)) / (n * alpha);
    pr.B_default = zeros(d, numel(lambda));
    pr.B_tight = zeros(d, numel(lambda));
    pr.B_single = zeros(d, numel(lambda), 'single');
    pr.intercept_default = zeros(1, numel(lambda));
    pr.intercept_tight = zeros(1, numel(lambda));
    pr.intercept_single = zeros(1, numel(lambda), 'single');
    for i = 1:numel(lambda)
        [B, s] = lasso(X, y, 'Alpha', alpha, 'Lambda', lambda(i));
        pr.B_default(:, i) = B;
        pr.intercept_default(i) = s.Intercept;
        [B, s] = lasso(X, y, 'Alpha', alpha, 'Lambda', lambda(i), 'RelTol', 1e-12, 'MaxIter', max_iter_tight);
        pr.B_tight(:, i) = B;
        pr.intercept_tight(i) = s.Intercept;
        [B, s] = lasso(single(X), single(y), 'Alpha', alpha, 'Lambda', lambda(i));
        pr.B_single(:, i) = B;
        pr.intercept_single(i) = s.Intercept;
    end
    pr.all_zero_default = all(pr.B_default == 0, 1);
    pr.kkt_default = kkt(X, y, pr.B_default, lambda, alpha);
    pr.kkt_tight = kkt(X, y, pr.B_tight, lambda, alpha);
    if isempty(problems)
        problems = pr;
    else
        problems(p) = pr;
    end
    fprintf(['%-12s %3dx%3d lambda_max %.4g  all zero %s  nnz %s  ', ...
        'max|default - tight| %s  max|single - tight| %s  min B %s  kkt %s %s\n'], ...
        pr.name, n, d, pr.lambda_max, mat2str(pr.all_zero_default), ...
        mat2str(sum(pr.B_tight ~= 0, 1)), ...
        mat2str(max(abs(pr.B_default - pr.B_tight), [], 1), 2), ...
        mat2str(max(abs(double(pr.B_single) - pr.B_tight), [], 1), 2), ...
        mat2str(min(pr.B_tight, [], 1), 2), mat2str(pr.kkt_default, 2), mat2str(pr.kkt_tight, 2));
end
fprintf('lasso runs: %.1f s\n', toc(t));
fprintf('voxels (1-based rows of mfod_ref): dki %d, noddi %d, csf %d\n', v_dki, v_noddi, v_csf);

info = struct();
info.voxel_rows = [v_dki v_noddi v_csf];
info.matlab = version;
save(fullfile(out_dir, 'lasso_problems.mat'), 'problems', 'info', '-v7');


function v = kkt(X, y, B, lambda, alpha)
% Largest violation of the elastic net's optimality conditions, in lasso.m's
% standardised problem (1 / (2 N)) |y0 - X0 b|^2 + lambda (alpha |b|_1 +
% (1 - alpha) / 2 |b|^2) with b = B .* std(X, 1)'.
n = size(X, 1);
sigma = std(X, 1);
X0 = (X - mean(X)) ./ sigma;
y0 = y - mean(y);
v = zeros(1, numel(lambda));
for i = 1:numel(lambda)
    b = B(:, i) .* sigma';
    g = X0' * (y0 - X0 * b) / n - lambda(i) * (1 - alpha) * b;
    on = b ~= 0;
    viol = max(abs(g) - lambda(i) * alpha, 0);
    viol(on) = abs(g(on) - lambda(i) * alpha * sign(b(on)));
    v(i) = max(viol);
end
end
