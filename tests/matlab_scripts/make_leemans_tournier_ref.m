% Makes ../data/leemans_tournier_ref.mat, the reference for converting SH coefficients
% from MRIToolkit's basis (Larmor's "leemans") to MRtrix's (Larmor's "tournier").
%
% fod_edti: the FOD of ../data/deconv_obj.mat as MRIToolkit writes it to disk
%   (MRTTrack.SaveOutputToNii: least-squares fit of the 300 amplitudes on dir300.txt
%   with the SH class), without SaveOutputToNii's overall scale factor.
% fod_mrtrix: the same FOD in MRIToolkit's "mrtrix" basis (MRTTrack.SHFitMatrix),
%   fitted to the FOD's amplitudes on 2000 directions. Rotating about z keeps each
%   degree, so this fit is exact up to rounding.
%
% Run from inside tests/matlab_scripts, with MRIToolkit on the MATLAB path, after
% run_grl.m has made ../data/deconv_obj.mat.

lmax = 8;

load('../data/deconv_obj.mat', 'deconv_obj');
amplitudes = double(deconv_obj.fod);  % (6, 6, 6, 300), on dir300.txt
[sx, sy, sz, n_dirs] = size(amplitudes);
amplitudes = reshape(amplitudes, [], n_dirs)';  % (300, n_voxels)

% As SaveOutputToNii writes an FOD.
dirs = load('dir300.txt');
sh = SH(lmax, dirs);
coef_edti = sh.coef(amplitudes);  % (45, n_voxels)

% The same functions, refitted in MRIToolkit's "mrtrix" basis.
rng(0);
v = randn(2000, 3);
v = v ./ sqrt(sum(v .^ 2, 2));
b = 3000 * ones(size(v, 1), 1);
edti = MRTTrack.SHFitMatrix('bvals', b, 'bvecs', v, 'basis', 'edti', 'lmax', lmax);
mrtrix = MRTTrack.SHFitMatrix('bvals', b, 'bvecs', v, 'basis', 'mrtrix', 'lmax', lmax);
coef_mrtrix = mrtrix \ (edti * coef_edti);

% The SH class and SHFitMatrix's "edti" basis are the same functions, and the refit
% reproduces the amplitudes: both differences should be at rounding level.
fprintf('SH class vs SHFitMatrix edti: %g\n', max(abs(SH.eval(lmax, v) - edti), [], 'all'));
fprintf('mrtrix refit residual: %g (amplitudes up to %g)\n', ...
    max(abs(mrtrix * coef_mrtrix - edti * coef_edti), [], 'all'), ...
    max(abs(edti * coef_edti), [], 'all'));

[status, commit] = system(['git -C "' fileparts(fileparts(which('MRTTrack'))) ...
    '" rev-parse --short HEAD']);
if status ~= 0
    commit = 'unknown';
end

ref.fod_edti = reshape(coef_edti', sx, sy, sz, []);
ref.fod_mrtrix = reshape(coef_mrtrix', sx, sy, sz, []);
ref.matlab_version = version;
ref.mritoolkit_commit = strtrim(commit);
save('../data/leemans_tournier_ref.mat', 'ref', '-v7');
fprintf('MATLAB %s, MRIToolkit %s\n', ref.matlab_version, ref.mritoolkit_commit);
