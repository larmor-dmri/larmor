% Makes ../data/matlab/values_v6.mat, values_v7.mat and values_v73.mat: the
% same variables saved with -v6 (version 5 format, uncompressed), -v7 (version
% 5 format, compressed) and -v7.3 (HDF5), for the generic MATLAB reader.
%
% The variables cover every kind of value the reader handles: numeric arrays
% of several classes and dimensions (a 2x3x4 array shows the column-major
% order), logical, char (a row, a char matrix, a non-ASCII character), complex,
% sparse (double and logical), empty values (numeric, char, cell, struct), a
% scalar struct with a nested struct, a struct array, a cell array holding
% other kinds, and two values the reader refuses: a string object and a
% function handle. Complex values also sit inside a struct (single) and a cell
% (double), and a cell and a struct array have 150 elements, more than the
% listing expands.
% values_v4.mat holds values version 4 can store (2D double, char,
% sparse, complex). expected.mat (-v7) holds, for each array, the values in
% MATLAB's linear (column-major) order, so tests can check the layout.
%
% Run from inside tests/matlab_scripts.

clear;
out = fullfile('..', 'data', 'matlab');
if ~exist(out, 'dir')
    mkdir(out);
end

array3d = reshape(1:24, 2, 3, 4);
single2d = single([1.5 2.5 3.5; 4.5 5.5 6.5]);
int16_row = int16([-3 7 9]);
uint8_mat = uint8(magic(3));
int64_col = int64([-2^40; 2^40 + 1]);
scalar = 42;
flags = logical([1 0 1; 0 1 1]);
text = 'hello';
text_rows = ['ab'; 'cd'; 'ef'];
text_accent = ['caf' char(233)];
complex_row = [1+2i, 3-4i, -5i];
sparse_mat = sparse([1 0 2; 0 0 3; 0 4 0; 0 0 0]);
sparse_flags = sparse(logical([0 1; 1 0; 0 0]));
empty_double = zeros(0, 3);
empty_text = '';
empty_cell = {};
empty_struct = struct([]);
record.number = 1;
record.name = 'x';
record.inner.values = [1 2 3];
records = struct('v', {1, [2 3], 'z'});
cells = {1, 'two', [3 4 5]; int8(6), {7, 'eight'}, struct('q', 8)};
text_object = "a string object";
handle = @sin;
nested.z = single([1+1i, 2-3i]);
nested_cell = {'a', [1+2i; 3-4i]};
long_cell = num2cell(1:150);
long_records = struct('v', num2cell(1:150));
double2d = [1 2 3; 4 5 6];

names = setdiff(who, {'out'});
save(fullfile(out, 'values_v6.mat'), names{:}, '-v6');
save(fullfile(out, 'values_v7.mat'), names{:}, '-v7');
save(fullfile(out, 'values_v73.mat'), names{:}, '-v7.3');
v4_names = {'double2d', 'text', 'text_rows', 'complex_row', 'sparse_mat'};
save(fullfile(out, 'values_v4.mat'), v4_names{:}, '-v4');

linear.array3d = array3d(:)';
linear.uint8_mat = double(uint8_mat(:)');
linear.flags = double(flags(:)');
linear.sparse_mat = full(sparse_mat(:)');
save(fullfile(out, 'expected.mat'), 'linear', '-v7');
