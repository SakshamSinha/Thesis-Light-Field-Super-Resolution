clc; close all;

addpath('matlab scripts/');
addpath('data');
addpath('data/train');
addpath('data/test');

% Load the light fields to be processed in this simulation
[lf_names, datasets] = read_configuration('superresolution.cfg');
N = size(lf_names,2);
for n = 1:N
    % Get the dataset
    dataset = datasets{n};
    % Get the light field name
    lf_name = lf_names{n};
    fprintf('Processing light field %s\n',lf_name);
    LF_lfname = load_stanford_lf('data/test/', lf_name);
    out_filename = sprintf('matlab scripts/%s',lf_name);
    save(out_filename, 'LF_lfname');
    fprintf('light field %s processed\n',lf_name);
end