function install
% run this file first to install all external packages  and
% also the main software appropriately

maindir = fileparts(mfilename('fullpath'));
extdir = 'ext';

ssminstall = fullfile(maindir, extdir , 'SSMTool','install.m');
ssmlearninstall = fullfile(maindir, extdir, 'SSMLearn', 'install.m');
run(ssminstall);
run(ssmlearninstall)

addpath(fullfile(maindir, extdir));
addpath(fullfile(maindir, extdir, 'MatlabProgressBar'));
addpath(fullfile(maindir, 'visualization'));
addpath(fullfile(maindir, 'utils'));
addpath(fullfile(maindir, 'modal_trajectory_data'));
addpath(fullfile(maindir, 'modal_trajectory_data/decay'));

end