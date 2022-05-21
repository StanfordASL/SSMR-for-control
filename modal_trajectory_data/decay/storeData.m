% Collect trajectory data
clearvars;
close all
clc

%fileIndex = [1 2 3 12 13 23];
fileIndex = ['001' '0001'];
xData = cell(length(fileIndex),2);
for iFile = 1:length(fileIndex)
mode_initial_condition = fileIndex(iFile);

% decay_filename = sprintf('mode%d_decay.csv', mode_initial_condition);
% q0_decay_trajectory = readmatrix(decay_filename).';

decay_filename = sprintf('mode%s_decay.csv', mode_initial_condition);
q0_decay_trajectory = readmatrix(decay_filename).';

% decay_vel_filename = sprintf('mode%d_decay_velocity.csv', mode_initial_condition);
% v0_decay_trajectory = readmatrix(decay_vel_filename).';

decay_vel_filename = sprintf('mode%s_decay_velocity.csv', mode_initial_condition);
v0_decay_trajectory = readmatrix(decay_vel_filename).';

xData{iFile,1} = linspace(0,3,size(q0_decay_trajectory,2));
xData{iFile,2} = [q0_decay_trajectory; v0_decay_trajectory];
end
save('dataDecay','xData')