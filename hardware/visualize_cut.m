% Visualize and cut trajectory data
clear vars; close all; clc;
warning('off','all')

% Number of observables
numObs = 15;
load obsDecayData

trajs = [13, 14, 15];
numCoord = length(trajs);
%%
for iTraj = 1:size(oData,1)
fig = customFigure('subPlot',[numCoord 1],'figNumber', iTraj);
fig.WindowStyle = 'docked';
labels = ["$x_a$", "$y_a$", "$z_a$", ...
    "$x_b$", "$y_b$", "$z_b$", ...
    "$x_c$", "$y_c$", "$z_c$", ...
    "$x_d$", "$y_d$", "$z_d$", ...
    "$x_e$", "$y_e$", "$z_e$"];
for iPlt = 1:numCoord
    subplot(numCoord,1,iPlt);
    xlabel('$t$','Interpreter','latex');
    ylabel(labels(trajs(iPlt)),'Interpreter','latex');
end


for iObs = 1:numCoord
    subplot(numCoord,1,iObs);
    plot(oData{iTraj,1},oData{iTraj,2}(trajs(iObs),:),'Linewidth',1)
end
end

%%
% 1: also 2
% 7: also .11
% 
cutTime = [.2 1.56 .66 1 .5 .5 .7 .1 .5 .1 .1 .7 .1 .1 .1 .4 .16 .42 .11 .10 .09 1.5 1.5 1 .17 .18 .12 .13 .18 .34 .1 .23 .09 .28 .3 .18 1.5 .12 .25];
for iTraj = 1:size(oData,1) 
    time = oData{iTraj,1};
    indIni = sum(time<cutTime(iTraj))+1;
    indEnd = sum(time<4)+1;
    oData{iTraj,1} = oData{iTraj,1}(indIni:indEnd);
    oData{iTraj,2} = oData{iTraj,2}(:,indIni:indEnd);
end
save('obsDecayDataC.mat','oData')