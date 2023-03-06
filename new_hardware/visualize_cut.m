% Visualize and cut trajectory data
clear vars; close all; clc;
warning('off','all')

% Number of observables
numObs = 15;
load obsDecayDataNew

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

% Save non-truncated data
oDataRaw = oData;

%% Check 33 (0.66) and 34 (0.54), 20 (.18 should be .38, perhaps)
cutTime = [.3 .3 .3 .25 .24 .68 .6 .6 .37 .58 .6 .21 .22 .21 .31 .29 .31 .36 .19 .18 .32 .25 .2 .1 .1 .1 .1 .1 .1 .2 .21 .12 .66 .54 .12 .2 .09 .06 .3 .19 .05 .11 .18 .23 .18];
% cutTime = [.1 .1 .1 .25 .24 .68 .3 .3 .37 .38 .6 .21 .22 .21 .31 .29 .31 .36 .19 .18 .32 .25 .2 .1 .1 .1 .1 .1 .1 .2 .21 .12 .3 .3 .12 .2 .09 .06 .1 .05 .05];

for iTraj = 1:size(oData,1) 
    time = oData{iTraj,1};
    indIni = sum(time<cutTime(iTraj))+1;
    indEnd = sum(time<4)+1;
    oData{iTraj,1} = oData{iTraj,1}(indIni:indEnd);
    oData{iTraj,2} = oData{iTraj,2}(:,indIni:indEnd);
end
save('obsDecayDataC.mat','oData')
%% Plot xyz of tip before and after truncation
outdofs = trajs;
nTraj = size(oData,1);

customFigure; colororder(cool(nTraj));
for iTraj = 1:nTraj
    plot3(oDataRaw{iTraj,2}(outdofs(1),:),oDataRaw{iTraj,2}(outdofs(2),:),oDataRaw{iTraj,2}(outdofs(3),:),'Linewidth',1)
end
view(3)
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
zlabel('$z$','Interpreter','latex');

customFigure('subPlot',[3 1]); 
subplot(311); colororder(cool(nTraj));
xlabel('$t$','Interpreter','latex');
ylabel('$x$','Interpreter','latex');
subplot(312); colororder(cool(nTraj));
xlabel('$t$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
subplot(313); colororder(cool(nTraj));
xlabel('$t$','Interpreter','latex');
ylabel('$z$','Interpreter','latex');

% Plotting truncated
for iTraj = 1:nTraj
    subplot(311);
    plot(oData{iTraj,1},oData{iTraj,2}(outdofs(1),:),'Linewidth',1)
    subplot(312);
    plot(oData{iTraj,1},oData{iTraj,2}(outdofs(2),:),'Linewidth',1)
    subplot(313);
    plot(oData{iTraj,1},oData{iTraj,2}(outdofs(3),:),'Linewidth',1)
end

customFigure; colororder(cool(nTraj));
for iTraj = 1:nTraj
    plot3(oData{iTraj,2}(outdofs(1),:),oData{iTraj,2}(outdofs(2),:),oData{iTraj,2}(outdofs(3),:),'Linewidth',1)
end
view(3)
xlabel('$x$','Interpreter','latex');
ylabel('$\dot{x}$','Interpreter','latex');
zlabel('$z$','Interpreter','latex');