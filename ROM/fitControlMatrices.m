function [Br,Bv,regErrors] = fitControlMatrices(yuxData,RDInfo,IMInfo)
% Assume graph type approach where reduced coordinates are a linear
% projection. xuTraj is a list of full trajectories (of potentially delay 
% embedded data) where first column is time, second are the embedded
% observables, third the control and last the reduced coordinates 

V = IMInfo.parametrization.tangentSpaceAtOrigin;
paramFun = IMInfo.parametrization.map;
redynFun = RDInfo.reducedDynamics.map;
[Vc,~,~] = svd(V); Vc = Vc(:,size(V,2)+1:end);

t = [];       % time values
Xr = [];      % reduced coordinates at time k
U = [];       % controls at time k
dDvCdt = [];  % time derivatives at time k, difference between  
              % observables and autonomous parametrization 
dXrdt = [];   % time derivatives of reduced coordinates at time k
apprxOrd = 3; % approximation order for the derivative
% Data in matrices
for ii = 1:size(yuxData,1)
    t_in = yuxData{ii,1}; Y_in = yuxData{ii,2}; 
    U_in = yuxData{ii,3}; Xr_in = yuxData{ii,4}; 
    DvC_in = transpose(Vc)*(Y_in - paramFun(Xr_in));
    [dXridt,Xri,ti] = finiteTimeDifference(Xr_in,t_in,apprxOrd);
    [dDvCidt,~,~] = finiteTimeDifference(DvC_in,t_in,apprxOrd);
    Ui = U_in(:, 1+apprxOrd:end-apprxOrd);
    t = [t ti]; Xr = [Xr Xri]; dXrdt = [dXrdt dXridt]; 
    U = [U Ui]; dDvCdt = [dDvCdt dDvCidt]; 
end

% Fit reduced dynamics control matrix
deltaDerivatives = dXrdt - redynFun(Xr);
% Learn whole B matrix
[Br,~,~] = ridgeRegression(U, deltaDerivatives, ones(size(t)), [], 0);
regErrorBr = mean(sqrt(sum((deltaDerivatives - Br*U).^2)))/...
             max(sqrt(sum((deltaDerivatives).^2)));
         
% Fit parametrization control matrix
[Bvc,~,~] = ridgeRegression(U, dDvCdt, ones(size(t)), [], 0);
regErrorBv = mean(sqrt(sum((dDvCdt - Bvc*U).^2)))/...
             max(sqrt(sum((dDvCdt).^2)));
Bv = Vc*Bvc;        
% Output errors of the regression in %
regErrors = [regErrorBr regErrorBv]*100;
end