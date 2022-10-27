function X = integrateTimeSeries(x,Dt)
% According to trapezoidal rule, assume constant time-stepping Dt
X = (cumsum(x,2)-0.5*(repmat(x(:,1),1,size(x,2))+x))*Dt;
end