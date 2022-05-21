function [W,A,V,lambda] = linearpart(M, C, K, varargin)
%   [W,A,V,lambda] = linearpart(M, C, K)
%   [W,A,V,lambda] = linearpart(M, C, K, 'nModes', m)
%   [W,A,V,lambda] = linearpart(M, C, K, 'modeType', 'damped')
%   [W,A,V,lambda] = linearpart(M, C, K, 'nModes', m, 'modeType', 'conservative')
%
%   Spectral analysis of the linear part of a mechanical system
%   
%   INPUT
%   M             (n x n)           Mass matrix of a n-dof system
%   C             (n x n)           Damping matrix of a n-dof system
%   K             (n x n)           Damping matrix of a n-dof system
%   KEYWORD ARGUMENTS
%   nModes        m: integer        number of modes to be considered.
%                                   Default: m = n if n < 1e4; m = 1e4 otw.
%   modeType      string            Select between conservative (default)
%                                   or damped modes. Conservative modes to
%                                   be used only with proportional damping

p = inputParser;
n = size(M,1); if n < 1e4; m = n; else m = 1e4; end
addParameter(p, 'nModes', m);
addParameter(p, 'modeType', 'conservative');
parse(p, varargin{:});
m = p.Results.nModes;
% Default gives conservative mode shapes
A = [sparse(n,n) speye(n,n); -M\K -M\C];
if strcmp(p.Results.modeType,'conservative')
    % Compute conservative eigenvectors
    [Vcons,Dcons] = eigs(K,M,m,'smallestabs');
    dfull = diag(Dcons);
    [dfull,pos] = sort(dfull); Vcons = Vcons(:,pos);
    % Phase space modes: definition and sorting
    D = diag(Vcons.'*M*Vcons);
    disp(diag(Dcons))
    omega = sqrt(dfull); 
    beta = diag(Vcons.'*C*Vcons);
    zeta = (beta./(D.*omega))/2;
    lambda = [omega.*(-zeta + sqrt(zeta.^2 - 1)); omega.*(-zeta - sqrt(zeta.^2 - 1))];
    Vcons = Vcons*diag(1./sqrt(D));
    V = [Vcons sparse(n,m); sparse(n,m) Vcons];
    W = [transpose(Vcons)*M sparse(m,n); ...
        sparse(m,n) transpose(Vcons)*(M)];
    VO = V; V(:,1:2:end) = VO(:,1:m);
    V(:,2:2:end) = VO(:,m+1:end);
    WO = W; W(1:2:end,:) = WO(1:m,:);
    W(2:2:end,:) = WO(m+1:end,:);
else
    % Compute full (damped) eigenvectors
    [V,D] = eig(A);
    dex = diag(D); [~,pos] = sort(abs(imag(dex))); dex = dex(pos);
    V = V(:,pos); 
    VO = V; V = real(V); V(:,2:2:end) = imag(VO(:,1:2:end));
    W = inv(V);
end
end