function [W,V] = projectionMatrix(M,Vcons)

n = size(Vcons,1); m = size(Vcons, 2);
D = diag(Vcons.'*M*Vcons);

Vcons = Vcons*diag(1./sqrt(D));
V = [Vcons sparse(n,m); sparse(n,m) Vcons];
W = [transpose(Vcons)*M sparse(m,n); ...
    sparse(m,n) transpose(Vcons)*(M)];
VO = V; V(:,1:2:end) = VO(:,1:m);
V(:,2:2:end) = VO(:,m+1:end);
WO = W; W(1:2:end,:) = WO(1:m,:);
W(2:2:end,:) = WO(m+1:end,:);

end

