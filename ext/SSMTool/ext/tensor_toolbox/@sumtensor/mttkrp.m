function V = mttkrp(X,U,n)
%MTTKRP Matricized tensor times Khatri-Rao product for sumtensor.
%
%   V = MTTKRP(X,U,n) efficiently calculates the matrix product of the
%   n-mode matricization of X with the Khatri-Rao product of all
%   entries in U, a cell array of matrices, except the nth.  How to
%   most efficiently do this computation depends on the type of tensor
%   involved.
%
%   Examples
%   T1 = tensor(rand(3,4,3));
%   T2 = sptensor([2 1 1; 3 4 2; 1 2 3], 1, [3,4,3]);
%   T = sumtensor(T1, T2); %<--Declaring a sumtensor
%
%   mttkrp(T, {rand(3,2), rand(4,3), rand(3,2)}, 2)
%
%   See also SUMTENSOR, TENSOR/MTTKRP, SPTENSOR/MTTKRP, KTENSOR/MTTKRP, 
%   TTENSOR/MTTKRP
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

V = mttkrp(X.part{1},U,n);
for i = 2:length(X.part)
    V = V + mttkrp(X.part{i},U,n);
end
