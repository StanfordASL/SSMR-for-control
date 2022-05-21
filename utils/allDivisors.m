function divs = allDivisors(N)
% compute the set of all integer divisors of the positive integer N
% first, get the list of prime factors of N. 
% Credit: https://www.mathworks.com/matlabcentral/answers/1624440-find-
% closest-divisor-of-a-number
facs = factor(N);
divs = [1,facs(1)];
for fi = facs(2:end)
    % if N is prime, then facs had only one element,
    % and this loop will not execute at all
    
    % this outer product will generate all combinations of
    % the divisors found so far, and the current divisor
    divs = [1;fi]*divs;
    
    % unique eliminates the replicate divisors
    divs = unique(divs(:)');
end
end