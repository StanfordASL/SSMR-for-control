function yData = projectTrajectories(V, xData)
% projectTrajectories: Project trajectories down to reduced coordinates
%   V: 6*num_DOF x 2*k matrix of dominant k modes
%   xData: cell of times and trajectories of full state (

num_traj = size(xData, 1);
yData = cell(num_traj,2);

for i=1:num_traj
    yData{i, 1} = xData{i, 1};
    yData{i, 2} = transpose(V) * xData{i, 2};
end

end

