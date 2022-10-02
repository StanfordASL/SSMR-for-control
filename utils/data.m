classdef data
    %DATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Property1
    end
    
    methods
        function obj = data(varargin)
            %DATA Construct an instance of this class
            %   Detailed explanation goes here
            % obj.Property1 = inputArg1 + inputArg2;
        end
    end
        
     methods(Static)
        function data_resampled = resample(data, Ts)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            n_traj = size(data.oData, 1);
            data_resampled = {}; % Pre-define
            
            % get sampled points
            t_data = data.oData{1};
            tq = ( t_data(1) : Ts : t_data(end) )';
            
            for idx=1:n_traj
                % Resample observed state data
                data_resampled.oData{idx, 1} = tq';
                data_resampled.oData{idx, 2} = interp1( t_data , data.oData{idx, 2}', tq)';
                
                % Resample full state data
                data_resampled.yData{idx, 1} = tq';
                data_resampled.yData{idx, 2} = interp1( t_data , data.yData{idx, 2}', tq)';
            end
        end
    end
end

