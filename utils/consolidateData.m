function [filePath, xData] = consolidateData(path, files, Tspan, numNodes, ...
    sampleBatch, varargin)

    numDOF = 3*numNodes;
    if ~iscell(files)
        files = {files};
    end

    xData = cell(length(files),2);

    p = inputParser;
    addParameter(p, 'storeType', 'both');
    parse(p, varargin{:});

    numIterations = length(files);
    bar = ProgressBar(numIterations, 'Title', 'Load Data');
    for iFile = 1:numIterations
        currFile = files(iFile);
        decay_file = append(path, currFile{:});
        qv_decay_trajectory = readmatrix(decay_file);

        xData{iFile,1} = Tspan;
        if strcmp(p.Results.storeType, 'both')
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, :).';
        elseif strcmp(p.Results.storeType, 'q')
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, 1:numDOF).';
        else
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, numDOF+1:end).';
        end

        bar(1, [], []);
    end
    bar.release();

    % Saving data as mat file. Make sure to load this
    if strcmp(p.Results.storeType, 'both')
        filePath = append(path, 'output/', 'qvConsolidatedData');
        save(filePath,'xData');
    elseif strcmp(p.Results.storeType, 'q')
        filePath = append(path, 'output/', 'qConsolidatedData');
        save(filePath,'xData');
    else
        filePath = append(path, 'output/', 'vConsolidatedData');
        save(filePath,'xData');
    end

end

