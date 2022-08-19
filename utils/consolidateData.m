function [filePath, xData] = consolidateData(path, files, Tspan, numNodes, ...
    sampleBatch, varargin)

    numDOF = 3*numNodes;
    if ~iscell(files)
        files = {files};
    end

    xData = cell(length(files),2);
    
    % Parse inputs
    p = inputParser;
    addParameter(p, 'storeType', 'both');
    addParameter(p, 'outputNode', 0)
    parse(p, varargin{:});
    outputNode = p.Results.outputNode;

    numIterations = length(files);
    bar = ProgressBar(numIterations, 'Title', 'Load Data');
    for iFile = 1:numIterations
        currFile = files(iFile);
        decay_file = append(path, currFile{:});
        qv_decay_trajectory = readmatrix(decay_file);

        xData{iFile,1} = Tspan(1:sampleBatch:end);
        if strcmp(p.Results.storeType, 'both')
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, :).';
        elseif strcmp(p.Results.storeType, 'q')
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, 1:numDOF).';
        elseif strcmp(p.Results.storeType, 'v')
            xData{iFile,2} = qv_decay_trajectory(1:sampleBatch:end, numDOF+1:end).';
        elseif strcmp(p.Results.storeType, 'output')
            qout = qv_decay_trajectory(1:sampleBatch:end, 3*outputNode+1:3*outputNode+3);
            vout = qv_decay_trajectory(1:sampleBatch:end, (numDOF + 3*outputNode+1):(numDOF + 3*outputNode+3));
            y_out = cat(2, qout, vout);
            xData{iFile,2} = y_out.';
        end

        bar(1, [], []);
    end
    bar.release();

    % Saving data as mat file. Make sure to load this
    if strcmp(p.Results.storeType, 'both')
        filePath = append(path, 'output/', 'qvConsolidatedData');
    elseif strcmp(p.Results.storeType, 'q')
        filePath = append(path, 'output/', 'qConsolidatedData');
    elseif strcmp(p.Results.storeType, 'v')
        filePath = append(path, 'output/', 'vConsolidatedData');
    elseif strcmp(p.Results.storeType, 'output')
        filePath = append(path, 'output/', 'qv_out_ConsolidatedData');
    end
    save(filePath,'xData', '-v7.3')

end

