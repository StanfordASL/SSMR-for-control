function oData = consolidateSSMTrainData(path, files, idx_end, varargin)

    if ~iscell(files)
        files = {files};
    end

    % Parse inputs
    p = inputParser;
    addParameter(p, 'center', true)
    parse(p, varargin{:});
    toCenter = p.Results.center;

    oData = cell(length(files),2);

    numIterations = length(files);
    bar = ProgressBar(numIterations, 'Title', 'Load HW train Data for SSM');
    for iFile = 1:numIterations
        currFile = files(iFile);
        dataFile = append(path, currFile{:});
        data = matfile(dataFile);

        oData{iFile,1} = data.t(1, 1:idx_end);

        if toCenter
            oData{iFile,2} = data.y(:, 1:idx_end) - data.y_eq';
        else
            oData{iFile,2} = data.y(:, 1:idx_end);
        end
        
        bar(1, [], []);
    end
    bar.release();

end

