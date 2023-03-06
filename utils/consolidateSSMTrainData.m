function oData = consolidateSSMTrainData(path, files, varargin)

    if ~iscell(files)
        files = {files};
    end

    % Parse inputs
    p = inputParser;
    addParameter(p, 'center', true)
    addParameter(p, 'idx_end', 0)
    addParameter(p, 'addInput', false)
    parse(p, varargin{:});
    toCenter = p.Results.center;
    idx_end = p.Results.idx_end;
    addInput = p.Results.addInput;

    oData = cell(length(files),2);

    numIterations = length(files);
    bar = ProgressBar(numIterations, 'Title', 'Load HW train Data for SSM');
    for iFile = 1:numIterations
        currFile = files(iFile);
        dataFile = append(path, currFile{:});
        data = matfile(dataFile);
        sizeData = length(data.t);
        if idx_end ~= 0
            oData{iFile,1} = data.t(1, 1:idx_end);
        else
            oData{iFile,1} = data.t(1, 1:sizeData);
        end

        if toCenter
            if idx_end ~= 0
                oData{iFile,2} = data.y(:, 1:idx_end) - data.y_eq';
            else
                oData{iFile,2} = data.y(:, 1:sizeData) - data.y_eq';
            end
        else
            if idx_end ~= 0
                oData{iFile,2} = data.y(:, 1:idx_end);
                if addInput
                    oData{iFile,3} = data.t(:, 1:idx_end);
                    oData{iFile,4} = data.u(:, 1:idx_end);
                end
            else
                oData{iFile,2} = data.y(:, 1:sizeData);
                if addInput
                    oData{iFile,3} = data.t(:, 1:sizeData);
                    oData{iFile,4} = data.u(:, 1:sizeData);
                end
            end
        end
        
        bar(1, [], []);
    end
    bar.release();

end

