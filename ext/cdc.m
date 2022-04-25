% copied from https://www.mathworks.com/matlabcentral/fileexchange/56624-change-directory-to-current-file-in-editor

function cdc()
    % Set [path to] Current Location (Change Directory to Current - CDC)
    %
    % DESCRIPTION:
    %   Changes the directory to the path of the current file in editor.
    %
    % SETUP:
    %   Place cdc.m in a directory and run the following line once.
    %
    %       addpath('<path to the directory of cdc.m>'); savepath;
    %
    % Eg:
    %   addpath('/Users/totallySomeoneElse/Documents/MATLAB/'); savepath;
    try
        active_editor = matlab.desktop.editor.getActive;
        file = active_editor.Filename;
        path = fileparts(file);
        cd(path);
    catch e
        error('cdc failed.\n\n%s',e.message);
    end
end