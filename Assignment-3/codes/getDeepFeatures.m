% DO NOT CHANGE THIS CODE
function feat = getDeepFeatures(v_idx)

% This code runs your python script and load data

command = sprintf('source activate torchreid; python extractDeepFeatures.py %d', v_idx);
[status, result] = system(command);
result;
feat = load('./feat.mat');
feat = feat.feat;
