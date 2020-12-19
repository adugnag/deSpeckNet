function net = deSpeckNet_Init()

% Create DAGNN object
net = dagnn.DagNN();

% conv + relu
blockNum = 1;
inVar = 'input';
channel= 1; % grayscale image
dims   = [3,3,channel,64];
pad    = [1,1];
dilate = [1,1];
stride = [1,1];
lr     = [1,1];
%FCN clean
[net, inVar1, blockNum] = addConv(net, blockNum, inVar, dims, pad,dilate, stride, lr);
[net, inVar1, blockNum] = addReLU(net, blockNum, inVar1);

for i = 1:15
    % conv + bn + relu
    dims0   = [3,3,64,64];
    [net, inVar1, blockNum] = addConv(net, blockNum, inVar1, dims0, pad,dilate, stride, lr);
    n_ch   = dims0(4);
    [net, inVar1, blockNum] = addBnorm(net, blockNum, inVar1, n_ch);
    [net, inVar1, blockNum] = addReLU(net, blockNum, inVar1);
end

% conv
dims1   = [3,3,64,channel];
[net, inVar5, blockNum] = addConv(net, blockNum, inVar1, dims1, pad,dilate, stride, lr);

%__________________________________________________________________________
%FCN noise

[net, inVar8, blockNum] = addConv(net, blockNum, inVar, dims, pad,dilate, stride, lr);
[net, inVar8, blockNum] = addReLU(net, blockNum, inVar8);

for i = 1:15
    % conv + bn + relu
    [net, inVar8, blockNum] = addConv(net, blockNum, inVar8, dims0, pad,dilate, stride, lr);
    n_ch   = dims0(4);
    [net, inVar8, blockNum] = addBnorm(net, blockNum, inVar8, n_ch);
    [net, inVar8, blockNum] = addReLU(net, blockNum, inVar8);
end

% conv
[net, inVar13, blockNum] = addConv(net, blockNum, inVar8, dims1, pad,dilate, stride, lr);


% % % Multiply and reconstruct noisy image
inVarr = {inVar13,inVar5};
[net, inVar30, blockNum] = addMultiply(net, blockNum, inVarr); 
% 

outputName = 'prediction'; 
net.renameVar(inVar5,outputName)

%__________________________________________________________________________


% loss clean
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;


outputName1 = 'prediction1';  %Final noisy image reconstruction
net.renameVar(inVar30,outputName1)

% loss noisy
net.addLayer('loss1', dagnn.Loss('loss','L2'), {'prediction1','input'}, {'objective1'},{});
net.vars(net.getVarIndex('prediction1')).precious = 1;

end


% Add a multiply layer
function [net, inVar, blockNum] = addMultiply(net, blockNum, inVar)

outVar   = sprintf('mult%d', blockNum);
layerCur = sprintf('mult%d', blockNum);

block    = dagnn.Multiply();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, dilate, stride, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad, 'dilate', dilate, 'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end
