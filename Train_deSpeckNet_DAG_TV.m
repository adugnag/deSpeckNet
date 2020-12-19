%author: Adugna Mullissa

%Description: This script runs deSpeckNet training task on single polarization SAR intensity images. 
% The input image and the label should be stored in the same folder as this script. The default names are 
%Train2 and Test2. 
% This script is modified from https://github.com/cszn/DnCNN.

clc;
clear all

rng('default')

addpath('utilities');
addpath('../../matlab/matlab');
%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
opts.modelName        = 'deSpeckNet'; % model name
opts.learningRate     = [logspace(-3,-3,35) logspace(-4,-4,35)];% Learning rate
opts.batchSize        = 128; % Batch size
opts.gpus             = 1; %set to [] when using CPU
opts.numSubBatches    = 2;

% solver
opts.solver           = 'Adam'; % global
opts.derOutputs       = {'objective',100 ,'objective0',0 , 'objective1',1} ; %Loss weights for Lclean, LTV and Lnoisy respectively.
opts.backPropDepth    = Inf;
%-------------------------------------------------------------------------
%   Initialize model
%-------------------------------------------------------------------------
net = deSpeckNet_Init_TV(); 
%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------
[net, info] = deSpecknet_train_dag_TV(net,  ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'numSubBatches',opts.numSubBatches, ...
    'backPropDepth',opts.backPropDepth, ...
    'solver',opts.solver, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






