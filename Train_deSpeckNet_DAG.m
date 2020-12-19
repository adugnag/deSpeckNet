%author: Adugna Mullissa

%Description: This script runs deSpeckNet training task on single polarization SAR intensity images. 
% The input image and the label should be stored in the same folder as this script. The default names are 
%Train2 and Test2. This script trains deSpeckNet without TV loss.
% This script is modified from https://github.com/cszn/DnCNN.

clc;
clear all

rng('default')

addpath('utilities');
addpath('../../matlab/matlab');
%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
opts.modelName        = 'deSspeckNet'; % model name
opts.learningRate     = [logspace(-3,-3,25) logspace(-4,-4,25)];%  learning rate
opts.batchSize        = 128; % 
opts.gpus             = 1; %set to [] when using CPU
opts.numSubBatches    = 2;

% solver
opts.solver           = 'Adam'; % global
opts.derOutputs       = {'objective',100, 'objective1',1} ; %Loss weights for Lclean and Lnoisy respectively

opts.backPropDepth    = Inf;
%-------------------------------------------------------------------------
%   Initialize model
%-------------------------------------------------------------------------
net = deSpeckNet_Init();
%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net, info] = deSpecknet_train_dag(net,  ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'numSubBatches',opts.numSubBatches, ...
    'backPropDepth',opts.backPropDepth, ...
    'solver',opts.solver, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






