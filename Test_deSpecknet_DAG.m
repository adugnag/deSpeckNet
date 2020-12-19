%author: Adugna Mullissa

%Description: This script tests a trained deSpeckNet model using single polarization SAR intensity image. 
% This script is modified from https://github.com/cszn/DnCNN.

clc;
clear all

%% testing set
addpath(fullfile('utilities'));

folderModel = 'model';
gpu         = [];

load('./data/model/xxxxx.mat'); %update path accordingly

net = dagnn.DagNN.loadobj(net) ;

out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;
net.mode = 'test';

Img = imread('./xxxx.tif'); %The image should be a grayscale image
Img = im2single(Img);
   
tic
net.eval({'input', input}) ;
output1 = gather(squeeze(gather(net.vars(out1).value)));
toc