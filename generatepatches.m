function [imdb] = generatepatches

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('utilities');
batchSize     = 128;        % batch size
folder_train  = 'Train2';  %
folder_test = 'Test2';
nchannel      = 1;           % number of channels
patchsize     = 40;

stride        = 9;

step1         = 2;
step2         = 4;
count         = 0;
ext           =  {'*.tif' };
filepaths     =  [];

for i = 1 : length(ext)
    filepath_train = cat(1,filepaths, dir(fullfile(folder_train, ext{i})));
    filepath_test = cat(1,filepaths, dir(fullfile(folder_test, ext{i})));
end

% count the number of extracted patches
scales  = [1 0.9 0.8 0.7]; % scale the image to augment the training data

for i = 1 : length(filepath_train)
    
    image_train = imread(fullfile(folder_train,filepath_train(i).name)); % read train
    image_test = imread(fullfile(folder_test,filepath_test(i).name)); % read label
    if size(image_train,3)==3
        image_train = rgb2gray(image_train);
        image_test = rgb2gray(image_test);
    end
    %[~, name, exte] = fileparts(filepaths(i).name);
    if mod(i,100)==0
        disp([i,length(filepath_train)]);
        disp([i,length(filepath_test)]);
    end
    for s = 1:4
        image_train = imresize(image_train,scales(s),'bicubic');
        image_test = imresize(image_test,scales(s),'bicubic');
        [hei,wid,~] = size(image_train);
        for x = 1+step1 : stride : (hei-patchsize+1)
            for y = 1+step2 :stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches  = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([int2str(numPatches),' = ',int2str(numPatches/batchSize),' X ', int2str(batchSize)]);


count = 0;
imdb.inputs  = zeros(patchsize, patchsize, nchannel, numPatches,'single');
imdb.labels  = zeros(patchsize, patchsize, nchannel, numPatches,'single');

for i = 1 : length(filepath_train)
    
    image_train = imread(fullfile(folder_train,filepath_train(i).name)); % read train
    image_test = imread(fullfile(folder_test,filepath_test(i).name)); % read label
    %[~, name, exte] = fileparts(filepaths(i).name);
    if size(image_train,3)==3
        image_train = rgb2gray(image_train);
        image_test = rgb2gray(image_test);
    end
    if mod(i,100)==0
        disp([i,length(filepath_train)]);
        disp([i,length(filepath_test)]);
    end
    for s = 1:4
        image_train = imresize(image_train,scales(s),'bicubic');
        image_test = imresize(image_test,scales(s),'bicubic');
        for j = 1:1
            %image_aug   = data_augmentation(image, j);  % augment data
%             im_label    = im2single(image_aug);         % single
              im_input    = im2single(image_train); % single
              im_label    = im2single(image_test);
            [hei,wid,~] = size(im_input);
            
            for x = 1+step1 : stride : (hei-patchsize+1)
                for y = 1+step2 :stride : (wid-patchsize+1)
                    count       = count+1;
                    imdb.inputs(:, :, :, count)   = im_input(x : x+patchsize-1, y : y+patchsize-1,:);
                    imdb.labels(:, :, :, count)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                    if count<=diffPatches
                        imdb.inputs(:, :, :, end-count+1)   = im_input(x : x+patchsize-1, y : y+patchsize-1,:);
                        imdb.labels(:, :, :, end-count+1)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                    end
                end
            end
        end
    end
end

imdb.set    = uint8(ones(1,size(imdb.inputs,4)));

