%%
close all
clc
clear

%% Read Image
im = imread('input1big.jpg');
figure;
imshow(im);

%% Convert to grayscale and resize
im = rgb2gray(im);
im = imresize(im, [300 300]);
figure;
imshow(im);

%% Gaussian Filter to remove salt and pepper noise
% im = imgaussfilt(im, 1);
% figure;
% imshow(im);

%% Adaptive Thresholding
T = adaptthresh(im, 0.4, 'ForegroundPolarity','dark');
figure;
imagesc(T);
im = imbinarize(im,T);
figure;
imshow(im);

%% Bitwise Inversion
im = 1-im;
figure;
imshow(im);

%% Display diff regions in image
% [rgn, n] = bwlabel(im);
% figure;
% imagesc(rgn); axis equal;

%% Remove noise
CC = bwconncomp(im);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
[rows, cols] = size(im);
BW2 = ismember(L, find([S.Area] >= 50 * (mod(rows, 100)+1)));
% bw = bwareaopen(im, 50);
figure;
imshow(BW2);

%% Image Fill
imf = imfill(BW2, 'holes');
figure;
imshow(imf);

%% Edge Detection
edge_BW = edge(imf, 'zerocross');
figure;
imshow(edge_BW);

%% Hough Transform

[H,theta,rho] = hough(edge_BW);
thetas_cum = sum(H);
h = histogram(thetas_cum)
