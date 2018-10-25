%% Init
close all
clc
clear

%% Initialize files and folders
n = '6'; %Input Image Number
inpfil = strcat(n, '.jpg'); %Input File
inpfold = './Inputs'; %Input Folder
outfold = strcat('./Outputs/', n); %Output Folder
inppath = fullfile(inpfold, inpfil); %Input Path 
mkdir(outfold);

%% Read Image
im = imread(inppath);
figure;
imshow(im);
title('Original Image');

%% Convert to grayscale and resize
im = rgb2gray(im);
[nrows, ncols] = size(im);
nrows = nrows/10;
ncols = ncols/10;
% r = nrows/ncols;
im = imresize(im, [nrows, ncols]);
figure;
imshow(im);
title('Grayscaled and Resized Image');

%% Gaussian Filter to remove salt and pepper noise
% im = imgaussfilt(im, 1);
% figure;
% imshow(im);

%% Adaptive Thresholding
T = adaptthresh(im, 0.3, 'ForegroundPolarity','dark');
figure;
imagesc(T);
im = imbinarize(im,T);
figure;
imshow(im);
title('Binary Image');
outfil = 'BinaryImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Bitwise Inversion
im = 1-im;
figure;
imshow(im);
title('Inverted Image');
outfil = 'InvertedImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Display diff regions in image
[rgn, n] = bwlabel(im);
figure;
imagesc(rgn); axis equal;
title('Image Regions');
outfil = 'ImageRegions.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Remove noise
CC = bwconncomp(im);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
BW2 = ismember(L, find([S.Area] >= 50*nrows/100));
% bw = bwareaopen(im, 50);
figure;
imshow(BW2);
title('Image Cleaned');
outfil = 'CleanedImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Image Fill
imf = imfill(BW2, 'holes');
figure;
imshow(imf);
title('Filled Image');
outfil = 'FilledImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Edge Detection
edge_BW = edge(imf, 'zerocross');
figure;
imshow(edge_BW);
title('Edge Detection');
outfil = 'EdgeDetection.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Hough Transform (In-built)
% angle = horizon(imf);
% rot = imrotate(imf, angle);
% figure;
% imshow(rot);

%% Hough Transform
[H,theta,rho] = hough(edge_BW);
peaks = houghpeaks(H, 100);

figure;
imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)),rho(peaks(:,1)),'s','color','white');
colormap(gca, hot);
title('Hough Transform Plot');

lines = houghlines(edge_BW,theta,rho,peaks,'FillGap',5,'MinLength',2);
figure, imshow(BW2), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
title('Detected Lines');

lines = houghlines(edge_BW, theta, rho, peaks);

best_angle = mode([lines.theta])+90;
im_rot = imrotate(BW2, best_angle);
imf_rot = imrotate(imf, best_angle);
figure;
imshow(im_rot);
title('Rotated Image');
outfil = 'RotatedImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

figure;
imshow(imf_rot);
title('Filled Image');
outfil = 'RotatedFilledImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

%% Decomposition

se = strel('diamond', 5);
eroded = imopen(imf_rot, se);
% figure;
% imshow(eroded);
% title('Remove Arrows');

bw = bwareaopen(eroded, 50);
% figure, imshow(bw);

shps = im_rot - bw;
% figure, imshow(shps);
shps = imbinarize(shps);
bw3 = bwareaopen(shps, 20);
figure, imshow(bw3);
title('Only Arrows');
outfil = 'Arrows.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);

shps = im_rot - bw3;
figure;
imshow(shps);
title('Only Shapes');
outfil = 'Shapes.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im, outpath);