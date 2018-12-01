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
mkdir(outfold); %Create new output folder

%% Read Image
im = imread(inppath); 
figure;
imshow(im);
title('Original Image');

%% Convert to grayscale and resize
im = rgb2gray(im);
[nrows, ncols] = size(im); %Image size
nrows = nrows/10; 
ncols = ncols/10;
% r = nrows/ncols; %Ratio
im = imresize(im, [nrows, ncols]); %Reduce to 1/10th
% im = imresize(im, [nrows/r, ncols*r]); %Reduce to 1/10th
figure;
imshow(im);
title('Grayscaled and Resized Image');

%% Gaussian Filter to remove salt and pepper noise
% im = imgaussfilt(im, 1);
% figure;
% imshow(im);

%% Adaptive Thresholding and Binarization
T = adaptthresh(im, 0.3, 'ForegroundPolarity','dark'); %Adaptive Thresholding
figure;
imagesc(T);
im = imbinarize(im,T); %Binarization
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
% [rgn, n] = bwlabel(im);
% figure;
% imagesc(rgn); axis equal;
% title('Image Regions');
% outfil = 'ImageRegions.jpg';
% outpath = fullfile(outfold, outfil);
% imwrite(rgn, outpath);

%% Remove noise
CC = bwconncomp(im);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
BW2 = ismember(L, find([S.Area] >= 50*nrows/100)); %Keep only those regions which are of area more than specified
% bw = bwareaopen(im, 50);
figure;
imshow(BW2);
title('Image Cleaned');
outfil = 'CleanedImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(BW2, outpath);

%% Image Fill
imf = imfill(BW2, 'holes');
figure;
imshow(imf);
title('Filled Image');
outfil = 'FilledImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(imf, outpath);

%% Edge Detection
edge_BW = edge(imf, 'zerocross');
figure;
imshow(edge_BW);
title('Edge Detection');
outfil = 'EdgeDetection.jpg';
outpath = fullfile(outfold, outfil);
imwrite(edge_BW, outpath);

%% Hough Transform (In-built)
% angle = horizon(imf);
% rot = imrotate(imf, angle);
% figure;
% imshow(rot);

%% Hough Transform
[H,theta,rho] = hough(edge_BW); %Hough Transform
peaks = houghpeaks(H, 100); %Peaks in hough transform

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

%% Find Best Angle and rotate
lines = houghlines(edge_BW, theta, rho, peaks);

best_angle = mode([lines.theta])+90; %Most common angle among all lines
im_rot = imrotate(BW2, best_angle); %Rotate wrt angle 
imf_rot = imrotate(imf, best_angle);
figure;
imshow(im_rot);
title('Rotated Image');
outfil = 'RotatedImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(im_rot, outpath);

figure;
imshow(imf_rot);
title('Filled Image');
outfil = 'RotatedFilledImage.jpg';
outpath = fullfile(outfold, outfil);
imwrite(imf_rot, outpath);

%% Decomposition into arrows and shapes

se = strel('diamond', 5); %Diamond shaped kernel
filt = imopen(imf_rot, se); %Open Filter
bw = bwareaopen(filt, 50); %Remove arrows
% figure, imshow(bw);
arrs = im_rot - bw; %Remove shapes (Only arrows)
arrs = imbinarize(arrs);

bw3 = bwareaopen(arrs, 20); %Remove noise
figure, imshow(bw3);
title('Only Arrows');
outfil = 'Arrows.jpg';
outpath = fullfile(outfold, outfil);
imwrite(bw3, outpath);

shps = im_rot - bw3; %Only shapes
figure;
imshow(shps);
title('Only Shapes');
outfil = 'Shapes.jpg';
outpath = fullfile(outfold, outfil);
imwrite(shps, outpath);



%%
[labelledIm,n] = bwlabel(shps);
figure; imagesc(labelledIm); axis equal;

% coloredLabels = label2rgb(labelledIm,'hsv');
% figure; imshow(coloredLabels);

%%
% [centers,radii] = imfindcircles(labelledIm,[20 25],'ObjectPolarity','dark')

%%

stats= regionprops(labelledIm, 'all');

Centroid = cat(1, stats.Centroid);
Perimeter = cat(1,stats.Perimeter);
Area = cat(1,stats.ConvexArea);

CircleMetric = (Perimeter.^2)./(4*pi*Area);  %circularity metric
RectangleMetric = NaN(n,1);
% figure; imshow(stats.Image);

% for every blob
for i = 1: n
    [p,q] = size(stats(i).FilledImage);
    RectangleMetric(i) = Area(i)/(p*q);
    figure;imshow(stats(i).FilledImage);
    
end

isCircle = (CircleMetric < 1.1);
isRectangle = (RectangleMetric > 0.75);
isRectangle = isRectangle .* ~isCircle;
isDiamond = (RectangleMetric <= 0.75);
isDiamond = isDiamond .* ~isCircle;
