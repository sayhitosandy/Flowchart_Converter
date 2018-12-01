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

stats = regionprops(labelledIm, 'all');

Centroid = cat(1, stats.Centroid);
Perimeter = cat(1,stats.Perimeter);
Area = cat(1,stats.ConvexArea);

CircleMetric = (Perimeter.^2)./(4*pi*Area);  %circularity metric
RectangleMetric = NaN(n,1);
% figure; imshow(stats.Image);

% for every blob
for i = 1:n
    [p,q] = size(stats(i).FilledImage);
    RectangleMetric(i) = Area(i)/(p*q);
    figure;imshow(stats(i).FilledImage);
    
end

isCircle = (CircleMetric < 1.1);
isRectangle = (RectangleMetric > 0.75);
isRectangle = isRectangle .* ~isCircle;
isDiamond = (RectangleMetric <= 0.75);
isDiamond = isDiamond .* ~isCircle;

%%
[labelledArrows,n] = bwlabel(bw3);
figure; imagesc(labelledArrows); axis equal;

%%

arrows = regionprops(labelledArrows, 'all');

Centroid = cat(1, arrows.Centroid);
bb = cat(1, arrows.BoundingBox);
centres = [bb(:, 1) + 0.5*bb(:, 3), bb(:, 2) + 0.5*bb(:, 4)];

figure; imshow(bw3);
hold on;
plot(centres(:, 1), centres(:, 2), 'r*', 'LineWidth', 2, 'MarkerSize', 5);
plot(Centroid(:, 1), Centroid(:, 2), 'b*', 'LineWidth', 2, 'MarkerSize', 5);

midpts = [];
heads = [];
tails = [];

for i = 1: n
%     figure;imshow(arrows(i).FilledImage);    
    hold on;
    orient = arrows(i).Orientation;
    if (abs(abs(orient)-90) > abs(orient))
        midpt = [bb(i, 1), centres(i, 2);  bb(i, 1) + bb(i, 3), centres(i, 2)];
        hold on;
%         plot(midpt(:, 1), midpt(:, 2), 'g*', 'LineWidth', 2, 'MarkerSize', 5);
    else
        midpt = [centres(i, 1), bb(i, 2); centres(i, 1), bb(i, 2) + bb(i, 4)];
        hold on;
%         plot(midpt(:, 1), midpt(:, 2), 'y*', 'LineWidth', 2, 'MarkerSize', 5);
    end
    
    if (pdist([Centroid(i, :); midpt(1, :)], 'euclidean') <= pdist([centres(i, :); midpt(1, :)], 'euclidean'))
        head = midpt(1, :);
        tail = midpt(2, :);
    else
        head = midpt(2, :);
        tail = midpt(1, :);
    end
    plot(head(:, 1), head(:, 2), 'g*', 'LineWidth', 2, 'MarkerSize', 5);
    plot(tail(:, 1), tail(:, 2), 'y*', 'LineWidth', 2, 'MarkerSize', 5);
    midpts = [midpts; midpt];
    heads = [heads; head];
    tails = [tails; tail];
end

%%

mids = [];
for i=1:size(stats,1) %clockwise
    Boxx = stats(i).BoundingBox;

    mid1 = [Boxx(1) + 0.5*Boxx(3), Boxx(2)];
    mid2 = [Boxx(1) + Boxx(3), Boxx(2) + 0.5*Boxx(4)];
    mid3 = [Boxx(1) + 0.5*Boxx(3), Boxx(2) + Boxx(4)];
    mid4 = [Boxx(1), Boxx(2) + 0.5*Boxx(4)];
    mids = [mids; mid1; mid2; mid3; mid4];
end
figure;imshow(shps);
hold on;
plot(mids(:, 1), mids(:, 2), 'r.');

%%
allheads = [];
alltails = [];
for i=1:size(heads, 1)
    distances = [];
    for j=1:size(mids, 1)
        distance = pdist([heads(i, 1), heads(i, 2); mids(j, 1), mids(j, 2)],'euclidean');
        distances = [distances; distance];
    end
    [~, minidx] = min(distances(:));
    hold on;
%     plot(mids(minidx, 1), mids(minidx, 2), 'g*');
%     plot(heads(i, 1), heads(i, 2), 'b*');
    allheads = [allheads;  mids(minidx, :)];
    alltails = [alltails; tails(i, :)];
end
hold on;
plot(allheads(:, 1), allheads(:, 2), 'r*');
plot(alltails(:, 1), alltails(:, 2), 'y*');
%%
