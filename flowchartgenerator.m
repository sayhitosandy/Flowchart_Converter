%% Init
close all
clc
clear

%% Initialize files and folders
imgNo = '6'; %Input Image Number
inpFile = strcat(imgNo, '.jpg'); %Input File
inpFolder = './Inputs'; %Input Folder
outFolder = strcat('./Outputs/', imgNo); %Output Folder
inpPath = fullfile(inpFolder, inpFile); %Input Path 
mkdir(outFolder); %Create new output folder

%% Read Image
originalIm = imread(inpPath); 
figure;
imshow(originalIm);
title('Original Image');

%% Convert to grayscale and resize
grayIm = rgb2gray(originalIm);
[nrows, ncols] = size(grayIm); %Image size
nrows = nrows/10; 
ncols = ncols/10;
% r = nrows/ncols; %Ratio
resizedIm = imresize(grayIm, [nrows, ncols]); %Reduce to 1/10th
% resizedIm = imresize(grayIm, [nrows/r, ncols*r]); %Reduce to 1/10th
figure;
imshow(resizedIm);
title('Grayscaled and Resized Image');

%% Gaussian Filter to remove salt and pepper noise
% filteredIm = imgaussfilt(resizedIm, 1);
% figure;
% imshow(filteredIm);

%% Adaptive Thresholding and Binarization
T = adaptthresh(resizedIm, 0.3, 'ForegroundPolarity','dark'); %Adaptive Thresholding
% figure;
% imagesc(T);
binaryIm = imbinarize(resizedIm, T); %Binarization
figure;
imshow(binaryIm);
title('Binary Image');
outFile = 'BinaryImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(binaryIm, outPath);

%% Bitwise Inversion
invertedIm = 1-binaryIm;
figure;
imshow(invertedIm);
title('Inverted Image');
outFile = 'InvertedImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(invertedIm, outPath);

%% Display diff regions in image
% [rgn, n_rgn] = bwlabel(invertedIm);
% figure;
% imagesc(rgn); axis equal;
% title('Image Regions');
% outfil = 'ImageRegions.jpg';
% outpath = fullfile(outfold, outfil);
% imwrite(rgn, outpath);

%% Remove noise
CC = bwconncomp(invertedIm);
S = regionprops(CC, 'Area');
L = labelmatrix(CC);
cleanedIm = ismember(L, find([S.Area] >= 50*nrows/100)); %Keep only those regions which are of area more than specified
% bw = bwareaopen(im, 50);
figure;
imshow(cleanedIm);
title('Image Cleaned');
outFile = 'CleanedImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(cleanedIm, outPath);

%% Image Fill
filledIm = imfill(cleanedIm, 'holes'); % Fill holes in the image
figure;
imshow(filledIm);
title('Filled Image');
outFile = 'FilledImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(filledIm, outPath);

%% Edge Detection
edgeIm = edge(filledIm, 'zerocross'); %Zero cross edge detection
figure;
imshow(edgeIm);
title('Edge Detection');
outFile = 'EdgeDetection.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(edgeIm, outPath);

%% Hough Transform (In-built)
% angle = horizon(filledIm);
% rot = imrotate(filledIm, angle);
% figure;
% imshow(rot);

%% Hough Transform
[H,theta,rho] = hough(edgeIm); %Hough Transform
peaks = houghpeaks(H, 100); %Peaks in hough transform

figure;
imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)),rho(peaks(:,1)),'s','color','white');
colormap(gca, hot);
title('Hough Transform Plot');

lines = houghlines(edgeIm,theta,rho,peaks,'FillGap',5,'MinLength',2); 
figure, imshow(cleanedIm), hold on
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
lines = houghlines(edgeIm, theta, rho, peaks);

bestAngle = mode([lines.theta])+90; %Most common angle among all lines
cleanedRotatedIm = imrotate(cleanedIm, bestAngle); %Rotate wrt angle 
filledRotatedIm = imrotate(filledIm, bestAngle);
figure;
imshow(cleanedRotatedIm);
title('Rotated Image');
outFile = 'RotatedImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(cleanedRotatedIm, outPath);

[nrows, ncols] = size(cleanedRotatedIm);

figure;
imshow(filledRotatedIm);
title('Filled Image');
outFile = 'RotatedFilledImage.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(filledRotatedIm, outPath);

%% Decomposition into arrows and shapes

se = strel('diamond', 5); %Diamond shaped kernel
openedIm = imopen(filledRotatedIm, se); %Open Filter
bwIm = bwareaopen(openedIm, 50); %Remove arrows
% figure, imshow(bwIm);
arrIm = cleanedRotatedIm - bwIm; %Remove shapes (Only arrows)
arrIm = imbinarize(arrIm);

arrowsIm = bwareaopen(arrIm, 20); %Remove noise
figure, imshow(arrowsIm);
title('Only Arrows');
outFile = 'Arrows.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(arrowsIm, outPath);

shapesIm = cleanedRotatedIm - arrowsIm; %Only shapes
figure;
imshow(shapesIm);
title('Only Shapes');
outFile = 'Shapes.jpg';
outPath = fullfile(outFolder, outFile);
imwrite(shapesIm, outPath);


%% Decompose Shapes into Circles, Rectangles and Diamonds

[shapeLabels, n_shapeLabels] = bwlabel(shapesIm); %Find the number of shapes in the image
% figure; imagesc(shapeLabels); axis equal;

shapeProps = regionprops(shapeLabels, 'all'); %Extract all properties of shapes

shapeCentroids = cat(1, shapeProps.Centroid); %Centroid of each shape
shapePerimeters = cat(1, shapeProps.Perimeter); %Perimeter of each shape
shapeArea = cat(1, shapeProps.ConvexArea); %Convex Hull Area of each shape
shapeBBs = cat(1, shapeProps.BoundingBox); %Axis Aligned Bounding Box for each shape

circleAreaRatio = (shapePerimeters.^2)./(4*pi*shapeArea); %Detect circles
rectAreaRatio = NaN(n_shapeLabels,1); %Detect rectangles and diamonds

for i = 1:n_shapeLabels
    [p,q] = size(shapeProps(i).FilledImage); %Area of Bounding Box of each shape
    rectAreaRatio(i) = shapeArea(i)/(p*q);
%     figure; imshow(shapeProps(i).FilledImage);
end

isShapeCircle = (circleAreaRatio < 1.1); % 1 if shape is a Circle
isShapeRect = (rectAreaRatio > 0.75); 
isShapeRect = logical(isShapeRect .* ~isShapeCircle); % 1 if shape is Rectangle
isShapeDiad = (rectAreaRatio <= 0.75);
isShapeDiad = logical(isShapeDiad .* ~isShapeCircle); % 1 if shape is Diamond

%% Find Arrow Orientation, Arrow Head and Arrow Tail

[arrowLabels, n_arrowLabels] = bwlabel(arrowsIm); %Find all arrows in the image
% figure; imagesc(arrowLabels); axis equal;

arrowProps = regionprops(arrowLabels, 'all'); %Extract all properties of each arrow

arrowCentroids = cat(1, arrowProps.Centroid); %Centroid of each arrow
arrowBBs = cat(1, arrowProps.BoundingBox); %Axis Aligned Bounding Box of each arrow
arrowCentres = [arrowBBs(:, 1) + 0.5*arrowBBs(:, 3), arrowBBs(:, 2) + 0.5*arrowBBs(:, 4)]; %Centre of Bounding Box of each arrow

% figure; imshow(arrowsIm);
% hold on;
% plot(arrowCentres(:, 1), arrowCentres(:, 2), 'r*', 'LineWidth', 2, 'MarkerSize', 5);
% plot(arrowCentroids(:, 1), arrowCentroids(:, 2), 'b*', 'LineWidth', 2, 'MarkerSize', 5);

% Find head and tail of the arrow based on its orientation
arrowBBsMidpts = [];
allArrowHeads = []; %Head (x,y) of each arrow
allArrowTails = []; %Tail (x,y) of each arrow

for i = 1:n_arrowLabels
%     hold on;
    arrowOrient = arrowProps(i).Orientation; %Orientation of each arrow in deg.
    if (abs(abs(arrowOrient)-90) > abs(arrowOrient)) %Horizontal arrow
        arrowBBMidpt = [arrowBBs(i, 1), arrowCentres(i, 2);  arrowBBs(i, 1) + arrowBBs(i, 3), arrowCentres(i, 2)];
    else %Vertical arrow
        arrowBBMidpt = [arrowCentres(i, 1), arrowBBs(i, 2); arrowCentres(i, 1), arrowBBs(i, 2) + arrowBBs(i, 4)];
    end
    
    % Arrow head is closer to centroid than centre of the Bounding Box
    if (pdist([arrowCentroids(i, :); arrowBBMidpt(1, :)], 'euclidean') <= pdist([arrowCentres(i, :); arrowBBMidpt(1, :)], 'euclidean'))
        arrowHead = arrowBBMidpt(1, :);
        arrowTail = arrowBBMidpt(2, :);
    else
        arrowHead = arrowBBMidpt(2, :);
        arrowTail = arrowBBMidpt(1, :);
    end
%     plot(arrowHead(:, 1), arrowHead(:, 2), 'g*', 'LineWidth', 2, 'MarkerSize', 5);
%     plot(arrowTail(:, 1), arrowTail(:, 2), 'y*', 'LineWidth', 2, 'MarkerSize', 5);
    arrowBBsMidpts = [arrowBBsMidpts; arrowBBMidpt];
    allArrowHeads = [allArrowHeads; arrowHead];
    allArrowTails = [allArrowTails; arrowTail];
end

%% Find Closest Shape to Arrow Head

% Find mid points of sides of bounding box for each shape (We connect the arrow head here.)
shapeBBsMidpts = [];

for i = 1:n_shapeLabels %clockwise search
    
    shapeBB = shapeProps(i).BoundingBox; %Axis Aligned Bounding Box for each shape

    shapeBBMidpt1 = [shapeBB(1) + 0.5*shapeBB(3), shapeBB(2)];
    shapeBBMidpt2 = [shapeBB(1) + shapeBB(3), shapeBB(2) + 0.5*shapeBB(4)];
    shapeBBMidpt3 = [shapeBB(1) + 0.5*shapeBB(3), shapeBB(2) + shapeBB(4)];
    shapeBBMidpt4 = [shapeBB(1), shapeBB(2) + 0.5*shapeBB(4)];
    shapeBBsMidpts = [shapeBBsMidpts; shapeBBMidpt1; shapeBBMidpt2; shapeBBMidpt3; shapeBBMidpt4];
end

figure;imshow(shapesIm);
hold on;
plot(shapeBBsMidpts(:, 1), shapeBBsMidpts(:, 2), 'r.');

arrowHeads = []; %Final set of arrow heads
arrowTails = []; %Final set of arrow tails

for i = 1:size(allArrowHeads, 1)
    arr_shape_dists = [];
    
    for j = 1:size(shapeBBsMidpts, 1)
        arr_shape_dist = pdist([allArrowHeads(i, 1), allArrowHeads(i, 2); shapeBBsMidpts(j, 1), shapeBBsMidpts(j, 2)],'euclidean');
        arr_shape_dists = [arr_shape_dists; arr_shape_dist];
    end
    
    [~, minidx] = min(arr_shape_dists(:)); %Find shape at min distance from arrow head
%     hold on;
%     plot(mids(minidx, 1), mids(minidx, 2), 'g*');
%     plot(heads(i, 1), heads(i, 2), 'b*');
    arrowHeads = [arrowHeads;  shapeBBsMidpts(minidx, :)];
    arrowTails = [arrowTails; allArrowTails(i, :)];
end

% Plot final arrow heads and tails
% hold on;
% plot(arrowHeads(:, 1), arrowHeads(:, 2), 'r*');
% plot(arrowTails(:, 1), arrowTails(:, 2), 'y*');


%% Plot Circles

finalIm = ones(nrows, ncols);
figure; imshow(finalIm);
circleCentres = shapeCentroids(isShapeCircle,:); %Centre of each circle
circleRadii = shapePerimeters(isShapeCircle,:)./(2*pi); %Radius of each circle
viscircles(circleCentres, circleRadii, 'Color', 'k');

%% Plot arrows

% arrow('Start', arrowTails, 'Stop', arrowHeads, 'BaseAngle', 29, 'TipAngle', 30, 'EdgeColor', 'k', 'FaceColor','k', 'LineWidth', 2);
arrow('Start', arrowTails, 'Stop', arrowHeads, 'Type', 'line', 'LineWidth', 2);

%% Plot Rectangles

rectsBBs = shapeBBs(isShapeRect, :);

for i = 1:size(rectsBBs, 1)
    rectangle('Position', [rectsBBs(i,1) rectsBBs(i,2)...
        rectsBBs(i,3) rectsBBs(i,4)], 'EdgeColor','k',...
    'LineWidth',3);
    hold on;
end

%% Plot Diamonds

hold on;
diadsBBs = shapeBBs(isShapeDiad, :);

for i = 1:size(diadsBBs,1)
    patch([diadsBBs(i,1)+ 0.5*diadsBBs(i,3) diadsBBs(i,1)+diadsBBs(i,3) ...
        diadsBBs(i,1)+0.5*diadsBBs(i,3) diadsBBs(i,1) ],...
        [diadsBBs(i,2) diadsBBs(i,2)+0.5*diadsBBs(i,4) ...
        diadsBBs(i,2)+diadsBBs(i,4) diadsBBs(i,2)+0.5*diadsBBs(i,4) ], 'w', 'EdgeColor', 'k', 'LineWidth',3);
%     plot(p,'EdgeColor','w','LineWidth',3)
    hold on;
end