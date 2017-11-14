%目标收缩第二种种方案：使用聚类的方法判断形状。
%本方案在第一种方案基失败础上成功：失败是：4378:1315

load('Scope.mat');

%设置成功和失败的数据库
DataBase_success = uint8(zeros(1,32,32,3));
DataBase_success_y = uint8(zeros(1,1));

DataBase_No = uint8(zeros(1,32,32,3));
DataBase_No_y = uint8(zeros(1,1));

Counter_DataBase_success = 1;
Counter_DataBase_No = 1;

%调入需要处理的数据
DataBase = DataBase_NoCircle;
n = size(DataBase,1);
for i = 1:1:n
    AFig = PickAFig(DataBase,i);
    
    %利用原来的图直接检查有没有圆形
    [centers,radii] = FindCircle(AFig);
    if(~isempty(centers))
        %生成蒙版
        [Mask] = CircleMask(centers,radii);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        %保存结果
        DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
        DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
        Counter_DataBase_success = Counter_DataBase_success + 1;
        imshow(AFig_Masked);
    end
    
    if(isempty(centers))
        %利用三色聚类方法将图片分割为两类
        [BW,~] = segmentImage(AFig);
        BW = ~BW;
        %保存一个没有去掉边角的聚类结果
        BW_Kmean = BW;
        %去掉那些不足5个像素的零散分类
        Se = strel('disk',2);
        BW = imopen(BW,Se);
        %最小的分类不能少于25个像素，保守起见设置为20个
        if(sum(sum(BW)) <= 20)
            BW = false(32,32);
        end
        %首先查找分类的结果类面有没有圆形
        label = BW * 255;
        [centers,radii] = FindCircle(label);
        if(~isempty(centers))
            %生成蒙版
            [Mask] = CircleMask(centers,radii);
            RGBmask(:,:,1) = Mask;
            RGBmask(:,:,2) = Mask;
            RGBmask(:,:,3) = Mask;
            AFig_Masked = immultiply(AFig,RGBmask);
            %保存结果
            DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
            DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
            Counter_DataBase_success = Counter_DataBase_success + 1;
            imshow(AFig_Masked);
            %pause(1);
        end
        if(isempty(centers))
            %如果聚类里面没有发现有圆形，那么就自己画一个圆覆盖最多的聚类
            if(sum(sum(BW_Kmean)) > 86)
                %如果聚类的结果是其他的图像，那么就画一个圆最大限度的包围这个图像
                [row,col] = find(BW_Kmean == 1); 
                [radii,centers,~] = ExactMinBoundCircle([row  col]);
                %生成蒙版
                [Mask] = CircleMask(centers,radii);
                RGBmask(:,:,1) = Mask;
                RGBmask(:,:,2) = Mask;
                RGBmask(:,:,3) = Mask;
                AFig_Masked = immultiply(AFig,RGBmask);
                %保存结果
                DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
                DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
                Counter_DataBase_success = Counter_DataBase_success + 1;
                imshow(AFig_Masked);
            else
                %如果聚类的结果过少，算法失败
                DataBase_No(Counter_DataBase_No,:,:,:) = shiftdim(AFig,-1);
                DataBase_No_y(Counter_DataBase_No,1) = DataBase_NoCircle_y(i,1);
                Counter_DataBase_No = Counter_DataBase_No + 1;
            end
        end
    end
    drawnow;
    pause(1);
    disp(i);
end
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_success DataBase_success_y DataBase_No DataBase_No_y
save('Scope2.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig 指定显示并导出图片
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [BW,maskedImage] = segmentImage(RGB)
%  以下的代码是利用 imageSegmenter App自动生成的。
%  核心是利用聚类的方法做颜色的分割
%  [BW,MASKEDIMAGE] = segmentImage(RGB) segments image RGB using
%  auto-generated code from the imageSegmenter App. The final segmentation
%  is returned in BW, and a masked image is returned in MASKEDIMAGE.

% Convert RGB image into L*a*b* color space.
X = rgb2lab(RGB);

% Auto clustering
sz = size(X);
im = single(reshape(X,sz(1)*sz(2),[]));
im = im - mean(im);
im = im ./ std(im);
s = rng;
rng('default');
L = kmeans(im,2,'Replicates',2);
rng(s);
BW = L == 2;
BW = reshape(BW,[sz(1) sz(2)]);

% Clear borders
BW = imclearborder(BW);

% Fill holes
BW = imfill(BW, 'holes');

% Invert mask
BW = imcomplement(BW);

% Create masked image.
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
end
function [centers,radii] = FindCircle(label)
    radiusRange = [7 13];
    Sensitivity = 0.9;
    [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','bright');
    if(isempty(centers))
        [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','dark');
        if(isempty(centers))
            [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','bright');
            if(isempty(centers))
                [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','dark');
            end
        end
    end
end

function [Mask] = CircleMask(centers,radii)
    %生成的Mask是这些圆形的并集
    %图片的长宽
    FigH = 32;
    FigL = 32;
    %针对半径的偏移量。为了扩大圆形的孔洞。
    Offset = 2;
    n = size(centers);
    Driver = linspace(0, 2*pi, 50);
    Mask = zeros(FigH,FigL);
    for i = 1:1:n
        r = radii(i) + Offset;
        c = centers(i,:);
        AMask = poly2mask(r*cos(Driver)+c(1), r*sin(Driver)+c(2), FigH, FigL);
        Mask = Mask | AMask;
    end
end