%目标收缩第三种方案，分水岭分割方法。
%这是最后一种方法，所有没有成功的图片均会被增加一个默认的圆形Mask。以保证Mask形状
%本身不会干扰学习过程。
%这个过程不会改变数据库中数据的顺序。
%成功失败的比率：622：539

load('Scope2.mat');

%设置成功数据库。
DataBase_success3 = uint8(zeros(1,32,32,3));
DataBase_success3_y = DataBase_No_y;

Counter_DataBase_success3 = 1;
Counter_DataBase_No3 = 1;

%调入要处理的数据库
DataBase = DataBase_No;
n = size(DataBase,1);
for i = 1:1:n
    AFig = PickAFig(DataBase,i);
    %利用分水岭方法做边界检查。
    AFig_gray = rgb2gray(AFig);
    %% 标记前景色
    %将前景色的渐变涂抹掉。
    Se = strel('disk',2);

    %腐蚀以后要重建图片。这个过程需要一个腐蚀以后的图片
    AFig_Rode = imerode(AFig_gray,Se);
    AFig_Rode_Construct = imreconstruct(AFig_Rode,AFig_gray);
    
    AFig_Dilate = imdilate(AFig_Rode_Construct,Se);
    AFig_Dilate_Construct = imreconstruct(imcomplement(AFig_Dilate),imcomplement(AFig_Rode_Construct));
    AFig_Dilate_Construct = imcomplement(AFig_Dilate_Construct);
    
    %计算区域内部的最大值
    fgm = imregionalmax(AFig_Dilate_Construct);
    %fgm = imopen(fgm,Se);
    %fgm = imerode(fgm,Se);
    
    I2 = AFig_gray;
    I2(fgm) = 255;
    
    bw = imbinarize(AFig_Dilate_Construct);
    D = bwdist(bw);
    DL = watershed(D);
    bgm = DL == 0;
    
    hy = fspecial('sobel');
    Iy = imfilter(double(AFig_gray),hy,'replicate');
    Ix = imfilter(double(AFig_gray),hy','replicate');
    gradmag = sqrt(Ix.^2 + Iy.^2);
    gradmag = imimposemin(gradmag, bgm | fgm);
    L = watershed(gradmag);
    
    %找到中间的部分的标记符号当作交通标志的分隔标号
    Middle = [L(16,16:17) L(17,16:17)];
    Middle = mode(Middle);
    %找到分类结果里面对应的标记范围
    [row,col] = find(L == Middle); 
    [radii,centers,~] = ExactMinBoundCircle([row  col]);
    %生成蒙版
    [Mask] = CircleMask(centers,radii+4);
    Mask = imclearborder(Mask,4);
    RGBmask(:,:,1) = Mask;
    RGBmask(:,:,2) = Mask;
    RGBmask(:,:,3) = Mask;
    if(sum(sum(Mask)) > 0)
        AFig_Masked = immultiply(AFig,RGBmask);
        DataBase_success3(Counter_DataBase_success3,:,:,:) = shiftdim(AFig_Masked,-1);
        Counter_DataBase_success3 = Counter_DataBase_success3 + 1;
    else
        %算法失败，增加一个默认的Mask
        [Mask] = CircleMask([16 16],13);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        DataBase_success3(Counter_DataBase_success3,:,:,:) = shiftdim(AFig_Masked,-1);
        Counter_DataBase_success3 = Counter_DataBase_success3 + 1;
        Counter_DataBase_No3 = Counter_DataBase_No3 + 1;
    end
    imshow(AFig_Masked);
    drawnow;
    %pause(1);
    disp(i);
end
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_success DataBase_success_y DataBase_success3 DataBase_success3_y Counter_DataBase_No3
save('Scope3.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig 指定显示并导出图片
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [] = WatershedSeg(Fig)
    
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