%连续处理图片逻辑，将交通标志所在位置凸显出来
%利用标准颜色做颜色分割，然后从中找到圆形
%第一种方案的成功：失败测试1是：17805 ：16994
%第一种方案的成功：失败测试2是：17805 ：16994
%说明采用的方法没有随机性。

load('FixFig.mat');

DataBase = DataBase_Fix;
n = size(DataBase,1);

%这个程序将会生成两个数据库，一个是找到了圆形的数据库，其中除了圆形的部分都被蒙版去掉了
%一个是找不到圆形的数据库
DataBase_Circle = uint8(zeros(1,32,32,3));
DataBase_Circle_y = uint8(zeros(1,1));
DataBase_NoCircle = uint8(zeros(1,32,32,3));
DataBase_NoCircle_y = uint8(zeros(1,1));
Counter_DataBase_Circle = 1;
Counter_DataBase_NoCircle = 1;

%% 利用颜色分割方法
%F = figure();
for i = 1:n
    AFig = PickAFig(DataBase,i);
    %转换到LAB空间，并找出表示颜色的两位
    lab_AFig = rgb2lab(AFig);
    a = lab_AFig(:,:,2);
    b = lab_AFig(:,:,3);
    distance = zeros([size(a), nColors]);
    for count = 1:nColors
        distance(:,:,count) = ( (a - STD_Color(count,1)).^2 + ...
        (b - STD_Color(count,2)).^2 ).^0.5;
    end
    [~, label] = min(distance,[],3);
    %利用调查的结果，绘制一个灰度图像
    color_labels = uint8([0 127 255]);
    label = color_labels(label);
    %做圆圈查找
    %使用方法1
    radiusRange = [7 13];
    Sensitivity = 0.9;
    [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','bright');
    if(isempty(centers))
        [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','dark');
        if(isempty(centers))
            [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','bright');
            if(isempty(centers))
                [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','dark');
                if(isempty(centers))
                    %如果找不到圆圈，则归为一个找不到数据库
                    DataBase_NoCircle(Counter_DataBase_NoCircle,:,:,:) = shiftdim(AFig,-1);
                    DataBase_NoCircle_y(Counter_DataBase_NoCircle,1) = y_train(i,1);
                    Counter_DataBase_NoCircle = Counter_DataBase_NoCircle + 1;
                end
            end
        end
    end
    if(~isempty(centers))
        %生成蒙版
        [Mask] = CircleMask(centers,radii);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        %保存结果
        DataBase_Circle(Counter_DataBase_Circle,:,:,:) = shiftdim(AFig_Masked,-1);
        DataBase_Circle_y(Counter_DataBase_Circle,1) = y_train(i,1);
        Counter_DataBase_Circle = Counter_DataBase_Circle + 1;
        
        %绘图比较
        %pause(0.1);
        disp(i);
        %delete(h);
    end
end

%清除没有用的变量
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_NoCircle DataBase_NoCircle_y
save('Scope.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig 指定显示并导出图片
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [] = CompareFigs(FigA,FigB)
%用于比较修复以后的图像
%两个图像均做sRGB转换，然后并排显示
    FigA = lin2rgb(FigA);
    FigB = lin2rgb(FigB);
    %figure();
    imshowpair(FigA,FigB,'montage');
end

function [] = CompareFigs2(FigA,centers,radii,FigB)
%一次性比较三个图
%FigA：原图+圆圈
%FigB：颜色识别以后的东西
    subplot(1,2,1);
    FigA = lin2rgb(FigA);
    imshow(FigA);
    if(~isempty(centers))
        viscircles(centers,radii);
    end
    subplot(1,2,2);
    imagesc(FigB);
    colorbar;
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