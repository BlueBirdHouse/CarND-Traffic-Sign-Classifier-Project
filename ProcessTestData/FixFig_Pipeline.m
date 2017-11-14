%实现连续处理图片的逻辑
%目的是使图片变得更清晰
run('.\StandardColor\StandardColorMake.m');
load('FromPython.mat');

DataBase_Fix = X_test;
n = size(DataBase_Fix,1);
%F = figure();
for i = 1:1:n
    AFig = PickAFig(DataBase_Fix,i);
    AFig_ch = FixFig(AFig);
    %CompareFigs(AFig,AFig_ch);
    %保存处理结果
    DataBase_Fix(i,:,:,:) = shiftdim(AFig_ch,-1);
    %imshow(AFig_ch);
    disp(i);
    %drawnow;
    %pause(0.1);
end
%清除没有用的变量
clearvars -except DataBase_Fix STD_Color nColors y_test
save('FixFig');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig 指定显示并导出图片
AFig = Figs(NumFig,:,:,:);
AFig = squeeze(AFig);
%imtool(AFig)
end

function [Fig_Out] = FixFig(Fig_in)
%FixFig 实现图像整体调整逻辑
%RGB颜色平衡使用算法
% Cheng, Dongliang, Dilip K. Prasad, and Michael S. Brown. 
%"Illuminant estimation for color constancy: why spatial-domain methods work 
% and the role of the color distribution." JOSA A 31.5 (2014): 1049-1058.
%估计环境光照
illuminan = illumpca(Fig_in, 3.5);
Fig_in = chromadapt(Fig_in, illuminan, 'ColorSpace', 'linear-rgb');

%局部对比度增强
Fig_in = localcontrast(Fig_in);

Fig_Out = Fig_in;
end

function [] = CompareFigs(FigA,FigB)
%用于比较修复以后的图像
%两个图像均做sRGB转换，然后并排显示
FigA = lin2rgb(FigA);
FigB = lin2rgb(FigB);
%figure();
imshowpair(FigA,FigB,'montage');
end
