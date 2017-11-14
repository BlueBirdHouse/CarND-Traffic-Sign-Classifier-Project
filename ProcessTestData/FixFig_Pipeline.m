%ʵ����������ͼƬ���߼�
%Ŀ����ʹͼƬ��ø�����
run('.\StandardColor\StandardColorMake.m');
load('FromPython.mat');

DataBase_Fix = X_test;
n = size(DataBase_Fix,1);
%F = figure();
for i = 1:1:n
    AFig = PickAFig(DataBase_Fix,i);
    AFig_ch = FixFig(AFig);
    %CompareFigs(AFig,AFig_ch);
    %���洦����
    DataBase_Fix(i,:,:,:) = shiftdim(AFig_ch,-1);
    %imshow(AFig_ch);
    disp(i);
    %drawnow;
    %pause(0.1);
end
%���û���õı���
clearvars -except DataBase_Fix STD_Color nColors y_test
save('FixFig');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig ָ����ʾ������ͼƬ
AFig = Figs(NumFig,:,:,:);
AFig = squeeze(AFig);
%imtool(AFig)
end

function [Fig_Out] = FixFig(Fig_in)
%FixFig ʵ��ͼ����������߼�
%RGB��ɫƽ��ʹ���㷨
% Cheng, Dongliang, Dilip K. Prasad, and Michael S. Brown. 
%"Illuminant estimation for color constancy: why spatial-domain methods work 
% and the role of the color distribution." JOSA A 31.5 (2014): 1049-1058.
%���ƻ�������
illuminan = illumpca(Fig_in, 3.5);
Fig_in = chromadapt(Fig_in, illuminan, 'ColorSpace', 'linear-rgb');

%�ֲ��Աȶ���ǿ
Fig_in = localcontrast(Fig_in);

Fig_Out = Fig_in;
end

function [] = CompareFigs(FigA,FigB)
%���ڱȽ��޸��Ժ��ͼ��
%����ͼ�����sRGBת����Ȼ������ʾ
FigA = lin2rgb(FigA);
FigB = lin2rgb(FigB);
%figure();
imshowpair(FigA,FigB,'montage');
end
