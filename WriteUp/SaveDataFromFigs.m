%% 保存来自外来下载的图像数据，并将其整理为需要的格式。
Fig_11 = imread('OutsideExample_11.jpg');
Fig_11 = imresize(Fig_11, [26 26]);
Fig_11 = padarray(Fig_11,[3,3],255);
Fig_11 = shiftdim(Fig_11,-1);

Fig_18 = imread('OutsideExample_18.jpg');
Fig_18 = imresize(Fig_18, [26 26]);
Fig_18 = padarray(Fig_18,[3,3],255);
Fig_18 = shiftdim(Fig_18,-1);

Fig_24 = imread('OutsideExample_24.jpg');
Fig_24 = imresize(Fig_24, [26 26]);
Fig_24 = padarray(Fig_24,[3,3],255);
Fig_24 = shiftdim(Fig_24,-1);

Fig_27 = imread('OutsideExample_27.jpg');
Fig_27 = imresize(Fig_27, [26 26]);
Fig_27 = padarray(Fig_27,[3,3],255);
Fig_27 = shiftdim(Fig_27,-1);

Fig_30 = imread('OutsideExample_30.jpg');
Fig_30 = imresize(Fig_30, [26 26]);
Fig_30 = padarray(Fig_30,[3,3],255);
Fig_30 = shiftdim(Fig_30,-1);

X_Figs = [Fig_11 ; Fig_18 ; Fig_24 ; Fig_27 ; Fig_30];
y_Figs = uint8([11;18;24;27;30]);

clearvars -except X_Figs y_Figs
save('OutSideFig.mat');