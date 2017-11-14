%播放数据库中的内容
DataBase = Masked_DataBase;
n = size(DataBase,1);
n = 10;
for i = 1:1:n
    Rand_Number = randi([0 n],1);
    Rand_Number = i;
    AFig = PickAFig(DataBase,Rand_Number);
    imshow(AFig);
    drawnow;
    pause(1);
end


function [AFig] = PickAFig(Figs,NumFig)
%PickAFig 指定显示并导出图片
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end