%�������ݿ��е�����
DataBase = No_4;
n = size(DataBase,1);
for i = 1:1:n
    %Rand_Number = randi([0 n],1);
    Rand_Number = i;
    AFig = PickAFig(DataBase,Rand_Number);
    imshow(AFig);
    drawnow;
    pause(1);
end


function [AFig] = PickAFig(Figs,NumFig)
%PickAFig ָ����ʾ������ͼƬ
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end