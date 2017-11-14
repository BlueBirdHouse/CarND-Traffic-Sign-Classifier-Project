run('LoadCSV.m')
load('TotalSoftMax.mat');
%这个脚本绘制出对于每一类样本，前5的SoftMax值
for sampleType = 1:43
    SoftMaxList = TotalSoftMax(sampleType,:);
    [Sorted_SoftMax,I] = sort(SoftMaxList,'descend');
    Top_5_Sorted_SoftMax = Sorted_SoftMax(1:5);
    Top_5_I = categorical(I(1:5) - 1);
    subplot(6,8,sampleType)
    bar(Top_5_I,Top_5_Sorted_SoftMax);
    title(num2str(sampleType - 1));
end
%找到前5中最容易混淆的分类
%找到每一类中的最大值，然后比大小.
Max_SoftMax = max(TotalSoftMax,[],1);
[~,I] = sort(Max_SoftMax);
ID = I -1;

%% 从图上可以看出，27，11，18，24，30这五种最容易混淆
%调出这些图片的训练样本示例
load('FromPython_Train.mat','X_train','y_train');
No_27 = FindFigs(27,X_train,y_train);
No_11 = FindFigs(11,X_train,y_train);
No_18 = FindFigs(18,X_train,y_train);
No_24 = FindFigs(24,X_train,y_train);
No_30 = FindFigs(30,X_train,y_train);

No_4 = FindFigs(4,X_train,y_train);

function [FoundFigs] = FindFigs(Num,X_train,y_train)
    FoundFigs = y_train == Num;
    FoundFigs = X_train(FoundFigs,:,:,:);
end




