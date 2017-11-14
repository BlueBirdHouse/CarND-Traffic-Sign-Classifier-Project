%首先从Python导入,工作区中只有一个变量“TotalSoftMax”
load('ToPython.mat','Masked_DataBase_y')
%这个脚本用来绘制外部数据的SoftMax值
%这个脚本绘制出对于每一类样本，前5的SoftMax值
for sampleType = 1:5
    SoftMaxList = TotalSoftMax(sampleType,:);
    [Sorted_SoftMax,I] = sort(SoftMaxList,'descend');
    Top_5_Sorted_SoftMax = Sorted_SoftMax(1:5);
    Top_5_I = categorical(I(1:5) - 1);
    subplot(2,3,sampleType)
    bar(Top_5_I,Top_5_Sorted_SoftMax);
    title(num2str(Masked_DataBase_y(sampleType)));
end


function [FoundFigs] = FindFigs(Num,X_train,y_train)
    FoundFigs = y_train == Num;
    FoundFigs = X_train(FoundFigs,:,:,:);
end




