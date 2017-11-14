%���ȴ�Python����,��������ֻ��һ��������TotalSoftMax��
load('ToPython.mat','Masked_DataBase_y')
%����ű����������ⲿ���ݵ�SoftMaxֵ
%����ű����Ƴ�����ÿһ��������ǰ5��SoftMaxֵ
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




