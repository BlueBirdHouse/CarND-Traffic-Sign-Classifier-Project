run('LoadCSV.m')
load('TotalSoftMax.mat');
%����ű����Ƴ�����ÿһ��������ǰ5��SoftMaxֵ
for sampleType = 1:43
    SoftMaxList = TotalSoftMax(sampleType,:);
    [Sorted_SoftMax,I] = sort(SoftMaxList,'descend');
    Top_5_Sorted_SoftMax = Sorted_SoftMax(1:5);
    Top_5_I = categorical(I(1:5) - 1);
    subplot(6,8,sampleType)
    bar(Top_5_I,Top_5_Sorted_SoftMax);
    title(num2str(sampleType - 1));
end
%�ҵ�ǰ5�������׻����ķ���
%�ҵ�ÿһ���е����ֵ��Ȼ��ȴ�С.
Max_SoftMax = max(TotalSoftMax,[],1);
[~,I] = sort(Max_SoftMax);
ID = I -1;

%% ��ͼ�Ͽ��Կ�����27��11��18��24��30�����������׻���
%������ЩͼƬ��ѵ������ʾ��
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




