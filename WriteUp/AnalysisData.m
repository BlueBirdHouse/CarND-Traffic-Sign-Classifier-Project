%首先检查各数据的占比是否一致
load('FromPython_Train.mat','y_test','y_train','y_valid');
ClassNumList_train = zeros(43,1);
ClassNumList_test = zeros(43,1);
ClassNumList_valid = zeros(43,1);

for i = 1:1:43
    ClassNum = FindClassNumber(y_train,i-1);
    ClassNumList_train(i,1) = ClassNum;
end
for i = 1:1:43
    ClassNum = FindClassNumber(y_valid,i-1);
    ClassNumList_valid(i,1) = ClassNum;
end
for i = 1:1:43
    ClassNum = FindClassNumber(y_test,i-1);
    ClassNumList_test(i,1) = ClassNum;
end

%subplot(2,2,1);
figure(1);
pie(ClassNumList_train)
title('Potencies of Sets for Train');

%subplot(2,2,2);
figure(2);
pie(ClassNumList_valid)
title('Potencies of Sets for Valid');

%subplot(2,2,3);
figure(3);
pie(ClassNumList_test)
title('Potencies of Sets for Test');

function [ClassNum] = FindClassNumber(DataBase,Class)
    %给出指定类别的个数
    FoundNum = DataBase == Class;
    FoundClass = DataBase(FoundNum,:);
    ClassNum = size(FoundClass,1);
end

function [FoundFigs] = FindFigs(Num,X_train,y_train)
    FoundFigs = y_train == Num;
    FoundFigs = X_train(FoundFigs,:,:,:);
end