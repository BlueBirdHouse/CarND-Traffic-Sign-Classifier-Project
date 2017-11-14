%% 本文件综合Matlab得到的结论，并将其传输给Python
load('Scope3.mat');
Masked_DataBase = [DataBase_Circle ; DataBase_success ; DataBase_success3];
Masked_DataBase_y = [DataBase_Circle_y ; DataBase_success_y ; DataBase_success3_y];

%去掉最后的空分类
Masked_DataBase = Masked_DataBase(1:5,:,:,:);
Masked_DataBase_y = Masked_DataBase_y(1:5);

clearvars -except Masked_DataBase Masked_DataBase_y
save('ToPython.mat');

