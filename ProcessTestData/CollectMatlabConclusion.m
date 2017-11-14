%% 本文件综合Matlab得到的结论，并将其传输给Python
load('Scope3.mat');
Masked_DataBase = [DataBase_Circle ; DataBase_success ; DataBase_success3];
Masked_DataBase_y = [DataBase_Circle_y ; DataBase_success_y ; DataBase_success3_y];
clearvars -except Masked_DataBase Masked_DataBase_y
save('ToPython.mat');

