%% 保存来自Python的数据，并将其整理为需要的格式。
y_test = y_test';
y_train = y_train';
y_valid = y_valid';

clearvars -except y_test y_train y_valid X_test X_train X_valid
save('FromPython.mat');