%% ��������Python�����ݣ�����������Ϊ��Ҫ�ĸ�ʽ��
%y_test = y_test';
%y_train = y_train';
y_valid = y_valid';

clearvars -except y_valid X_valid
save('FromPython.mat');