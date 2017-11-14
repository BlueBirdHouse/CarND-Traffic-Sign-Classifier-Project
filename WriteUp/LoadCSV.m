%% 导入文本文件中的数据。
% 用于从以下文本文件导入数据的脚本:
%
%    signnames.csv
%
% 要将代码扩展到其他选定数据或其他文本文件，请生成函数来代替脚本。

% 由 MATLAB 自动生成于 2017/11/11 22:06:59

%% 初始化变量。
filename = 'signnames.csv';
delimiter = ',';
startRow = 2;

%% 每个文本行的格式:
%   列1: 双精度值 (%f)
%	列2: 文本 (%s)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%f%s%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r');

%% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% 关闭文本文件。
fclose(fileID);

%% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

%% 创建输出变量
signnames = table(dataArray{1:end-1}, 'VariableNames', {'ClassId','SignName'});

%% 清除临时变量
clearvars filename delimiter startRow formatSpec fileID dataArray ans;