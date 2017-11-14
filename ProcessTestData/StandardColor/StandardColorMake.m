%本文件产生标准交通标志颜色
load('StandardColor');
%STD_Black = MeanColor(Black_Fig);
STD_Blue = MeanColor(Blue_Fig);
STD_Red = MeanColor(Red_Fig);
%不能区分白色和黑色，因为这样中颜色的色度向量是一样的[0 0]
%STD_White = [0 0];
%STD_White = MeanColor(White_Fig);
STD_Yellow = MeanColor(Yellow_Fig);

nColors = 3;

STD_Color = [STD_Red ; STD_Blue ; STD_Yellow ];
clearvars -except STD_Color nColors

function [Out] = MeanColor(In)
    In = rgb2lab(In);
    a = mean2(In(:,:,2));
    b = mean2(In(:,:,3));
    Out = [a  b];
    
end