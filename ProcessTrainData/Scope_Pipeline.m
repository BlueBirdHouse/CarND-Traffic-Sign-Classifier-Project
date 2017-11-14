%��������ͼƬ�߼�������ͨ��־����λ��͹�Գ���
%���ñ�׼��ɫ����ɫ�ָȻ������ҵ�Բ��
%��һ�ַ����ĳɹ���ʧ�ܲ���1�ǣ�17805 ��16994
%��һ�ַ����ĳɹ���ʧ�ܲ���2�ǣ�17805 ��16994
%˵�����õķ���û������ԡ�

load('FixFig.mat');

DataBase = DataBase_Fix;
n = size(DataBase,1);

%������򽫻������������ݿ⣬һ�����ҵ���Բ�ε����ݿ⣬���г���Բ�εĲ��ֶ����ɰ�ȥ����
%һ�����Ҳ���Բ�ε����ݿ�
DataBase_Circle = uint8(zeros(1,32,32,3));
DataBase_Circle_y = uint8(zeros(1,1));
DataBase_NoCircle = uint8(zeros(1,32,32,3));
DataBase_NoCircle_y = uint8(zeros(1,1));
Counter_DataBase_Circle = 1;
Counter_DataBase_NoCircle = 1;

%% ������ɫ�ָ��
%F = figure();
for i = 1:n
    AFig = PickAFig(DataBase,i);
    %ת����LAB�ռ䣬���ҳ���ʾ��ɫ����λ
    lab_AFig = rgb2lab(AFig);
    a = lab_AFig(:,:,2);
    b = lab_AFig(:,:,3);
    distance = zeros([size(a), nColors]);
    for count = 1:nColors
        distance(:,:,count) = ( (a - STD_Color(count,1)).^2 + ...
        (b - STD_Color(count,2)).^2 ).^0.5;
    end
    [~, label] = min(distance,[],3);
    %���õ���Ľ��������һ���Ҷ�ͼ��
    color_labels = uint8([0 127 255]);
    label = color_labels(label);
    %��ԲȦ����
    %ʹ�÷���1
    radiusRange = [7 13];
    Sensitivity = 0.9;
    [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','bright');
    if(isempty(centers))
        [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','dark');
        if(isempty(centers))
            [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','bright');
            if(isempty(centers))
                [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','dark');
                if(isempty(centers))
                    %����Ҳ���ԲȦ�����Ϊһ���Ҳ������ݿ�
                    DataBase_NoCircle(Counter_DataBase_NoCircle,:,:,:) = shiftdim(AFig,-1);
                    DataBase_NoCircle_y(Counter_DataBase_NoCircle,1) = y_train(i,1);
                    Counter_DataBase_NoCircle = Counter_DataBase_NoCircle + 1;
                end
            end
        end
    end
    if(~isempty(centers))
        %�����ɰ�
        [Mask] = CircleMask(centers,radii);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        %������
        DataBase_Circle(Counter_DataBase_Circle,:,:,:) = shiftdim(AFig_Masked,-1);
        DataBase_Circle_y(Counter_DataBase_Circle,1) = y_train(i,1);
        Counter_DataBase_Circle = Counter_DataBase_Circle + 1;
        
        %��ͼ�Ƚ�
        %pause(0.1);
        disp(i);
        %delete(h);
    end
end

%���û���õı���
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_NoCircle DataBase_NoCircle_y
save('Scope.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig ָ����ʾ������ͼƬ
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [] = CompareFigs(FigA,FigB)
%���ڱȽ��޸��Ժ��ͼ��
%����ͼ�����sRGBת����Ȼ������ʾ
    FigA = lin2rgb(FigA);
    FigB = lin2rgb(FigB);
    %figure();
    imshowpair(FigA,FigB,'montage');
end

function [] = CompareFigs2(FigA,centers,radii,FigB)
%һ���ԱȽ�����ͼ
%FigA��ԭͼ+ԲȦ
%FigB����ɫʶ���Ժ�Ķ���
    subplot(1,2,1);
    FigA = lin2rgb(FigA);
    imshow(FigA);
    if(~isempty(centers))
        viscircles(centers,radii);
    end
    subplot(1,2,2);
    imagesc(FigB);
    colorbar;
end

function [Mask] = CircleMask(centers,radii)
    %���ɵ�Mask����ЩԲ�εĲ���
    %ͼƬ�ĳ���
    FigH = 32;
    FigL = 32;
    %��԰뾶��ƫ������Ϊ������Բ�εĿ׶���
    Offset = 2;
    n = size(centers);
    Driver = linspace(0, 2*pi, 50);
    Mask = zeros(FigH,FigL);
    for i = 1:1:n
        r = radii(i) + Offset;
        c = centers(i,:);
        AMask = poly2mask(r*cos(Driver)+c(1), r*sin(Driver)+c(2), FigH, FigL);
        Mask = Mask | AMask;
    end
end