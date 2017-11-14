%Ŀ�����������ַ�������ˮ��ָ����
%�������һ�ַ���������û�гɹ���ͼƬ���ᱻ����һ��Ĭ�ϵ�Բ��Mask���Ա�֤Mask��״
%���������ѧϰ���̡�
%������̲���ı����ݿ������ݵ�˳��
%�ɹ�ʧ�ܵı��ʣ�622��539

load('Scope2.mat');

%���óɹ����ݿ⡣
DataBase_success3 = uint8(zeros(1,32,32,3));
DataBase_success3_y = DataBase_No_y;

Counter_DataBase_success3 = 1;
Counter_DataBase_No3 = 1;

%����Ҫ��������ݿ�
DataBase = DataBase_No;
n = size(DataBase,1);
for i = 1:1:n
    AFig = PickAFig(DataBase,i);
    %���÷�ˮ�뷽�����߽��顣
    AFig_gray = rgb2gray(AFig);
    %% ���ǰ��ɫ
    %��ǰ��ɫ�Ľ���ͿĨ����
    Se = strel('disk',2);

    %��ʴ�Ժ�Ҫ�ؽ�ͼƬ�����������Ҫһ����ʴ�Ժ��ͼƬ
    AFig_Rode = imerode(AFig_gray,Se);
    AFig_Rode_Construct = imreconstruct(AFig_Rode,AFig_gray);
    
    AFig_Dilate = imdilate(AFig_Rode_Construct,Se);
    AFig_Dilate_Construct = imreconstruct(imcomplement(AFig_Dilate),imcomplement(AFig_Rode_Construct));
    AFig_Dilate_Construct = imcomplement(AFig_Dilate_Construct);
    
    %���������ڲ������ֵ
    fgm = imregionalmax(AFig_Dilate_Construct);
    %fgm = imopen(fgm,Se);
    %fgm = imerode(fgm,Se);
    
    I2 = AFig_gray;
    I2(fgm) = 255;
    
    bw = imbinarize(AFig_Dilate_Construct);
    D = bwdist(bw);
    DL = watershed(D);
    bgm = DL == 0;
    
    hy = fspecial('sobel');
    Iy = imfilter(double(AFig_gray),hy,'replicate');
    Ix = imfilter(double(AFig_gray),hy','replicate');
    gradmag = sqrt(Ix.^2 + Iy.^2);
    gradmag = imimposemin(gradmag, bgm | fgm);
    L = watershed(gradmag);
    
    %�ҵ��м�Ĳ��ֵı�Ƿ��ŵ�����ͨ��־�ķָ����
    Middle = [L(16,16:17) L(17,16:17)];
    Middle = mode(Middle);
    %�ҵ������������Ӧ�ı�Ƿ�Χ
    [row,col] = find(L == Middle); 
    [radii,centers,~] = ExactMinBoundCircle([row  col]);
    %�����ɰ�
    [Mask] = CircleMask(centers,radii+4);
    Mask = imclearborder(Mask,4);
    RGBmask(:,:,1) = Mask;
    RGBmask(:,:,2) = Mask;
    RGBmask(:,:,3) = Mask;
    if(sum(sum(Mask)) > 0)
        AFig_Masked = immultiply(AFig,RGBmask);
        DataBase_success3(Counter_DataBase_success3,:,:,:) = shiftdim(AFig_Masked,-1);
        Counter_DataBase_success3 = Counter_DataBase_success3 + 1;
    else
        %�㷨ʧ�ܣ�����һ��Ĭ�ϵ�Mask
        [Mask] = CircleMask([16 16],13);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        DataBase_success3(Counter_DataBase_success3,:,:,:) = shiftdim(AFig_Masked,-1);
        Counter_DataBase_success3 = Counter_DataBase_success3 + 1;
        Counter_DataBase_No3 = Counter_DataBase_No3 + 1;
    end
    imshow(AFig_Masked);
    drawnow;
    %pause(1);
    disp(i);
end
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_success DataBase_success_y DataBase_success3 DataBase_success3_y Counter_DataBase_No3
save('Scope3.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig ָ����ʾ������ͼƬ
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [] = WatershedSeg(Fig)
    
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