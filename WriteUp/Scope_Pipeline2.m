%Ŀ�������ڶ����ַ�����ʹ�þ���ķ����ж���״��
%�������ڵ�һ�ַ�����ʧ�ܴ��ϳɹ���ʧ���ǣ�4378:1315

load('Scope.mat');

%���óɹ���ʧ�ܵ����ݿ�
DataBase_success = uint8(zeros(1,32,32,3));
DataBase_success_y = uint8(zeros(1,1));

DataBase_No = uint8(zeros(1,32,32,3));
DataBase_No_y = uint8(zeros(1,1));

Counter_DataBase_success = 1;
Counter_DataBase_No = 1;

%������Ҫ���������
DataBase = DataBase_NoCircle;
n = size(DataBase,1);
for i = 1:1:n
    AFig = PickAFig(DataBase,i);
    
    %����ԭ����ͼֱ�Ӽ����û��Բ��
    [centers,radii] = FindCircle(AFig);
    if(~isempty(centers))
        %�����ɰ�
        [Mask] = CircleMask(centers,radii);
        RGBmask(:,:,1) = Mask;
        RGBmask(:,:,2) = Mask;
        RGBmask(:,:,3) = Mask;
        AFig_Masked = immultiply(AFig,RGBmask);
        %������
        DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
        DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
        Counter_DataBase_success = Counter_DataBase_success + 1;
        imshow(AFig_Masked);
    end
    
    if(isempty(centers))
        %������ɫ���෽����ͼƬ�ָ�Ϊ����
        [BW,~] = segmentImage(AFig);
        BW = ~BW;
        %����һ��û��ȥ���߽ǵľ�����
        BW_Kmean = BW;
        %ȥ����Щ����5�����ص���ɢ����
        Se = strel('disk',2);
        BW = imopen(BW,Se);
        %��С�ķ��಻������25�����أ������������Ϊ20��
        if(sum(sum(BW)) <= 20)
            BW = false(32,32);
        end
        %���Ȳ��ҷ���Ľ��������û��Բ��
        label = BW * 255;
        [centers,radii] = FindCircle(label);
        if(~isempty(centers))
            %�����ɰ�
            [Mask] = CircleMask(centers,radii);
            RGBmask(:,:,1) = Mask;
            RGBmask(:,:,2) = Mask;
            RGBmask(:,:,3) = Mask;
            AFig_Masked = immultiply(AFig,RGBmask);
            %������
            DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
            DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
            Counter_DataBase_success = Counter_DataBase_success + 1;
            imshow(AFig_Masked);
            %pause(1);
        end
        if(isempty(centers))
            %�����������û�з�����Բ�Σ���ô���Լ���һ��Բ�������ľ���
            if(sum(sum(BW_Kmean)) > 86)
                %�������Ľ����������ͼ����ô�ͻ�һ��Բ����޶ȵİ�Χ���ͼ��
                [row,col] = find(BW_Kmean == 1); 
                [radii,centers,~] = ExactMinBoundCircle([row  col]);
                %�����ɰ�
                [Mask] = CircleMask(centers,radii);
                RGBmask(:,:,1) = Mask;
                RGBmask(:,:,2) = Mask;
                RGBmask(:,:,3) = Mask;
                AFig_Masked = immultiply(AFig,RGBmask);
                %������
                DataBase_success(Counter_DataBase_success,:,:,:) = shiftdim(AFig_Masked,-1);
                DataBase_success_y(Counter_DataBase_success,1) = DataBase_NoCircle_y(i,1);
                Counter_DataBase_success = Counter_DataBase_success + 1;
                imshow(AFig_Masked);
            else
                %�������Ľ�����٣��㷨ʧ��
                DataBase_No(Counter_DataBase_No,:,:,:) = shiftdim(AFig,-1);
                DataBase_No_y(Counter_DataBase_No,1) = DataBase_NoCircle_y(i,1);
                Counter_DataBase_No = Counter_DataBase_No + 1;
            end
        end
    end
    drawnow;
    pause(1);
    disp(i);
end
clearvars -except DataBase_Circle DataBase_Circle_y DataBase_success DataBase_success_y DataBase_No DataBase_No_y
save('Scope2.mat');

function [AFig] = PickAFig(Figs,NumFig)
%PickAFig ָ����ʾ������ͼƬ
    AFig = Figs(NumFig,:,:,:);
    AFig = squeeze(AFig);
%imtool(AFig)
end

function [BW,maskedImage] = segmentImage(RGB)
%  ���µĴ��������� imageSegmenter App�Զ����ɵġ�
%  ���������þ���ķ�������ɫ�ķָ�
%  [BW,MASKEDIMAGE] = segmentImage(RGB) segments image RGB using
%  auto-generated code from the imageSegmenter App. The final segmentation
%  is returned in BW, and a masked image is returned in MASKEDIMAGE.

% Convert RGB image into L*a*b* color space.
X = rgb2lab(RGB);

% Auto clustering
sz = size(X);
im = single(reshape(X,sz(1)*sz(2),[]));
im = im - mean(im);
im = im ./ std(im);
s = rng;
rng('default');
L = kmeans(im,2,'Replicates',2);
rng(s);
BW = L == 2;
BW = reshape(BW,[sz(1) sz(2)]);

% Clear borders
BW = imclearborder(BW);

% Fill holes
BW = imfill(BW, 'holes');

% Invert mask
BW = imcomplement(BW);

% Create masked image.
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
end
function [centers,radii] = FindCircle(label)
    radiusRange = [7 13];
    Sensitivity = 0.9;
    [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','bright');
    if(isempty(centers))
        [centers,radii] = imfindcircles(label,radiusRange,'Method','TwoStage','Sensitivity',Sensitivity,'ObjectPolarity','dark');
        if(isempty(centers))
            [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','bright');
            if(isempty(centers))
                [centers,radii] = imfindcircles(label,radiusRange,'Method','PhaseCode','Sensitivity',Sensitivity,'ObjectPolarity','dark');
            end
        end
    end
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