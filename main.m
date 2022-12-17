% % % A Robust and Secure Video Steganography Method in DWT-DCT Domains
% % % % % % % Based on Multiple Object Tracking and ECC
clc;
clear all;
close all;
warning('off','all');
%% Motion Object Detection and Region Extraction
folder = dir('View_001\*.jpg');
for x = 1:length(folder)
    f = folder(x).name;
    images{x,:} = imread(fullfile('View_001\',f));
end
load a1.mat;
ij = 1;l = 1;k = 1;
[z z1] = size(images);

       if mask(i,j) == 1
           b = j;
           break;
       end    
    end
end
D = 2*(floor(cc.Centroid(2) - b));   
%%>>>>>>>>>>>>>>>>>>>>>>>>>>>SPATIAL CALIBRATION<<<<<<<<<<<<<<<<<<<<<<<<<<<
d1 = floor(D/10);    %Average radius of the OD(optic disk).             
d2 = floor(D/360);   %Size of the smallest MA(Microa-neurysms).
d3 = floor(D/28);    %Size of the largest HE(Hemorrhages).
%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>PREPROCESSING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
%% step:1--------->>>>>>illumination Equalization<<<<<<<-------------------
R1 =im(:,:,1);
G1 =im(:,:,2);
B1 =im(:,:,3);
h1 =fspecial('average',d1);
hm1 = imfilter(R1,h1);hm2 = imfilter(G1,h1);hm3 = imfilter(B1,h1);
hmm = cat(3,hm1,hm2,hm3);
avg = mean2(im);
ill_eqa = im + (avg - (im.*hmm));
figure,subplot(2,2,1);imshow(ill_eqa);title('ILLUMINATION');
%%step:2---------->>>>>>>>>>>>Denoising<<<<<<<<<<<<------------------------
R2 = ill_eqa(:,:,1);
G2 = ill_eqa(:,:,2);
B2 = ill_eqa(:,:,3);
h2 =fspecial('average',d2);
hm12 = imfilter(R2,h2);hm22 = imfilter(G2,h2);hm32 = imfilter(B2,h2);
dn = cat(3,hm12,hm22,hm32);
subplot(2,2,2);imshow(dn);title('DENOSING');
%%step:3---------->>>>>>Adaptive Contrast Equalization<<<<<<---------------
R3 = dn(:,:,1);
G3 = dn(:,:,2);
B3 = dn(:,:,3);
h3 = fspecial('average',d3);
hm13 = imfilter(R3,h3);hm23 = imfilter(G3,h3);hm33 = imfilter(B3,h3);
Iace = cat(3,hm13,hm23,hm33);
R31 = std2(R3);G31 = std2(G3);B31 = std2(B3);
so = cat(3,R31,G31,B31);
is = std2(so);
Ice = (dn + 1 .*(dn .*(1 - Iace))/is);
subplot(2,2,3);imshow(Iace);title('EQUALIZATION');
%%step:4--------->>>>>>Gamma/Color Normalization<<<<<<<<<------------------
% hgamma = vision.GammaCorrector(2.0,'Correction','Gamma');
% y = step(hgamma,Iace);
% imshow(y);
r3 = im2double(Ice(:,:,1));m1 = mean2(r3);s1 = std2(r3);f1min =( m1 - (s1))/10;f1max =3*(m1 + (s1))/10;
g3 = im2double(Ice(:,:,2));m2 = mean2(g3);s2 = std2(g3);f2min =( m2 - (s2))/10;f2max =3*(m2 + (s2))/10;
b3 = im2double(Ice(:,:,3));m3 = mean2(b3);s3 = std2(b3);f3min =( m3 - (s3))/10;f3max =3*(m3 + (s3))/10;
j1 = imadjust(r3,stretchlim(r3),[f1min f1max]);
j2 = imadjust(g3,stretchlim(g3),[f2min f2max]);
j3 = imadjust(b3,stretchlim(b3),[f3min f3max]);
p = cat(3,j1,j2,j3);mask = im2bw(mask);
Ip1 = j1.*mask;Ip2 = j2.*mask;Ip3 = j3.*mask;
Ip = cat(3,Ip1,Ip2,Ip3);subplot(2,2,4);,imshow(p);title('COLOR NORMALIZATION');
%%>>>>>>>>>>>>>>>>>>>>OPTIC DISC REMOVAL<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
mm = Ip(:,:,2);  %Extracted green channel
m4 = mean2(mm);
m5 = std2(mm);
figure,imshow(mm,[]);
%// Invert the green   
if isinteger(mm)
    z = intmax(class(mm))-mm;
elseif isfloat(mm)        
    z = 1 - mm;
elseif islogical(mm)
    z = ~mm;
end
figure,imshow(z,[]);title('Optic Disc Removal');
  se = strel('ball',85,85); 
%  adahist= histeq(z); % Structuring Element
% gopen = imopen( adahist,se);   % Morphological Open
% godisk =  adahist - gopen;    % Remove Optic Disk
% figure,imshow(godisk,[]);title('Optic Disk Removed');
%%>>>>>>>>>>>>>>>>>>>>>>CANDIDATE EXTRACTION<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
xx = imregionalmin(z,8);
 figure,subplot(1,2,1);imshow(xx);title('Reginal Minima');
%  j = imgradient(xx);figure,imshow(j);
 MM = bwareaopen(xx,20);subplot(1,2,2);imshow(MM);title('Finding Candidate');
% % % % % %-------->>>>>>DYNAMIC SHAPE FEATURES<<<<<<<<-------- % % % % % % 
lesion_dec = regionprops(MM,'All');
 figure,imshow(im1);title('Output');
xxxx = 3*ones(1,length(lesion_dec));
hold on
 for i = 1:length(lesion_dec)
viscircles(lesion_dec(i).Centroid,5,'Color','b');
end
hold off
n = numel(im1);
Istop = floor(avg - 0.5); % flooding level
Rarea = zeros(length(lesion_dec),length(Istop));
Elong  = zeros(length(lesion_dec),length(Istop));
Ecc  = zeros(length(lesion_dec),length(Istop));
Circ  = zeros(length(lesion_dec),length(Istop));
Rect  = zeros(length(lesion_dec),length(Istop));
Sol = zeros(length(lesion_dec),length(Istop));
for i = 1:Istop
    for k =1:length(lesion_dec)
        n1 = numel(lesion_dec(k).PixelList());
        Rarea(k,i) = n1 ./ n;
        a = lesion_dec(k).BoundingBox(1,3:4);
        Elong(k,i) = 1 -(a(1)/a(2));
        Ecc(k,i) = lesion_dec(k).Eccentricity();
        Circ(k,i) = (lesion_dec(k).Perimeter.^ 2)./ 4*pi*(lesion_dec(k).Area());
        Rect(k,i) = a(1).*a(2);
        Sol(k,i) = lesion_dec(k).Solidity(); 
    end
end
k = 1;
 M = 4 + 6 * (k + 4);
x = 1:Istop;
feature = zeros(length(lesion_dec),M);
 
for c = 1:length(lesion_dec)
    p1 = polyfit(x,Rarea(c,:),1);
    y1 = polyval(p1,x);
    y1e = sqrt(mean2(((Rarea(c,:)) - y1).^2));
    a1 = mean(Rarea(c,:)); aa1 = median(Rarea(c,:));f_area = ([p1(1),p1(2),y1e,a1,aa1]);
    
    p2 = polyfit(x,Elong(c,:),1);
    y2 = polyval(p2,x);
    y2e = sqrt(mean2(((Elong(c,:) - y2).^2)));
    a2 = mean(Elong(c,:)); aa2 = median(Elong(c,:));f_Elong = ([p2(1),p2(2),y2e,a2,aa2]);
    
    p3 = polyfit(x,Ecc(c,:),1);
    y3 = polyval(p3,x);
    y3e = sqrt(mean2(((Ecc(c,:) - y3).^2)));
    a3 = mean(Ecc(c,:)); aa3 = median(Ecc(c,:));f_Ecc = [p3(1),p3(2),y3e,a3,aa3];
    
    p4 = polyfit(x,Circ(c,:),1);
    y4 = polyval(p4,x);
    y4e = sqrt(mean2(((Circ(c,:) - y4).^2)));
    a4 = mean(Circ(c,:)); aa4 = median(Circ(c,:));f_Circ = [p4(1),p4(2),y4e,a4,aa4];
    
    p5 = polyfit(x,Rect(c,:),1);
    y5 = polyval(p5,x);
    y5e = sqrt(mean2(((Rect(c,:) - y5).^2)));
    a5 = mean(Rect(c,:)); aa5 = median(Rect(c,:));f_Rect = ([p5(1),p5(2),y5e,a5,aa5]);
    
    p6 = polyfit(x,Sol(c,:),1);
    y6 = polyval(p6,x);
    y6e = sqrt(mean2(((Sol(c,:) - y6).^2)));
    a6 = mean(Sol(c,:)); aa6 = median(Sol(c,:));f_Sol =([p6(1),p6(2),y6e,a6,aa6]);
    feature(c,:) = [f_area,f_Elong,f_Ecc,f_Circ,f_Rect,f_Sol,m1,m2,m3,m4];
end
feature = mean(feature);
load Feature;
train = ones(88,1);
train(17:end) = 2;
rng(1); % For reproducibility
BaggedEnsemble = TreeBagger(200,Feature,train,'OOBPrediction','On',...
    'Method','classification');
Yfit = predict(BaggedEnsemble,feature);
aaa=str2num(cell2mat(Yfit));
disp('RANDOM FOREST CLASSIFIER RESULT');
if aaa==1
    msgbox('Normal');
    disp('Normal');
else
    msgbox('Abnormal');
    disp('Abnormal');
end


