clc, clear
load 'Indian_pines_corrected.mat'
load 'Indian_pines_gt.mat'
data = indian_pines_corrected;
image_gt = indian_pines_gt;

delta = 0.7;
Res = 20;
scale_num =3;
k = generate_superpixel_num(delta, Res,image_gt,scale_num)%[1051, 525, 262]
%%
[rows,cols,depth]=size(data);
labels_superpixel = cell(size(k,2),1);
tic
[PCA_hsi, out_param] = PCA_img(data,1); % obtain first PC
for i = 1:size(k,2)
    [labels_superpixel{i,1},bmapOnImg]=suppixel(PCA_hsi,k(i),image_gt);
    labels_superpixel{i,1} = labels_superpixel{i,1}+1;
end
toc
% obtain superpixel
save labels_superpixel labels_superpixel




