images_fake = dir('high_res_fake'); 
images_real = dir('high_res_real'); 
count = 3;
avg_psnr = 0.0;
max_psnr = 0.0;
min_psnr = 0.0;
psnr_val = ones(1,288);
ssim_val = ones(1,288);
for v = 1:17
    for u = 1:17
         if count<290
        % Derive the image filename
%         img_filename = sprintf('%s/IMG_%d_%d.png',lf_foldername,v,u);
        real_img_filename = images_real(count).name;
        fake_img_filename = images_fake(count).name;
        real_img_filename = sprintf('%s/%s','high_res_real', real_img_filename);
        fake_img_filename = sprintf('%s/%s','high_res_fake', fake_img_filename);
        % Load the image
        I_real = imread(real_img_filename);
        I_fake = imread(fake_img_filename);
        psnr_val(count-2) = psnr(I_fake, I_real);
        ssim_val(count-2) = ssim(I_fake, I_real);
        count = count + 1;
        end
    end
end
avg_psnr = mean(psnr_val);
max_psnr = max(psnr_val);
min_psnr = min(psnr_val);

avg_psnr
max_psnr
min_psnr
mean(ssim_val)