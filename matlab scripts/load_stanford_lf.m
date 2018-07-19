function LF = load_stanford_lf(dataset_foldername,lf_name)
% This function is used to lead the stanford light fields (v,u,y,x,c)

% Derive the foldername of the lf_name
lf_foldername = [dataset_foldername,lf_name];
images = dir(lf_foldername); 
count = 3;
for v = 1:17
    for u = 1:17
        % Derive the image filename
%         img_filename = sprintf('%s/IMG_%d_%d.png',lf_foldername,v,u);
        img_filename = images(count).name;
        img_filename = sprintf('%s/%s',lf_foldername, img_filename);
        % Load the image
        I = imread(img_filename);
        I = imresize(I,[128,128]);
%         u_new = u;
%         v_new = 17 - v + 1;
        
        % Store the light field LF(v,u,y,x,c)
        LF(v,u,:,:,:) = I;
        count = count + 1;
    end
end