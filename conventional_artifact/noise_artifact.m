clc; clear all;
close all;
addpath("src/")
% cd irt
% setup
% cd ..

%% input ground truth (i.e., normal-dose CT with minimal TV applied to remove noise)
xtrue_path           = 'data/L506_sh_fd_slice_32/mk_L506_tv_slice_32.raw';
mask_path            = 'data/L506_sh_fd_slice_32/mk_of_L506_tv_slice_32.png';
xtrue                = read_raw_img(xtrue_path, [512 512], 'uint16');
mask                 = single(imread(mask_path));
nmask                = m_normalize(0, 1, mask);
xtrue_win            = ct_windowing(xtrue, 'L_105_W_800');

roi_a                = xtrue((310):(310+20), (320):(320+20));
roi_b                = xtrue((353):(353+20), (120):(120+20));
roi_c                = xtrue((288):(288+20), (400):(400+20));
sigma_hd             = [std(roi_a(:)), std(roi_b(:)), std(roi_c(:))];
save_imgs            ='True';
%fprintf('nd stds %f, %f, %f: \n', std(roi_a(:)), std(roi_b(:)), std(roi_c(:)));

%% forward model
sys_info.nb              = 986;
sys_info.na              = 720;
sys_info.ds              = 0.95;
sys_info.max_flux        = 2.25e5;

obj_info.fov             = 250;
misc_info.k_nd           = 0.60;
misc_info.k_ld           = 0.25; %dose inserted in forward projection
misc_info.intercept_k    = 1024; 
misc_info.filter_string  ='hann200';
misc_info.norm_type      ='remove_negative';
misc_info.mu_water_in_cm = 0.167;
misc_info.output_folder  = [];

%% noise insertion and reconstruction
fbp     = insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
fbp     = m_normalize(min(xtrue(:)), max(xtrue(:)), fbp);
froi_a  = fbp((310):(310+20), (320):(320+20));
froi_b  = fbp((353):(353+20), (120):(120+20));
froi_c  = fbp((288):(288+20), (400):(400+20));
fbp_win = ct_windowing(fbp,'L_105_W_800'); %in HU space

sigma_ld            = [std(froi_a(:)), std(froi_b(:)), std(froi_c(:))];
estimate_dose_level = 1/mean(sigma_hd./sigma_ld)^2; %noise level estimated on the reconstructed CT image
fprintf('hd acquisition is %f times the ld.\n', estimate_dose_level);

colormap gray;
figure (1)
im plc 1 2
im(1, xtrue_win', 'Reference'), cbar;
im(2, fbp_win', [num2str(floor(estimate_dose_level)),' times noisy']), cbar;


%% output filenames and write files
if strcmp (save_imgs, 'True')
    output_fname_uint8  = ['results/noise/irt_fbp_L506_sh_slice_32_unit8/']; %for plots in paper
    output_fname_uint16 = ['results/noise/input/']; % for sfrc analysis
    
    if ~exist(output_fname_uint8, 'dir')
            mkdir(output_fname_uint8)
    end
    if ~exist(output_fname_uint16, 'dir')
            mkdir(output_fname_uint16)
    end
    fname_uint8  = [output_fname_uint8, 'L506_sh_slice_32_25pd_L_1286_W_780.png']; %in HU + 1024 scaled space
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_25pd_uint16proj.raw'];
    
    imwrite(uint8(m_normalize(0, 255, fbp_win)), fname_uint8);
    write_raw_img(fname_uint16, uint16(fbp), 'uint16');
end 

