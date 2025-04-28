clc; clear all;
close all;
addpath("src/")
% cd irt
% setup
% cd ..

%% input ground truth
xtrue_path          = 'data/L506_sh_fd_slice_32/mk_L506_tv_slice_32.raw';
mask_path           = 'data/L506_sh_fd_slice_32/mk_of_L506_tv_slice_32.png';
xtrue               = read_raw_img(xtrue_path, [512 512], 'uint16');
mask                = single(imread(mask_path));
nmask               = m_normalize(0, 1, mask);
save_imgs           = 'True';


%% input parameters for reconstruction 
[Nx, Ny]            = size(xtrue);
FOV                 = 250;
DOWN                = 1;
mu_water            = 0.17/10; % in mm-1
intercept_k         = 1024;
scalefac_k          = 1024;
filter_string       = '';
misc_info.norm_type = 'gt_range';
mw_all_types        = {'pm60', 'pm70', 'pm80', 'pm90', 'pm180'};
mw_type             = mw_all_types{4}; % change missing wedge option here

%% output foldernames
output_fname_uint8  = ['results/missing_wedge/irt_fbp_L506_sh_slice_32_unit8/'];
output_fname_uint16 = ['results/missing_wedge/input/'];


%% IRT-based image and sinogram geometries
ig   = image_geom('nx',Nx, 'ny', Ny,'fov', FOV,'offset_x',0,'down', DOWN);

%% Different missing wedge options
if strcmp (mw_type, 'pm60')
    sgn  = sino_geom('fan', 'na',60, 'nb', 936, 'ds', 1.0, ...
        'dsd', 1085.6, 'dso',595, 'dfs',0, ...
        'source_offset',0.0, 'orbit',120, 'down', ...
        1, 'strip_width', 'd', 'orbit_start', 30);
    fname_uint8  = [output_fname_uint8, 'recon_theta_', mw_type, '_spac_2_L_1286_W_780_2_uint8.png'];
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_recon_theta_', mw_type, '_spac_2_uint16.raw' ];

elseif strcmp (mw_type, 'pm70')
    sgn  = sino_geom('fan', 'na',70, 'nb', 936, 'ds', 1.0, ...
        'dsd', 1085.6, 'dso',595, 'dfs',0, ...
        'source_offset',0.0, 'orbit',140, 'down', 1, ...
        'strip_width', 'd', 'orbit_start', 20);

    fname_uint8  = [output_fname_uint8, 'recon_theta_', mw_type, '_spac_2_L_1286_W_780_2_uint8.png'];
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_recon_theta_', mw_type, '_spac_2_uint16.raw' ];

elseif strcmp (mw_type, 'pm80')
    sgn  = sino_geom('fan', 'na',80, 'nb', 936, 'ds', 1.0, ...
        'dsd', 1085.6, 'dso',595, 'dfs',0, ...
        'source_offset',0.0, 'orbit',160, 'down', 1, ...
        'strip_width', 'd', 'orbit_start', 10);
    fname_uint8  = [output_fname_uint8, 'recon_theta_', mw_type, '_spac_2_L_1286_W_780_2_uint8.png'];
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_recon_theta_', mw_type, '_spac_2_uint16.raw' ];

elseif strcmp (mw_type, 'pm90')
    sgn  = sino_geom('fan', 'na',180, 'nb', 936, 'ds', 1.0, ...
        'dsd', 1085.6, 'dso',595, 'dfs',0, ...
        'source_offset',0.0, 'orbit',180, 'down', 1, ...
        'strip_width', 'd', 'orbit_start', 0);
    fname_uint8  = [output_fname_uint8, 'recon_theta_', mw_type, '_spac_1_L_1286_W_780_2_uint8.png'];
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_recon_theta_', mw_type, '_spac_1_uint16.raw' ];

else
    sgn  = sino_geom('fan', 'na',720, 'nb', 936, 'ds', 1.0, ...
        'dsd', 1085.6, 'dso',595, 'dfs',0, ...
        'source_offset',0.0, 'orbit',360, 'down', 1, ...
        'strip_width', 'd', 'orbit_start', 0);

    fname_uint8  = [output_fname_uint8, 'recon_theta_', mw_type, '_spac_0.5_L_1286_W_780_2_uint8.png'];
    fname_uint16 = [output_fname_uint16, 'L506_sh_slice_32_recon_theta_', mw_type, '_spac_0.5_uint16.raw' ];
end

%% Forward projection, reconstruction, and normalization
G     = Gtomo2_dscmex(sgn, ig);
tmp   = fbp2(sgn, ig, 'type','std:mat');
xtrue = xtrue - intercept_k;
xtrue = xtrue'*mu_water/scalefac_k+ mu_water;

if (min(xtrue(:))<0)
    xtrue = xtrue + (-min(xtrue(:)));
end
sino_n = G*xtrue;
fbp    = fbp2(sino_n, tmp, 'window', filter_string);

if strcmp(misc_info.norm_type, 'positive_scale')
    if (min(fbp(:))<0)
        fbp = fbp + (-min(fbp(:)));
    end
elseif strcmp(misc_info.norm_type, 'remove_negative')
    if (min(fbp(:))<0)
        neg_ind = find(fbp<0);
        fbp(neg_ind) =0.0;
    end
elseif strcmp(misc_info.norm_type, 'gt_range')
    fbp = m_normalize(min(xtrue(:)), max(xtrue(:)), fbp);
else 
    fbp;
end

xtrue     = (xtrue' - mu_water)*scalefac_k/mu_water + intercept_k;
fbp       = (fbp' - mu_water)*scalefac_k/mu_water + intercept_k;

xtrue_win = ct_windowing(xtrue, 'irt_soft_artifact');
fbp_win   = ct_windowing(fbp, 'irt_soft_artifact');

figure (2)
colormap gray;
im plc 1 2
im(1, xtrue_win', 'Reference'), cbar;
im(2, fbp_win', ['rotation by ', mw_type, ' deg']), cbar;

%% write output if save images is true
if strcmp (save_imgs, 'True')
    if ~exist(output_fname_uint8, 'dir')
            mkdir(output_fname_uint8)
    end
    if ~exist(output_fname_uint16, 'dir')
            mkdir(output_fname_uint16)
    end
    imwrite(uint8(m_normalize(0, 255, fbp_win)), fname_uint8); % to be used for plotting
    write_raw_img(fname_uint16, uint16(fbp), 'uint16'); % To be used for sFRC analysis
    %imwrite(uint8(m_normalize(0, 255, xtrue_win)), 'data/L506_sh_fd_slice_32/mk_L506_sh_fd_tv_slice_32_L_1286_W_780_2_uint8.png');

end
