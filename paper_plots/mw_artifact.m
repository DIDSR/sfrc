clear all
clc 
shepplogan = 'F';
windowing  = 'soft';
%%========================================================================
%Phantom Parameters:
%========================================================================
if strcmp(shepplogan, 'T')
    N        = 256;
    theta    = (-70:2:70)+90;
    Ntheta   = length(theta);
    p        = N; %no of projection rays
    [A,b,gt] = paralleltomo(N,theta,p); %system matrix (A); projections (b); ground truth (gt)
    figure, imshow(reshape(gt, N, N), []), title('Ground Truth');
    figure, imshow(reshape(b, N, Ntheta), []), title ('Sinogram');
else
    gt       = double(imread('plot1/L291_tv_000163.tif'));
    ngt      = m_normalize(0, 1, gt);
    [nx, ny] = size(gt);

    N        = nx;
    theta    = (-60:1:60)+90;
    Ntheta   = length(theta);
    p        = N*6; %no of projection rays
    [A,sb,sgt] = paralleltomo(N,theta,p); %system matrix (A); 
    b          = A*ngt(:);
    figure, imshow(ngt, []), title('Ground Truth');
    figure, imshow(reshape(b, p, Ntheta), []), title ('Sinogram');

end

%========================================================================
%missing wedge-based conventional reconstruction:
%========================================================================
bp=A'*b;
figure, imshow(reshape(bp/max(bp(:)), N, N), []), title('BP');

sino=reshape(b, p, Ntheta);
%sinof=apply_ramlak(sino);
freqs=linspace(-1, 1, p).';
myFilter = abs( freqs );
myFilter = repmat(myFilter, [1 Ntheta]);

% do my own FT domain filtering
ft_R = fftshift(fft(sino,[],1),1);
filteredProj = ft_R .* myFilter;
filteredProj = ifftshift(filteredProj,1);
ift_R = real(ifft(filteredProj,[],1));
figure, imshow(ift_R, []), title ('filtered sino');

fbp=A'*ift_R(:);

if strcmp(shepplogan, 'T')
    figure, imshow(reshape(fbp/max(fbp(:)), N, N), []);
    print(['plot1/mw_fbp_shepp_recon.png'], '-dpng');
else
    nev_ind      = find(fbp<0);
    fbp(nev_ind) = 0;
    fbp_hu       = m_normalize(min(gt(:)), max(gt(:)), fbp);
    if strcmp (windowing, 'soft')
      LL = 1075-700/2;
      UL = 1075+700/2;
    elseif strcmp(windowing, 'bone')
      LL = 425-1500/2;
      UL = 425 + 1500/2;
    end 
    LL_ind         = find(fbp_hu<=LL);
    fbp_hu(LL_ind) = LL;
    UL_ind         = find(fbp_hu>=UL);
    fbp_hu(UL_ind) = UL;
    fbp_hu         = reshape(fbp_hu, N, N);
    figure, imshow(fbp_hu, []);
    imwrite(uint8(m_normalize(0, 255, fbp_hu)), ['plot1/mw_fbp_L291_000163_recon_L_1075_W_700_2_8bit.png']);
end 


function [datnorm]= m_normalize(a, b, dataset)
	dims = size(dataset);
	data = dataset(:);

	n 		= size(data);
	datnorm = zeros(n);
	Xmin    = min(data);
	Xmax    = max(data);
	Range   = Xmax-Xmin;
	datnorm = a+((data-Xmin).*(b-a)/Range);
	datnorm = reshape(datnorm, dims);

end

