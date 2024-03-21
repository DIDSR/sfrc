clear all
clc 
========================================================================
%Phantom Parameters:
%========================================================================
N        = 256;
theta    = (-70:2:70)+90;
Ntheta   = length(theta);
p        = N; %no of projection rays
[A,b,gt] = paralleltomo(N,theta,p); %system matrix (A); projections (b); ground truth (gt)
figure, imshow(reshape(gt, N, N), []), title('Ground Truth');
figure, imshow(reshape(b, N, Ntheta), []), title ('Sinogram');

%========================================================================
%missing wedge-based conventional reconstruction:
%========================================================================
bp=A'*b;
figure, imshow(reshape(bp/max(bp(:)), N, N), []), title('BP');

sino=reshape(b, N, Ntheta);
%sinof=apply_ramlak(sino);
freqs=linspace(-1, 1, N).';
myFilter = abs( freqs );
myFilter = repmat(myFilter, [1 Ntheta]);

% do my own FT domain filtering
ft_R = fftshift(fft(sino,[],1),1);
filteredProj = ft_R .* myFilter;
filteredProj = ifftshift(filteredProj,1);
ift_R = real(ifft(filteredProj,[],1));
figure, imshow(ift_R, []), title ('filtered sino');

fbp=A'*ift_R(:);
figure, imshow(reshape(fbp/max(fbp(:)), N, N), []);
print(['mw_fbp_recon.png'], '-dpng');