%addpath('/Users/PravatCaysei/Box Sync/mac_documents/carnegie/for_T_research/RESULTS/MATLAB/globalfiles')

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



