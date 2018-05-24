function D = distEmd( X, Y )
% Earth Mover's Distance (EMD) between positive vectors (histograms).
%   Note for 1D, with all histograms having equal weight, there is a simple
%   closed form for the calculation of the EMD.  The EMD between histograms
%   x and y is given by the sum(abs(cdf(x)-cdf(y))), where cdf is the
%   cumulative distribution function (computed simply by cumsum).
%copied from:
%http://uk.mathworks.com/matlabcentral/fileexchange/29004-feature-points-in
%-image--keypoint-extraction?focused=3805154&tab=function

Xcdf = cumsum(X,2);
Ycdf = cumsum(Y,2);

m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  ycdf = Ycdf(i,:);
  ycdfRep = ycdf( mOnes, : );
  D(:,i) = sum(abs(Xcdf - ycdfRep),2);
end
