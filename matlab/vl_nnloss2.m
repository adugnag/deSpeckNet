function Y = vl_nnloss2(c,X,dzdy,varargin)

% --------------------------------------------------------------------
% pixel-level total variation loss
% --------------------------------------------------------------------

if nargin <= 2 || isempty(dzdy)

ux = X(1:end-1, 1:end-1,:) - X(1:end-1, 2:end,:);
uy = X(1:end-1, 1:end-1,:) - X(2:end, 1:end-1,:);

v = ux.^2 + uy.^2;

Y = (0.5*mean(v(:)));
    
else
    
ux = X(1:end-1, 1:end-1, :) - X(1:end-1, 2:end  , :) ;
uy = X(1:end-1, 1:end-1, :) - X(2:end  , 1:end-1, :) ;

dzdx = zeros(size(X), 'like', X);

dzdx(1:end-1,1:end-1,:) = ux + uy;
dzdx(1:end-1,2:end,:) = dzdx(1:end-1,2:end,:) - ux;
dzdx(2:end,1:end-1,:) = dzdx(2:end,1:end-1,:) - uy;

% Then add the derivative 
Y = dzdx + dzdy;
  

end
end


