% code which computes band functions and Berry curvature distribution of the p_x + i p_y model

% settings
clear all;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex'); set(groot, 'defaulttextinterpreter','latex');

% parameters of the model. NB band functions degenerate when mu/2t = 2,0,-2
t = 1;
d = 1; % delta
m = 2; % mu

% numerical parameter: number of k points in each direction 
k_points = 60;

% function which returns Hamiltonian
H = '@(x,y,a,b,c) [ - c - 2*a*( cos(x) + cos(y) ) , b*( sin(x) - 1j*sin(y) ); b*( sin(x) + 1j*sin(y) ) , c + 2*a*( cos(x) + cos(y) ) ]';
H = str2func(H);

% function which returnts derivatives of Hamiltonian (necessary for computing Chern number)
dk1H = '@(x,y,a,b,c) [ 2*a*sin(x) , b*cos(x) ; b*cos(x) , -2*a*sin(x) ]';
dk2H = '@(x,y,a,b,c) [ 2*a*sin(y) , -1j*b*cos(y) ; 1j*b*cos(y) , -2*a*sin(y) ]';
dk1H = str2func(dk1H);
dk2H = str2func(dk2H);

% generate grid of k values
[K1,K2] = meshgrid(linspace(0,2*pi,k_points),linspace(0,2*pi,k_points));

% initialize band functions and Berry curvature distributions evaluated on grid
lower_band = zeros(size(K1)); upper_band = lower_band;
lower_berry_curv = zeros(size(K1)); upper_berry_curv = lower_berry_curv;
gap = zeros(size(K1));

% compute band functions and Berry curvature distributions at each grid point
for i = 1:k_points;
 for j = 1:k_points;
  % grid point co-ordinates
  k1 = K1(i,i); k2 = K2(j,j);
  % H evaluated at grid point
  H_loc = H(k1,k2,t,d,m);
  % diagonalize H, sort eigenvalues low to high
  [V,D] = eig(H_loc);
  [~,idx]=sort(diag(D));
  D = D(idx,idx); V = V(:,idx);
  % get band functions
  lower_band(i,j) = D(1,1); upper_band(i,j) = D(2,2);
  % get gap function
  gap(i,j) = D(2,2) - D(1,1);
  % get Berry curvature
  gapsquared = (gap(i,j))^2;
  lower_inner_products = dot( V(:,1) , dk1H(k1,k2,t,d,m)*V(:,2) )*dot( V(:,2) , dk2H(k1,k2,t,d,m)*V(:,1) ) - dot( V(:,1) , dk2H(k1,k2,t,d,m)*V(:,2) )*dot( V(:,2) , dk1H(k1,k2,t,d,m)*V(:,1) );
  upper_inner_products = dot( V(:,2) , dk1H(k1,k2,t,d,m)*V(:,1) )*dot( V(:,1) , dk2H(k1,k2,t,d,m)*V(:,2) ) - dot( V(:,2) , dk2H(k1,k2,t,d,m)*V(:,1) )*dot( V(:,1) , dk1H(k1,k2,t,d,m)*V(:,2) );
  lower_berry_curv(i,j) = 1j*lower_inner_products/gapsquared; upper_berry_curv(i,j) = 1j*upper_inner_products/gapsquared;
 end
end

% ignore imaginary part of Berry curvature
lower_berry_curv = real(lower_berry_curv); upper_berry_curv = real(upper_berry_curv);

% compute Chern number by integrating Berry curvature 
dk = K1(2,2) - K1(1,1);
S_lower = sum(lower_berry_curv,1); S_upper = sum(upper_berry_curv,1);
Chern_lower = sum(S_lower)*dk*dk/(2*pi); Chern_upper = sum(S_upper)*dk*dk/(2*pi);

% compute minimal gap
almost_min = min(gap,[],1);
min_gap = min(almost_min);

% plot band functions 
figure
surf( K1, K2, lower_band ); hold on; surf( K1, K2, upper_band );
title(['Upper and lower band functions. Gap function: ',num2str(min_gap)])
% plot Berry curvature distribution: lower band
figure
surf( K1, K2, lower_berry_curv ); 
title(['Berry curvature: lower band. Chern number: ',num2str(Chern_lower)])
% plot Berry curvature distribution: upper band
figure
surf( K1, K2, upper_berry_curv );
title(['Berry curvature: upper band. Chern number: ',num2str(Chern_upper)])
