% constants
k_points = 100;
t = 1; % should be positive
d = 1; % delta prime
m = -2; % mu prime

% define plus band function
E_plus = '@(x,y,a,b,c) 2*a*sqrt( (b^2)*( ( sin(x) )^2 + ( sin (y) )^2 ) + ( c + cos( x ) + cos( y ) )^2 )';
E_plus = str2func(E_plus);

% form grids
[K1,K2] = meshgrid(linspace(0,2*pi,k_points),linspace(0,2*pi,k_points));
T = t*ones(size(K1)); D = d*ones(size(K1)); M = m*ones(size(K1));

% evaluate band function on grid
Band = arrayfun(E_plus,K1,K2,T,D,M);

% figure
figure
surf( K1, K2, Band ); hold on; surf( K1, K2, -Band );
