% code which plots band functions of the p_x + i p_y model using explicit formula

% settings
clear all;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex'); set(groot, 'defaulttextinterpreter','latex');

% parameters of the model. NB band functions degenerate when mu/2t = 2,0,-2
t = 1; % assumed positive
d = 1; % delta
m = 2; % mu

% convert to delta' and mu'
d = d/(2*t);
m = m/(2*t);

% numerical parameter: number of k points in each direction
k_points = 60;

% define plus band function (assuming t positive)
E_plus = '@(x,y,a,b,c) 2*a*sqrt( (b^2)*( ( sin(x) )^2 + ( sin (y) )^2 ) + ( c + cos( x ) + cos( y ) )^2 )';
E_plus = str2func(E_plus);

% form grids
[K1,K2] = meshgrid(linspace(0,2*pi,k_points),linspace(0,2*pi,k_points));
T = t*ones(size(K1)); D = d*ones(size(K1)); M = m*ones(size(K1));

% evaluate band function on grid
Band = arrayfun(E_plus,K1,K2,T,D,M);

% plot lower and upper band functions 
figure
surf( K1, K2, Band ); hold on; surf( K1, K2, -Band );
title('Upper and lower band functions')
