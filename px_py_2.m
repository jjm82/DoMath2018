% constants
k_points = 100;
t = 1; % should be positive 
d = 1; % delta (not prime) 
m = 10; % mu (not prime) 

% define Hamiltonian
H = '@(x,y,a,b,c) [ - c - 2*a*( cos(x) + cos(y) ) , b*( sin(x) - 1j*sin(y) ); b*( sin(x) + 1j*sin(y) ) , c + 2*a*( cos(x) + cos(y) ) ]';
%dk1H = '@(x,y,a,b,c) [ 2*a* cos(x) + cos(y) ) , b*( sin(x) - 1j*sin(y) ); b*( sin(x) + 1j*sin(y) ) , c + 2*a*( cos(x) + cos(y) ) ]'; TO DO
H = str2func(H);

% generate grid of k values
[K1,K2] = meshgrid(linspace(0,2*pi,k_points),linspace(0,2*pi,k_points));

% initialize band function evaluated on grid
Band = zeros(size(K1));
Curvature = zeros(size(K1)); 

% compute band function over grid
for i = 1:k_points;
 for j = 1:k_points;
  k1 = K1(i,i); k2 = K2(j,j);
  H_loc = H(k1,k2,t,d,m);
  [V,D] = eig(H_loc);
  [~,idx]=sort(diag(D));
  D = D(idx,idx); V = V(:,idx);
  Band(i,j) = D(1,1); 
  %Curvature(i,j) =  TO DO
 end
end

% plot band functions 
figure
surf( K1, K2, Band ); hold on; surf( K1, K2, -Band );
