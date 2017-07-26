function [rotatedvect]=RotateEuler(theta, phi, vector)
%vector : vect ligne
theta = -theta;
rot_theta=[cosd(theta) 0 -sind(theta)  ;  0 1 0  ; sind(theta) 0 cosd(theta)];
%rot_phi=[cosd(phi) -sind(phi) 0 ;  sind(phi) cosd(phi) 0 ;0 0 1];
rot_phi=[1 0 0; 0 cosd(phi) -sind(phi);  0 sind(phi) cosd(phi)];

rotatedvect=(rot_phi*rot_theta*vector')';

end