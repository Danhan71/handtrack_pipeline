function [xyz,xyzR] = triangulate_standalone(R,T,ptNorm)

% function [xyz,xyzR] = triangulate_v3(R,T,ptNorm)
%
% Combines the normalized coordinates, camera rotation and translation to
% of k cameras to give an xyz coordinate - operates in a purely linear
% manner and is therefore not optimal in the sense of minimizing
% reprojection error in the two cameras.  In fact, this is a pure Euclidian
% ray-midpoint triangulation routine of the type distinctly not favored by
% Hartley.  On the other hand, practical tests conducted with it and other
% triangulation methods show it to be superior for wand-type calibration.
% This function is similar in concept to triangulate_v1, but deals
% correctly with 3 or more cameras.  Based off of the discussion at:
% http://www.multires.caltech.edu/teaching/courses/3DP/ftp/98/hw/1/triangul
% ation.ps
%
% This triangulation function is faster and more accurate than _v2 but also
% occasionally not stable for inputs where the cameras are close to a
% degnerate condition with the optical axes separated by 180deg.
%
%
% Inputs:
%  R - rotation matrix between the cameras   [3,3,k-1] matrix
%  T - translation vector between the two cameras   [3,1,k-1] vector
%  ptNorm - normalized coordinates [principal point subtracted & divided by
%          focal length]   [n,k*2] array
%
% Outputs:
%  xyz - [n,3] array of xyz coordinates
%  xyzR - [n,3,k-1] array of xyz coordinates in the reference frame of
%  non-primary cameras
%
% Ty Hedrick, 2009-09-16

% nPts=size(ptNorm,1); % number of points
nCams=size(R,3)+1; % number of cameras

% note on conventions: the input R and T arrays are for cameras 1 to k-1,
% the final camera is assumed to have R=eye(3) and T=zeros(3,1).

R(:,:,end+1)=eye(3);
T(:,:,end+1)=zeros(1,3);

% this method cannot be easily adapted to more than 2 cameras together, so
% we need to do every possible pair and then get the group mean
bc=binarycombinations(nCams);
bc=bc(sum(bc,2)==2,:);
bc(bc(:,end)==0,:)=[]; % remove combos
xyz = zeros(size(ptNorm,1),3,size(bc,1))*NaN;
for i=1:size(bc,1)
  idx=find(bc(i,:)==true);
  pdx=sort([idx*2-1,idx*2]);
  xyz(:,:,i)=triangulate_v3int(R(:,:,idx),T(:,:,idx),ptNorm(:,pdx));
end

% get mean of all combinations
xyz=nanmean(xyz,3);

% transform to the other cameras
xyzR = zeros(size(xyz))*NaN;
for i=1:nCams-1
  xyzR(:,:,i)=(R(:,:,i)*(xyz'+ repmat(T(:,:,i)',1,size(xyz,1))))';
end

function [xyz] = triangulate_v3int(R,T,ptNorm)

% function [xyz] = triangulate_v3int(R,T,ptNorm)
%
% Internal function that performs the triangulation operation for any two
% cameras, finishes by projecting the 3D point back into the space of the
% base camera with R=eye(3) and T=zeros(1,3).

% make sure that our 2nd camera is a "base" camera with R=eye(3) and
% T=zeros(1,3)
R2(:,:,1)=R(:,:,2)'*R(:,:,1)*R(:,:,2)';
R2(:,:,2)=eye(3);
T2(:,:,1)=-R(:,:,2)'*R(:,:,1)*R(:,:,2)'*T(:,:,2)'+R(:,:,2)'*T(:,:,1)'-R(:,:,2)'*T(:,:,2)';
T2(:,:,2)=zeros(3,1);

% solving based on camera 2
%
% get the transpose (inverse) of the 2nd rotation matrix
R_2T=R2(:,:,2)';

tVec=repmat(R_2T*(T2(:,:,1)-T2(:,:,2)),1,size(ptNorm,1));

% extract homogenous coordinates for cameras 1 & 2
pts_1=ptNorm(:,1:2)';
pts_1(3,:)=1;
pts_2=ptNorm(:,3:4)';
pts_2(3,:)=1;

alpha_2=-R_2T*R2(:,:,1)*pts_2;

% create numerator and denominator for explicit expression of the depth
% vector Z_2
num=dot(pts_1,pts_1) .* dot(alpha_2,tVec) - dot(alpha_2,pts_1) .* dot(pts_1,tVec);
den=dot(pts_1,pts_1) .* dot(alpha_2,alpha_2) - dot(alpha_2,pts_1) .* dot(alpha_2,pts_1);

% depth vector
Z_2=num ./ den;

% get 3D coordinates in the camera #2 view
xyz2=pts_2 .* (ones(3,1) * Z_2);


% solving based on camera 1
%
% get the transpose (inverse) of the 2nd rotation matrix
R_1T=R2(:,:,1)';

tVec=repmat(R_1T*(T2(:,:,2)-T2(:,:,1)),1,size(ptNorm,1));

alpha_1=-R_1T*R2(:,:,2)*pts_1;

% create numerator and denominator for explicit expression of the depth
% vector Z_2
num=dot(pts_2,pts_2) .* dot(alpha_1,tVec) - dot(alpha_1,pts_2) .* dot(pts_2,tVec);
den=dot(pts_2,pts_2) .* dot(alpha_1,alpha_1) - dot(alpha_1,pts_2) .* dot(alpha_1,pts_2);

% depth vector
Z_1=num ./ den;

% get 3D coordinates in the camera #1 view
xyz1=pts_1 .* (ones(3,1) * Z_1);

% transform to the view of camera #2 for comparison
xyz1t=R2(:,:,1)'*(xyz1-repmat(T2(:,:,1),1,size(xyz1,2)));

% get the mean of the view in camera 2 based on both cameras 1 and 2
xyz_m=(xyz2+xyz1t)./2;

% transform these coordinates back to the viewpoint of a neutral camera
% with R=eye(3) and T=zeros(3,1)
xyz=(R(:,:,2)'*(xyz_m-repmat(T(:,:,2)',1,size(xyz_m,2))))';

function [y]=binarycombinations(n)

% function [y]=binarycombinations(n)
%
% Create a matrix of 2^n rows and n columns with a unique binary state in
% each row.  The first row is always all zeros and the final row all ones.
% Works through the possibilities in order, with all possibilities
% including only one "1" occuring before any possibilities with two "1"s
% and so on.
%
% Ty Hedrick

num_hyp=2^n;			% generate the set of hypotheses
y=zeros(n,num_hyp);		% all possible bit combos
for index=1:n
  y(index,:)=(-1).^ceil((1:num_hyp)/(2^(index-1)));
end

% change -1s to 0s and rotate
idx=logical(y==-1);
y(idx)=0;
y=y';

% sort
y(:,end+1)=y*ones(n,1);
y=sortrows(y,n+1);
y(:,end)=[];
