function xyz = triangulate_middle_man(path_calibration, out)
% Function to call from outside to use easywand triangulate function.
% See triangulate function for proper credit.
%INPUTS:
% path_calibration, string, path to calibration data being used. Directory
% should contain:
%   easyWandData.mat file from easy wand calib saved with this name
%   [cam_name]_calibration_mtx.txt calib matrix from checkerboard
%   structured like [[fx,0,cx],
%                    [0,fy,cy],
%                    [0, 0, 1 ]]
% out, path to where temp data is stored, automatically generated based on
% pipe_path. 'out' becuase triangulated data will also be stored here.
% should contain:
%       cams.txt with newline delimited cam names in order
%       pts_t{trial}.txt files with wands pts for each trial
% cam_list, cam_list IN ORDER. Order should be automatic as long as you are
% using correct functions through the code. Automatically read from
% path_data

%This function will take in all info, extract the rotation and translation
%matrices, nomrlaize the points, triangulate the points, and return these
%points to python
%
%Exampe inputs:
% path_calibration = '/home/danhan/Documents/pipeline/wand_data/220914';
% path_data = '/home/danhan/Documents/pipeline/temp_matlab_files'


path_wand_data = [path_calibration,'/easyWandData.mat'];
data = load(path_wand_data);
data=data.easyWandData;

% get camera matrices
cam_list = readcell([out,'/cams.txt']);
cam_list = cam_list;
for i=1:length(cam_list)
    camnames{i} = matlab.lang.makeValidName(cam_list{i,1});
    cam_mats.(camnames{i})=readmatrix(sprintf('%s/%s_calibration_mtx.txt',path_calibration,cam_list{i}));
end

% load data files
data_pattern = fullfile(out,'/pts_t*.txt');
cam_pts = dir(data_pattern);

% Rotation and translation matrices, but remove last cam to work with
% traingulate function
R = data.rotationMatrices;
R = R(:,:,1:3);
T = data.translationVector;
T = T(:,:,1:3);

% pts = readmatrix(path_data);
%Loop thorugh trials and cameras
for k=1:length(cam_pts)
    base_file_name = cam_pts(k).name;
    full_path = fullfile(out,base_file_name);
    save_path = fullfile(out,['xyz_',base_file_name]);
    pts = readmatrix(full_path);
    pts_norm=zeros(size(pts));
    for i= 1:length(cam_list)
        this_mat = cam_mats.(camnames{i});
        %normalize pts: (pt-center)/focal_length
        pts_norm(:,(2*i-1)) = (pts(:,(2*i-1)) - this_mat(1,3))/this_mat(1,1);
        pts_norm(:,2*i) = (pts(:,2*i) - this_mat(2,3))/this_mat(2,2);
    end
    % trinagulate pts with all cams in this trial, save in out with
    % t[trial]xyz
    xyz = triangulate_standalone(R,T,pts_norm);
    % zxy = zeros(size(xyz))
    % assignin('base','xyz',xyz)
    % zxy(:,[1,2,3]) = xyz(:,[2,3,1]);
    % assignin('base','zxy',zxy)
    % display(save_path)
    writematrix(xyz,save_path)
end
% %%
% path_calibration = '/home/danhan/Documents/pipeline/wand_data/220914'
% out = '/home/danhan/Documents/pipeline/temp_matlab_files/Pancho/221015_dircolor1'
% triangulate_middle_man(path_calibration,out)



