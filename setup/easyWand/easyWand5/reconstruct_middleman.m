function xyz = reconstruct_middleman(out)
    if false
        coefs = readmatrix([out,'/dltCoefs.txt']);
        data_pattern = fullfile(out,'/pts_t*.txt');
        assignin('base','data_pattern',data_pattern);
        cam_pts = dir(data_pattern);
    else
        pts = readmatrix([out,'/xypts.csv']);
        coefs = readmatrix([out,'/dltCoefs.csv']);
    end

    save_path = [out,'/xyz_pts.txt'];
    [xyz,~] = dlt_reconstruct_standalone(coefs,pts);
    writematrix(xyz,save_path)
    % for k=1:length(cam_pts)
    %     base_file_name = cam_pts(k).name;
    %     full_path = fullfile(out,base_file_name);
    %     save_path = fullfile(out,['xyz_',base_file_name]);
    %     pts = readmatrix(full_path);
    % 
    %     [xyz,~] = dlt_reconstruct_standalone(coefs,pts);
    %     writematrix(xyz,save_path)
    % end