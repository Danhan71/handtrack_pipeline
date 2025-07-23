    
% pts_dir = '/home/danhan/code/test_data/supp_reg_pts/fly1_fly2_xy_pts_r.csv';
% coefs_dir = '/home/danhan/Documents/pipeline/dlt_coeffs/220914_f12_dlc/dltCoefs.csv';

pts_dir = '/home/danhan/code/test_data/supp_reg_pts/flea_bfs1_xy_pts.csv';
coefs_dir = '/home/danhan/Documents/pipeline/dlt_coeffs/220914_flea_bfs1_dlc/dltCoefs.csv';

pts = readmatrix(pts_dir);
coefs = readmatrix(coefs_dir);

save_path = '/home/danhan/code/test_data/supp_reg_pts/fleabfs1_xyz.csv';
% save_path = '/home/danhan/code/test_data/supp_reg_pts/fly1fly2_xyz_r.csv';
[xyz,~] = dlt_reconstruct_standalone(coefs,pts);
writematrix(xyz,save_path)