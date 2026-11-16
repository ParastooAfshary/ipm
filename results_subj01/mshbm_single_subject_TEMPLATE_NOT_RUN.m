% TEMPLATE ONLY. NOT RUN. Requires MATLAB, CBIG, FreeSurfer dependencies and group priors.
% Source: CBIG Kong2019_MSHBM/examples/README.md (single-subject section)
% Set up CBIG_CODE_DIR and add CBIG paths before running.
params = struct();
params.project_dir = 'C:/Users/USER/Desktop/ipm-1sub/results_subj01/mshbm_output';
params.censor_list = {'C:/Users/USER/Desktop/ipm-1sub/subj01_sess1_bld002_FDRMS0.2_DVARS50_motion_outliers.txt'};
params.lh_fMRI_list = {'C:/Users/USER/Desktop/ipm-1sub/lh.subj01_sess1_bld002_rest_skip4_stc_mc_residc_interp_FDRMS0.2_DVARS50_bp_0.009_0.08_fs6_sm6.nii.gz'};
params.rh_fMRI_list = {'C:/Users/USER/Desktop/ipm-1sub/rh.subj01_sess1_bld002_rest_skip4_stc_mc_residc_interp_FDRMS0.2_DVARS50_bp_0.009_0.08_fs6_sm6.nii.gz'};
params.target_mesh = 'fsaverage6';
params.w = '50';
params.c = '50';
[lh_labels, rh_labels] = CBIG_MSHBM_parcellation_single_subject(params);
