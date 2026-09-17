% This script is used to build the linear encoding model between property 3 principal components
% features and MRI responses on natural image stimuli

maxNumCompThreads(20)
%% Define data directory
basedir = '/n02dat01/users/qdzhao/THINGS/property_analysis/';
codedir = '/n02dat01/users/qdzhao/THINGS/property_analysis/code/';
addpath(genpath(codedir))
%% Training encoding model with ridge regression
sub = '03';
load([basedir, 'data/sub-',sub,'/betas_concept_v2.mat'])  %% 'betas_train':betas_train(648,V),'betas_test':betas_test(72,V)
load([basedir, 'data/sub-',sub,'/obj720_pro_pcs.mat']) %% 'prop_train':prop_train(648,12),'prop_test':prop_test(72,12)

% Standardize the predictors (word2vec features) for linear regression
m = mean(prop_train);
s = std(prop_train);
% save([basedir, 'encoding_dataset/training_statistics.mat'], 'm' ,'s')

% Standardize the predictors (word2vec features) for linear regression
prop_train_std = prop_train;
for i = 1:5
    prop_train_std(:,i) = (prop_train(:,i) - m(i))./s(i);
end
prop_test_std = prop_test;% 72*5
for i = 1:5
    prop_test_std(:,i) = (prop_test(:,i) - m(i))./s(i);
end

% Optional: change data to single precision
betas_train = single(betas_train); %648*#VOXEL
prop_train_std = single(prop_train_std); %648*5

% create a list of potential regularization parameters
k = 0:0.5:5;
lambda = 10.^(k);
reguList = lambda;
crossNum = 10;


% prop_name = ["manmade", "precious", "animacy", "heavy", "natural", "moves", "grasp", "hold","be_moved", "pleasant", "deprecatedArousing", "size"];
% training


[thetaFinal, validSquaError, CV, reguFinal, thetaRecord] = encoding(betas_train, prop_train_std, reguList, crossNum);
f_path = fullfile(basedir, sprintf(['derivatives/linear_model/sub-',sub,'/prop_pcs/encoding_training_result.mat']));
save(f_path, 'thetaRecord', 'thetaFinal', 'validSquaError', 'CV', 'reguFinal', '-v7.3');
clear validSquaError
clear CV
clear reguFinal
clear thetaRecord

%% Cross-validating encoding model
% set parameters
trial_num = 100000;
%trial_num = 10;
window = 30;
cross_num = 10;
operation_effi = true;
% optional: using gpu to accelerate
use_gpu = true;
gpu_device = 1;
if use_gpu
	input = gpuArray(prop_train_std);
	fmri_signal = gpuArray(betas_train);
end

% run permutation test
[avg_r, count, ~] = validation_permute(input, fmri_signal, trial_num, window, ...
	'regularization', 10.0, 'cross_num', cross_num, 'operation_effi', operation_effi);

% statistics
permu_record = gather(count);
trial_sum_all = gather(trial_num);
P = (permu_record+1)/(trial_sum_all+1); %calculate p value
% multiple correction
[h, ~, ~, adj_p]=fdr_bh(P,0.05,'dep','yes'); %fdr correction with q<0.05
mask = (adj_p<0.05)'; % a map of "semantic system" that contains significantly predictable voxels
f_path = fullfile(basedir, sprintf(['derivatives/linear_model/sub-',sub,'/prop_pcs/cross_validation_permutation_result.mat']));
save(f_path, 'adj_p', 'mask', 'trial_sum_all', '-v7.3')
clear permu_record
clear P
clear adj_p
clear mask
%%


%% Testing encoding model

% basedir = '/n02dat01/users/qdzhao/THINGS/property_analysis/';
% load([basedir, 'data/betas_concept.mat'])  %% 'betas_train':betas_train,'betas_test':betas_test
% load([basedir, 'data/obj_720_pro.mat']) %% 'prop_train':prop_train,'prop_test':prop_test
% load([basedir,'derivatives/encoding_training_result.mat'])

% m = mean(prop_train);
% s = std(prop_train);

X_pred = prop_test_std*thetaFinal; %72*#VOXEL

% temporally smooth betas_test
% betas_test_sm = betas_test;
% for i = 1:size(betas_test,1)
%     betas_test_sm(i,:) = smooth(betas_test(i,:));
% end

% calculate Pearson correlation between true and predicted testing data
R = correffi(betas_test,X_pred);
f_path = fullfile(basedir, sprintf(['derivatives/linear_model/sub-',sub,'/prop_pcs/model_test_result.mat']));
save(f_path, 'X_pred', 'betas_test', 'R', '-v7.3')

%% Permutation test for encoding performance on testing data
% test_fmri:  voxel-wise time series for testing data (size: 1 x time_length x voxel_size)
% Note that testing data temporal length after offset 4 is 540
[timeSize1, fMRISize] = size(betas_test);
test_fmri = zeros(1, timeSize1, fMRISize);
test_fmri(1,:,:) = betas_test;
% test_weight: encoding parameters (embedding_size x voxel_size)
test_weight = thetaFinal;
% test_conca_sig: word2vec time series for testing data (size: 1 x time_length x embedding_size)
[timeSize2, featureDim] = size(prop_test); % (72,5)
test_concat_sig = zeros(1, timeSize2, featureDim);
test_concat_sig(1,:,:) = prop_test_std;

% permutation test
window = 30;
trial_num = 80000;
% trial_num = 10;
[permu_record, ~] = test_permute(test_weight, test_fmri, test_concat_sig, 'window',window,'trial_num',trial_num,'usegpu',1);
% multiple correction
P = (permu_record+1)/(trial_num+1);
[h, ~, ~, adj_p]=fdr_bh(P,0.05,'dep','yes');
mask = (adj_p<0.05)';
f_path = fullfile(basedir, sprintf(['derivatives/linear_model/sub-',sub,'/prop_pcs/testing_permutation_result.mat']));
save(f_path, 'window', 'trial_num', 'permu_record', 'P', 'adj_p', 'mask');


%%