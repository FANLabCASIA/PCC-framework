%% Define data directory
maxNumCompThreads(20)
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

load([basedir, 'derivatives/linear_model/sub-',sub,'/prop_pcs/encoding_training_result.mat']) % 'thetaRecord', 'thetaFinal', 'validSquaError', 'CV', 'reguFinal'
clear validSquaError
clear CV
clear reguFinal
clear thetaRecord

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