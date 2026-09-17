# fit voxelwise linear model by using obj prop principal components to predict fmri signals
# plot encoding model performance and property map

import h5py
import math
import cortex
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from os.path import join as pjoin
from nilearn.image import load_img
from nilearn.masking import apply_mask, unmask

# basic dir
sub = '01'
basedir = r'/n02dat01/users/qdzhao/THINGS/'
result_dir = pjoin(basedir, r'property_analysis/derivatives/linear_model', 'sub-'+sub)
pro_nm = ['manmade', 'precious', 'animacy', 'heavy', 'natural', 'moves',
           'grasp', 'hold','be_moved', 'pleasant', 'deprecatedArousing', 'size']

def surf_plot(data, savefig='', dmax=1, dmin=0, color='hot'):
    # turn vec to volume
    bmask_dir = pjoin(basedir, r'THINGS_fMRI/brainmasks')
    bmask_f = pjoin(bmask_dir, f'sub-{sub}_space-T1w_brainmask.nii.gz')
    # use the brain mask to turn your results array into a 3D image
    data_img = unmask(data, bmask_f)

    # perf_p = np.swapaxes(load_img(pjoin(result_dir, f'mode_performance_p.nii.gz')).get_fdata(), 0, -1)
    data_3dimg = np.swapaxes(load_img(data_img).get_fdata(), 0, -1)
    #if threshold:
        # data_3dimg[data_3dimg<0] = np.nan
    #vol_data = cortex.Volume(data_3dimg, 'S1', 'align_auto', cmap=color,vmax=np.max(np.nan_to_num(data_3dimg)), vmin=0)
    vol_data = cortex.Volume(data_3dimg, 'S1', 'align_auto', cmap=color,vmax=dmax, vmin=dmin)
    # plot with pycortex
    fig = plt.figure(figsize=(8,4))
    roi_lists=['V1','V2','V3','OFA','FFA','pSTS','EBA','PPA','OPA','MPA','LOC']
    # cortex.quickshow(vol_data, pixelwise=True, nanmean=True, colorbar_location='left', with_rois=False, fig=fig)
    cortex.quickflat.make_figure(vol_data, colorbar_location='center', roi_list=roi_lists, with_labels=False, with_boders=True, with_curvature=True, curvature_contrast=0.5, curvature_brightness=0.5, curvature_threshold=True, fig=fig)
    # plt.title('linear model performance ')
    plt.savefig(pjoin(result_dir, r'prop_pcs/plot', str(savefig) + '.jpg'), dpi=300)
    # plt.show()
    plt.close(fig)

# load the model result
# significant p value of training set
adjp_f = pjoin(result_dir, r'prop_pcs', f'cross_validation_permutation_result.mat')
adjp = h5py.File(adjp_f)
print(adjp.keys())
adjp_ = adjp['adj_p']
adj_p = np.squeeze(adjp_)
adj_p_ = np.zeros(len(adj_p))
for i in range(len(adj_p)):
    adj_p_[i] = -math.log((adj_p[i]),10) # log10(p)
adj_p= np.where(adj_p_>1.3,adj_p_,0) # threshold=0.05
# adj_p_(adj_p_<1.3) = np.nan
print(adj_p.shape)
surf_plot(adj_p, 'training_perf_p', dmax=np.max(adj_p))

# prediction accuracy correlation r of whole brain
ped_r_f = pjoin(result_dir, r'prop_pcs', f'model_test_result.mat')
ped_r = h5py.File(ped_r_f)
print(ped_r.keys())
acc_r = ped_r['R']
acc_r_ = np.squeeze(acc_r)
print(acc_r_.shape)
surf_plot(acc_r_, 'test_perf_all_r', dmax=np.max(acc_r_))

# significant r with p < .05
cor_r = np.zeros(len(acc_r_))
cor_r= np.where(adj_p_>1.3,acc_r_,0)
surf_plot(cor_r,'test_perf_signif_r', dmax=np.max(acc_r_))

# weight matrix
weigh_f = pjoin(result_dir, r'prop_pcs', f'encoding_training_result.mat')
weigh_ = h5py.File(weigh_f)
print(weigh_.keys())
weigh_mrx = weigh_['thetaFinal']
weigh_mrx = np.array(weigh_mrx)
print('transform matrix shape ', weigh_mrx.shape) # (v,5)
pro_pc = ['prop_pc1','prop_pc2','prop_pc3','prop_pc4','prop_pc5']
# ma = np.max(weigh_mrx)
ma = 0.003
for i in range(5):
    weigh_mrx_i = np.squeeze(weigh_mrx[:,i])
    fname = pro_pc[i] + '_weights'
    surf_plot(weigh_mrx_i, fname, dmax=ma, dmin=-ma, color='BuBkRd')





