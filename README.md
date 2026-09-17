# PCC-framework
Property-cluster-category framework is proposed to explain the principles of functional organization of human higher visual cortex

Our new work titled 'Abstract Property-selective Clusters Link Continuous Property Representations to Discrete Category-selective Regions in Human Higher Visual Cortex' has been accepted in Journal of Cognitive Neuroscience. And it will soon online.

## Abstract
Object representations in the human higher visual cortex (HVC) support complex recognition behaviors early in development, yet the principles linking continuous dimensional representations and discrete category-selective areas in the HVC remain incompletely understood. Here, we propose a ‘Property-Cluster-Category’ organization that bridges object conceptual dimensions and discrete category areas in the HVC. Using a large-scale naturalistic stimulus dataset and voxel-wise encoding methods, we analyzed the encoding patterns of a low-dimensional abstract property space and identified distinct brain clusters with shared cortical property profiles. These clusters broadly aligned with category-selective areas, suggesting that category regions can be understood as local peaks within a continuous property topology. We further tested whether this visual organization could emerge in a visual-only Topographic Deep Artificial Neural Network (TDANN) trained without semantic supervision. The model recapitulated property tuning for physical and biological dimensions but showed weaker affective tuning, suggesting that affective dimensions may require embodied or nonvisual experience. Finally, model-based lesion and stimulation analyses showed that TDANN units aligned with brain clusters contributed selectively to object classification. Together, these results provide a framework for understanding how the continuous property topology and discrete category selectivity are jointly organized in the human HVC, and suggest that abstract visual properties – which shape this topological organization – can be learned in part from statistical regularities in visual input.

We provide the code of each analyzing step following the pipeline used in our work.

## Object property space
### 1_property_relation.ipynb analyzed the property relations and build the low-dimensional property space using PCA.

## Property tuning in human brain by encoding method
### 2_prepare_data_for_encoding_model.ipynb prepared training and testing set of image property - brain response pairs.
### 3_vox_encode_prop_pcs.m was the training procedure of voxel-wise encoding model.
### 3_permu_test_pro_pcs.m was the testing procedure of voxel-wise encoding model.
### 4_plot_prop_pcs_result.py plotted encoding performance and property tuning map in human brain cortex.
### 4_result_plotting_prop_pcs.ipynb generated the property-selective clusters and analyzed their functions.

## Property-clusters-category relations analyzed on TDANN
### 5_model_feature_extraction.ipynb extracted THINGS image features from many visual model.
### 5_tdann-feature_extractor.py extracted THINGS image features from TDANN model.
### 5_prepare_tdann_data_for_encoding_model.ipynb prepared training and testing set of image property - TDANN feature pairs.
### 6_ann-brain-plsr.ipynb projected brain clusters on TDANN and analyzed their property tuning and representational map in model.
### 6_tdann-prop_analysis.ipynb analyzed the classification performance during the cluster-selective units were controlled.

## Data and model availability
THINGS object concept and image database and metadata about human cognitive ratings of object properties are available at https://osf.io/jum2f/; 
fMRI data set is provided at https://doi.org/10.25452/Figshare.plus.c.6161151.v1. 
In addition, the TDANN model can be accessible at https://github.com/neuroailab/TDANN.

