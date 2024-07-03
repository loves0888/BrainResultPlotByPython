import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.plotting import plot_stat_map, find_cut_slices, plot_glass_brain
import os
from scipy import stats
from collections import Counter
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import loadmat

def create_and_save_contrast_maps_fdr0001(data_path: str, brain_region: str, cognitive_task: str,
                                 bg_img_file: str, desired_p_value: float = 0.05):
    """
    Create and save contrast maps for a given brain region and cognitive task, using data from the specified path.
    Perform statistical analysis and visualization for different p-value thresholds.

    Parameters:
    data_path (str): Path to the directory containing SPM results.    
    brain_region (str): Name of the brain region (e.g., 'hcp_HIPPOCAMPUS_Vn2_mag').
    cognitive_task (str): Name of the cognitive task (e.g., 'IWRD_RTC').
    bg_img_file (str): Path to the MNI152 template file.
    desired_p_value (float, optional): Desired significance level for p-values. Defaults to 0.05.

    Returns:
    None
    """
    
    # Step 1: Load data
    bg_img = nib.load(bg_img_file)
    spm_mat_file = os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\SPM.mat")
    t_image_file = os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\spmT_0001.nii")

    t_image = nib.load(t_image_file)
    spm_mat = loadmat(spm_mat_file)
    df = spm_mat['SPM']['xX'][0][0]['V'][0][0].shape[0] - spm_mat['SPM']['xX'][0][0]['X'][0][0].shape[1]

    raveled_t_values = t_image.get_fdata().ravel()
    
    non_zero_indices = raveled_t_values != 0
    non_zero_values = raveled_t_values[non_zero_indices]

    # 加载mask167图像
    mask_file = r'G:\LC\Meditation_resting\code\atlas\\AAL3\\Tr_AAL3v1_MNI2009c_2mm_ROI_167_mask.nii.gz'
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()  
    
    inverted_mask = ~mask_data.astype(bool)
     
    # 将mask数据ravel（拉平），以便与非零t值对应元素进行点乘
    raveled_mask = inverted_mask.ravel()
    
    # 对非零t值应用mask
    masked_t_values = non_zero_values * raveled_mask[non_zero_indices]
    
    # 加载mask168图像
    mask_file = r'G:\LC\Meditation_resting\code\atlas\\AAL3\\Tr_AAL3v1_MNI2009c_2mm_ROI_168_mask.nii.gz'
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()  
    
    inverted_mask = ~mask_data.astype(bool)
     
    # 将mask数据ravel（拉平），以便与非零t值对应元素进行点乘
    raveled_mask = inverted_mask.ravel()
    
    # 对非零t值应用mask
    masked_t_values = masked_t_values * raveled_mask[non_zero_indices]
    
    non_zero_values = masked_t_values
    
    # Step 2: Compute p-values and t-thresholds
    p_values = stats.t.sf(np.abs(non_zero_values), df) * 2

    # All p-values
    desired_t_threshold_all = 0
    # p-values < 0.05
    desired_t_threshold = stats.norm.ppf(1 - desired_p_value)  # For one-sided test (positive)
 

    # 定义文件路径和文件名
    txt_file_path = os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\{brain_region}_{cognitive_task}_no_p_result.txt")
    
    # # 如果小于 0.05 的 p 值的数量为 0，输出 txt 文件
    # if count_below_005 == 0:
    #     with open(txt_file_path, 'w') as f:
    #         f.write("没有找到 p 值小于 0.05 的原始 p 值。")
    #     print("文件已保存至:", txt_file_path)
    #     desired_t_threshold = None
        
    # FDR-corrected p < 0.05
    fdr_results = fdrcorrection(p_values, alpha=desired_p_value)
    fdr_corrected_p_values = fdr_results[1]
    selected_p_values = p_values[fdr_corrected_p_values < 0.001]
    if len(selected_p_values) > 0:
        threshold_p_value = np.max(selected_p_values)
        desired_t_threshold_fdr = stats.t.isf(threshold_p_value / 2, df)
    else:
        txt_file_path = os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\{brain_region}_{cognitive_task}_no_FDR_p_result.txt")
        desired_t_threshold_fdr = None
        with open(txt_file_path, 'w') as f:
            f.write("没有找到FDR p 值小于 0.001 的 p 值。")
        print("文件已保存至:", txt_file_path)
        
        
    # Step 3: Plot and save contrast maps
    max_t_value = 10
    fig = None
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0, wspace=0, top=0.9, bottom=0.1, left=0.1, right=0.9)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    titles = ['All p-values', 'Uncorrected p < 0.05', 'FDR-corrected p < 0.001']
    names = ['All p', 'Uncorrected p', 'FDR-corrected p001']
    for ax, threshold, title, name in zip(axs, [desired_t_threshold_all, desired_t_threshold, desired_t_threshold_fdr], titles, names):
        if threshold is not None:
            plot_glass_brain(t_image, threshold=threshold, vmax=max_t_value, display_mode='lyrz', black_bg=True,
                              cmap='RdBu_r', colorbar=True, title=title, plot_abs=False, axes=ax)
            #plt.savefig(os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\my_contrast_map_{brain_region}_{cognitive_task}_{name.replace(' ', '_')}_glass.png"), dpi=600) 
        else:
            ax.clear()
            ax.set_axis_off()

    plt.savefig(os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\my_contrast_maps_{brain_region}_{cognitive_task}_glass_togetherfdr001.png"), dpi=600)
    plt.close(fig)  # Optionally close the figure after saving to release resources
    fig = None
    for threshold, title, name in zip([desired_t_threshold_all, desired_t_threshold, desired_t_threshold_fdr], titles, names):
        if threshold is not None:
            plot_glass_brain(t_image, threshold=threshold, vmax=max_t_value, display_mode='lyrz', black_bg=True,
                              cmap='RdBu_r', colorbar=True, title=title, plot_abs=False)    
            plt.savefig(os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\my_contrast_map_{brain_region}_{cognitive_task}_{name.replace(' ', '_')}_glass.png"), dpi=600)
            plt.close(fig)            
    fig = None       
    for threshold, title, name in zip([desired_t_threshold_all, desired_t_threshold, desired_t_threshold_fdr], titles, names):
        if threshold is not None:
            plot_stat_map(t_image, threshold=threshold, vmax=max_t_value, display_mode='mosaic', black_bg=True,
                                  cmap='RdBu_r', colorbar=True, title=title)           
            filename = os.path.join(data_path, f"{brain_region}\\{cognitive_task}\\my_contrast_map_{brain_region}_{cognitive_task}_{name.replace(' ', '_')}_xyz.png")
            plt.savefig(filename, dpi=600)
            plt.close(fig)
            
            
    return desired_t_threshold is not None, desired_t_threshold_fdr is not None

if __name__ == "__main__":
    data_path = r'G:\LC\Meditation_resting\results'   
    brain_region = 'SPMonesample'
    cognitive_task = 'SPMresult_1'
    bg_img_file =  r'G:\LC\Meditation_resting\code\atlas\\mni_icbm152_nlin_asym_09c\\Tr_mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'   
    create_and_save_contrast_maps_fdr0001(data_path, brain_region, cognitive_task, bg_img_file)