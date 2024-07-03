from create_and_save_contrast_maps_fdr0001 import create_and_save_contrast_maps_fdr0001
import os

data_path = r'G:\LC\Meditation_resting\results'
bg_img_file = 'C:\\Users\\Administrator\\nilearn_data\\icbm152_2009\\mni_icbm152_nlin_sym_09a\\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'

#brain_region = 'hcp_HIPPOCAMPUS_Vn2_eig'
#cognitive_task = 'IWRD_RTC'
significant_results = {
    "Uncorrected p < 0.05": [],
    "FDR-corrected p < 0.0001": [],
}

for brain_region_dir in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, brain_region_dir)):
        brain_region = brain_region_dir

        for cognitive_task_dir in os.listdir(os.path.join(data_path, brain_region)):
            if os.path.isdir(os.path.join(data_path, brain_region, cognitive_task_dir)):
                cognitive_task = cognitive_task_dir

                # 获取显著性信息
                uncorrected_significant, fdr_significant = create_and_save_contrast_maps_fdr0001(
                    data_path,
                    brain_region,
                    cognitive_task,
                    bg_img_file
                )
                
                # 记录显著结果
                if uncorrected_significant:
                    significant_results["Uncorrected p < 0.05"].append((brain_region, cognitive_task))
                if fdr_significant:
                    significant_results["FDR-corrected p < 0.0001"].append((brain_region, cognitive_task))
