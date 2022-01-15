import numpy as np
from plyfile import PlyData

v_error_point_cloud = [
    r"D:\Projects\Reconstructability\training_data\v2\qiao_fine_ds_70\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\qiao_fine_ds_80\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xiaozhen_fine_ds_70\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xiaozhen_fine_ds_80\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xiaozhen_fine_ds_90\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xuexiao_fine_ds_70\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xuexiao_fine_ds_80\accuracy_projected.ply",
    r"D:\Projects\Reconstructability\training_data\v2\xuexiao_fine_ds_90\accuracy_projected.ply",
]

if __name__ == '__main__':
    avg_errors=[]
    for v_path in v_error_point_cloud:
        with open(v_path, "rb") as f:
            plydata = PlyData.read(f)
            sum_error_list = plydata['vertex']['sum_error'].copy()
            num_list = plydata['vertex']['num'].copy()
            avg_errors.append(())
            item = sum_error_list / (num_list + 1e-6)

            print("==================================")
            print(v_path)
            print("{}/{}".format((item==0).sum(),item.shape[0]))
            item = item[item != 0]
            item.sort()
            num=item.shape[0]
            print("80%: {}; 90%: {}; ".format(item[num//10*8],item[num//10*9]))

