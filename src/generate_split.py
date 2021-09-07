import os
import numpy as np

if __name__ == '__main__':
    
    # train split
    
    names=[]
    for item in os.listdir(r"D:\Unreal_model_split\Shenzhen_dataset\SZU\training\image_2"):
        name=item.split(".")[0]
        names.append(name)

    split_mode="yrs_total"
    split_list=np.arange(len(names))
    np.random.shuffle(split_list)
    train_index=int(len(names)*0.5)
    train_list,val_list=split_list[:train_index],split_list[train_index:]
    with open("{}_split_train.txt".format(split_mode),"w") as f:
        for i in train_list:
            f.write("{}\n".format(names[i]))
    with open("{}_split_val.txt".format(split_mode),"w") as f:
        for i in val_list:
            f.write("{}\n".format(names[i]))
            
    # debug split

    names = []
    for item in os.listdir(r"D:\Unreal_model_split\Shenzhen_dataset\SZU\training\image_2"):
        name = item.split(".")[0]
        names.append(name)

    split_mode = "yrs_total"
    split_list = np.arange(len(names))
    np.random.shuffle(split_list)
    split_list = split_list[:500]
    train_index = int(len(split_list))
    val_index = int(len(split_list) * 0.2)
    
    train_list, val_list = split_list[:train_index], split_list[:val_index]
    with open("{}_split_train.txt".format(split_mode), "w") as f:
        for i in train_list:
            f.write("{}\n".format(names[i]))
    with open("{}_split_val.txt".format(split_mode), "w") as f:
        for i in val_list:
            f.write("{}\n".format(names[i]))

    # path = r"D:\Unreal_model_split\Shenzhen_dataset\SZU\training\calib"
    # for item in os.listdir(r"D:\Unreal_model_split\Shenzhen_dataset\SZU\training\calib"):
    #     with open(os.path.join(path, item), 'w') as f:
    #         f.write('P2: 400 0 400 0 0 400 400 0 0 0 1 0')


