import os
import numpy as np

if __name__ == '__main__':
    names=[]
    for item in os.listdir(r"/mnt/d/siga_21/SZU/training/image_2"):
        name=item.split(".")[0]
        names.append(name)

    split_mode="szu_total"
    split_list=np.arange(len(names))
    np.random.shuffle(split_list)
    train_index=int(len(names)*0.8)
    train_list,val_list=split_list[:train_index],split_list[train_index:]
    with open("{}_split_train.txt".format(split_mode),"w") as f:
        for i in train_list:
            f.write("{}\n".format(names[i]))
    with open("{}_split_val.txt".format(split_mode),"w") as f:
        for i in val_list:
            f.write("{}\n".format(names[i]))

