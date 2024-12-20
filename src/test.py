import torch

if __name__ == '__main__':
    dict1 = torch.load(r"C:/Users/yilin/Desktop/epoch=210-step=650000.ckpt")["state_dict"]
    dict2 = torch.load(r"C:/Users/yilin/Desktop/epoch=126-step=750000.ckpt")["state_dict"]

    for key in dict1.keys():
        if "ae_model" not in key:
            continue
        assert torch.allclose(dict1[key], dict2[key], atol=1e-4), key

    pass