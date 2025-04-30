import numpy as np
import pandas as pd
import os,sys
from pathlib import Path
from tqdm import tqdm
import torch, ray 

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

root_path = Path(r"/mnt/d/yilin/img2brep/deepcad_v6_txt")

@ray.remote(num_gpus=1)
def worker(prefixes, v_id):
    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = SentenceTransformer(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    
    with torch.no_grad():
        for prefix in tqdm(prefixes, disable=v_id!=0):
            if not (root_path/prefix/"text.txt").exists():
                continue
            data = open(root_path/prefix/"text.txt").readlines()
            abs = data[0].strip()
            beg = data[1].strip()
            inter = data[2].strip()
            expert = data[3].strip()
            
            # batch_dict = tokenizer([abs, beg], max_length=8192, padding=True, truncation=True, return_tensors='pt')
            # outputs = model(**batch_dict)
            # embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            embeddings = model.encode([abs, beg, inter, expert])
            np.save(root_path/prefix/"text_feat.npy", embeddings)

if __name__ == "__main__":
    prefixes = os.listdir(root_path)
    num_workers = 1
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    
    ray.init(
        num_gpus=num_workers,
        # local_mode=True,
    )

    tasks = np.array_split(np.asarray(prefixes), num_workers)
    for i, item in enumerate(tasks):
        tasks[i] = worker.remote(item, i)
    for prefix in tqdm(tasks):
        ray.get(prefix)
    print("Done")            
