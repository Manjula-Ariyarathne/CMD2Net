from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as tfs
import numpy as np
import torch


class change_detection_dataset_local(Dataset):
    def __init__(self,path, transform) -> None:
        super().__init__()
        self.pre_change_path=os.path.join(path,"imageA")
        self.post_change_path=os.path.join(path,"imageB")
        self.change_gt_path=os.path.join(path,"label")
        self.fname_list=os.listdir(self.pre_change_path)
        self.transform=transform
    def __getitem__(self, index):
        fname = self.fname_list[index]
        pre_img  = Image.open(os.path.join(self.pre_change_path, fname)).convert("RGB")
        post_img = Image.open(os.path.join(self.post_change_path, fname)).convert("RGB")

        gt_img = Image.open(os.path.join(self.change_gt_path, fname)).convert("L")
        change_gt = torch.from_numpy(np.array(gt_img)).long()
        if change_gt.max() > 1:
            change_gt = (change_gt > 0).long()

        pre_tensor  = self.transform(pre_img)
        post_tensor = self.transform(post_img)
        gt_tensor   = change_gt   # <-- keep integer mask

        return {'pre': pre_tensor, 'post': post_tensor, 'gt': gt_tensor, 'fname': fname}

    def __len__(self):
        return len(self.fname_list)
    
class change_detection_dataset_HG(Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"]).squeeze().long()
        return {'pre':imageA,'post':imageB,'gt':label,'fname':str(index)+".png"}