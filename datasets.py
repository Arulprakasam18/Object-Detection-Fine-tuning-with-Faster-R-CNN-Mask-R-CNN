import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as T

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert the mask to a binary format
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        num_objs = torch.max(mask)

        # Create target dictionary
        target = {}
        target["boxes"] = self.get_bbox(mask)
        target["labels"] = torch.ones(num_objs, dtype=torch.int64)
        target["masks"] = mask
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def get_bbox(self, mask):
        # Find bounding box for the object
        pos = torch.where(mask > 0)
        xmin = torch.min(pos[1])
        xmax = torch.max(pos[1])
        ymin = torch.min(pos[0])
        ymax = torch.max(pos[0])

        return torch.tensor([xmin, ymin, xmax, ymax])

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
