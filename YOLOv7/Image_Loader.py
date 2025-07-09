import os
import torchvision
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


test_ClASS = ['person','cyclist','car','truck','bus']
target_size = (640,640)
def resize_aspect_ratio(image):
    width, height = image.size
    target_width, target_height = target_size
    #计算缩放比
    scale = min(target_width/width,target_height/height)
    #计算新的宽高
    new_width = int(width * scale)
    new_height = int(height * scale)
    #等比缩放
    resize = torchvision.transforms.Resize((new_width,new_height))
    resized_image = resize(image)
    return resized_image,scale,(width,height)
def pad_to_target_size(image):
    width, height = image.size
    target_width, target_height = target_size
    #计算需要补零的宽度和高度
    padding_left = (target_width - width) // 2
    padding_top = (target_height - height) // 2
    padding_right = target_width - width - padding_left
    padding_bottom = target_height - height - padding_top
    #补零操作
    padding = torchvision.transforms.Pad(padding=(padding_left,padding_top,padding_right,padding_bottom),fill=0)
    padded_image = padding(image)
    return padded_image,(padding_left,padding_top)
resize_transform = torchvision.transforms.Lambda(resize_aspect_ratio)
pad_transform = torchvision.transforms.Lambda(pad_to_target_size)
data_transforms = torchvision.transforms.Compose([
    resize_transform,
    pad_transform,
    torchvision.transforms.ToTensor()
])
class Loader(Dataset):
    def __init__(self,image_ir_dir,image_vis_dir,label_dir,transform=None):
        self.image_ir_dir = image_ir_dir
        self.image_vis_dir  = image_vis_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_ir_dir)
        self.to_tensor = torchvision.transforms.ToTensor()
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self,index):
        image_ir_path = os.path.join(self.image_ir_dir,self.image_files[index])
        image_vis_path = os.path.join(self.image_vis_dir,self.image_files[index])
        label_path = os.path.join(self.label_dir,self.image_files[index].replace(".jpg",".txt"))

        image_ir = Image.open(image_ir_path).convert('L')
        image_vis = Image.open(image_vis_path).convert('RGB')

        resized_vis,scale,(orig_w,orig_h) = resize_aspect_ratio(image_vis)
        resized_ir,_,_ = resize_aspect_ratio(image_ir)

        padded_vis, (pad_x, pad_y) = pad_to_target_size(resized_vis)
        padded_ir, _ = pad_to_target_size(resized_ir)

        image_vis_tensor = self.to_tensor(padded_vis)
        image_ir_tensor = self.to_tensor(padded_ir)
        image = torch.cat([image_vis_tensor, image_ir_tensor], dim=0)  # [4, H, W]
        with open(label_path,'r') as f:
            labels = {"labels":[],"boxes":[]}
            for line in f.readlines():
                data = line.strip().split()
                #data[0]:class   data[1]:center_x   data[2]:center_y   data[3]:w   data[4]:h,均被归一化
                class_id = int(data[0])
                x_center, y_center, w, h = map(float, data[1:5])

                x_center_abs = x_center * orig_w
                y_center_abs = y_center * orig_h
                w_abs = w * orig_w
                h_abs = h * orig_h
                x_min = x_center_abs - w_abs / 2
                y_min = y_center_abs - h_abs / 2
                x_max = x_center_abs + w_abs / 2
                y_max = y_center_abs + h_abs / 2

                # Step 2: 应用缩放
                x_min *= scale
                y_min *= scale
                x_max *= scale
                y_max *= scale

                # Step 3: 应用填充偏移
                x_min += pad_x
                y_min += pad_y
                x_max += pad_x
                y_max += pad_y

                # Step 4: 重新归一化到目标尺寸
                x_min /= target_size[0]
                y_min /= target_size[1]
                x_max /= target_size[0]
                y_max /= target_size[1]

                # Step 5: 边界检查（确保坐标在[0, 1]范围内）
                x_min = max(0, min(1, x_min))
                y_min = max(0, min(1, y_min))
                x_max = max(0, min(1, x_max))
                y_max = max(0, min(1, y_max))

                # 确保x_max > x_min且y_max > y_min
                if x_max <= x_min or y_max <= y_min:
                    continue  # 跳过无效标注框

                labels["labels"].append(class_id)
                labels["boxes"].append([x_min,y_min,x_max,y_max])
        labels["labels"] = torch.tensor(labels["labels"])
        labels["boxes"] = torch.tensor(labels["boxes"])
        return image, labels

def collate_fn(batch):
    """
    处理不同数量的标注框（Dynamic padding）
    Args:
        batch: List[(image, labels), ...]
    Returns:
        batched_images: Tensor [B, C, H, W]
        batched_labels: List[Tensor [N_i, 5], ...]
    """
    batched_images = torch.stack([x[0] for x in batch], 0)  # [B, 4, H, W]
    batched_labels = [x[1] for x in batch]      # List[Tensor[N_i, 5]]
    return batched_images, batched_labels
if __name__ == "__main__":
    annotation_dir = '../train/label'
    annotation_path = os.listdir(annotation_dir)

    image_ir_dir = '../train/ir'
    image_ir_path = os.listdir(image_ir_dir)
    image_vis_dir = '../train/vis'
    image_vis_path = os.listdir(image_vis_dir)

    # targets = []
    # for i in annotation_path:
    #     data = []
    #     i = os.path.join(annotation_dir,i)
    #     with open(i,'r') as f:
    #         for line in f:
    #             #split()：按分隔符分割字符串
    #             #strip()：去除字符串两端的空白字符
    #             data1 = line.strip().split()
    #             data.append([int(data1[0]), float(data1[1]), float(data1[2]), float(data1[3]), float(data1[4])])
    #     targets.append(data)
    #     del data
    #     print(targets)
    #     break
    dataset = Loader(image_ir_dir,image_vis_dir,annotation_dir)
    dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn  # 处理变长标签
        )
    for images, labels in dataloader:
            print(f"Batch images shape: {images.shape}")  # [B, 4, H, W]
            print(labels[0]['boxes'])
            print(len(labels))
            break
    # images = []
    # resize_transform = torchvision.transforms.Resize((100,100))
    #
    # for i,j in zip(image_ir_path,image_vis_path):
    #     i = os.path.join(image_ir_dir,i)
    #     j = os.path.join(image_vis_dir,j)
    #     image_ir = Image.open(i)
    #     image_vis = Image.open(j)
    #     image_ir = to_tensor(resize_transform.forward(image_ir))
    #     image_vis = to_tensor(resize_transform.forward(image_vis))
    #     image = torch.cat([image_ir,image_vis],dim=0)
    #     images.append(image)
    #     del i,image,image_ir,image_vis
    # images = np.array(images)
    # print(images.shape)