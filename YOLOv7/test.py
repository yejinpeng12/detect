from YOLOtest import YOLO
import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from Image_Loader import resize_aspect_ratio,pad_to_target_size
import torchvision
from thop import profile
from copy import deepcopy
import numpy as np

def xyxy_to_xywh(bboxes):
    """
    输入: 一维数组 [x1, y1, x2, y2]
    输出: 一维数组 [x_center, y_center, width, height]
    """
    x_center = (bboxes[0] + bboxes[2]) / 2
    y_center = (bboxes[1] + bboxes[3]) / 2
    width = bboxes[2] - bboxes[0]
    height = bboxes[3] - bboxes[1]
    return np.array([x_center, y_center, width, height])
def calculate_flops(model, input_size=(640, 640), device='cuda'):
    """
    计算模型的FLOPs和参数量
    :param model: 待计算的模型
    :param input_size: 输入图像尺寸(H,W)
    :param device: 计算设备
    :return: (FLOPs(G), Params(M))
    """
    # 创建虚拟输入
    dummy_input = torch.randn(1, 4, *input_size).to(device)  # 4通道输入(3+1)

    # 计算FLOPs和参数量
    flops, params = profile(deepcopy(model),
                            inputs=(dummy_input,),
                            verbose=False)

    # 转换为GFLOPs和百万参数
    return flops / 1e9, params / 1e6  # 返回GFLOPs和百万参数
class ImageDataset(Dataset):
    def __init__(self,image_ir_dir,image_vis_dir):
        self.image_ir_dir = image_ir_dir
        self.image_vis_dir = image_vis_dir
        self.image_file = os.listdir(image_ir_dir)
        self.to_tensor = torchvision.transforms.ToTensor()
    def __len__(self):
        return len(self.image_file)
    def __getitem__(self,index):
        name = self.image_file[index]
        image_ir_path = os.path.join(self.image_ir_dir,name)
        image_vis_path = os.path.join(self.image_vis_dir,name)

        image_ir = Image.open(image_ir_path).convert('L')
        image_vis = Image.open(image_vis_path).convert('RGB')

        resized_vis,_,_ = resize_aspect_ratio(image_vis)
        resized_ir,_,_ = resize_aspect_ratio(image_ir)

        padded_vis,_ = pad_to_target_size(resized_vis)
        padded_ir,_ = pad_to_target_size(resized_ir)

        image_vis_tensor = self.to_tensor(padded_vis)
        image_ir_tensor = self.to_tensor(padded_ir)
        image = torch.cat([image_vis_tensor,image_ir_tensor],dim=0)
        return image, name

if __name__ == "__main__":
    model = YOLO(trainable=False, depthwise=True).cuda()
    model.load_state_dict(torch.load('model1'))

    n_p = sum(x.numel() for x in model.parameters())
    print(f"{n_p / (1024 ** 2):.2f}",end=' ')

    flops_g, params_m = calculate_flops(model, input_size=(640, 640))
    print(f"{flops_g:.2f}")
    model.eval()
    image_ir_dir = '../val/ir'
    image_vis_dir = '../val/vis'
    dataset = ImageDataset(image_ir_dir, image_vis_dir)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
    with open('result.txt','w') as f:
        f.write(f"{n_p} {flops_g}\n")
        for image,name in dataloader:
            image = image.to('cuda')
            with torch.no_grad():
                bboxes,scores,labels = model(image)
                f.write(f"{name[0]} ")
                print(name[0], end=' ')
                for i,j in zip(bboxes,labels):
                    array = xyxy_to_xywh(i)
                    f.write(f"{array[0]} {array[1]} {array[2]} {array[3]} {j} ")
                    print(array[0], array[1], array[2], array[3], j, end=' ')
                f.write("\n")
                print()
