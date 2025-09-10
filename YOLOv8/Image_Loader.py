import os
import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl

# 指定支持中文的字体（按优先级尝试）
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
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
    return padded_image,(padding_left,padding_top,padding_right,padding_bottom)
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

        padded_vis, (padding_left, padding_top,_,_) = pad_to_target_size(resized_vis)
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

                x_min = x_center_abs - (w_abs / 2)
                y_min = y_center_abs - (h_abs / 2)
                x_max = x_center_abs + (w_abs / 2)
                y_max = y_center_abs + (h_abs / 2)

                # Step 2: 应用缩放
                x_min *= scale
                y_min *= scale
                x_max *= scale
                y_max *= scale


                # Step 3: 应用填充偏移
                x_min += padding_left
                y_min += padding_top
                x_max += padding_left
                y_max += padding_top

                # Step 4: 重新归一化到目标尺寸
                # x_min /= target_size[0]
                # y_min /= target_size[1]
                # x_max /= target_size[0]
                # y_max /= target_size[1]

                labels["labels"].append(class_id)
                labels["boxes"].append([x_min,y_min,x_max,y_max])
        labels["labels"] = torch.tensor(labels["labels"])
        labels["boxes"] = torch.tensor(labels["boxes"])
        return image, labels

def collate_fn(batch):
    batched_images = torch.stack([x[0] for x in batch], 0)  # [B, 4, H, W]
    labels_list = [x[1] for x in batch]      # List[Tensor[N_i, 5]]

    batch_size = len(batch)
    max_num_bboxes = max(len(labels['labels']) for labels in labels_list)
    padded_labels = torch.full((batch_size, max_num_bboxes), -1, dtype=torch.int64)  # 用-1填充无效位置
    padded_boxes = torch.zeros((batch_size, max_num_bboxes, 4), dtype=torch.float32)
    mask_gt = torch.zeros((batch_size, max_num_bboxes, 1), dtype=torch.bool)
    for i, labels in enumerate(labels_list):
        num_gt = len(labels['labels'])
        if num_gt > 0:
            padded_labels[i, :num_gt] = labels['labels']
            padded_boxes[i, :num_gt] = labels['boxes']
            mask_gt[i, :num_gt] = True
    batched_labels = {
        'labels': padded_labels,
        'boxes': padded_boxes
    }
    return batched_images, batched_labels, mask_gt

if __name__ == "__main__":
    annotation_dir = '../train/label'
    annotation_path = os.listdir(annotation_dir)
    image_ir_dir = '../train/ir'
    image_ir_path = os.listdir(image_ir_dir)
    image_vis_dir = '../train/vis'
    image_vis_path = os.listdir(image_vis_dir)

    dataset = Loader(image_ir_dir,image_vis_dir,annotation_dir)

def validate_annotations(dataset, num_samples=5, show_ir_channel=False):
    """
    验证标签标注是否正确
    Args:
        dataset: 加载好的数据集对象
        num_samples: 要验证的样本数量
        show_ir_channel: 是否显示红外通道
    """
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        image, labels = dataset[idx]

        # 准备图像显示
        fig, ax = plt.subplots(1, figsize=(12, 12))

        # 处理图像显示
        if show_ir_channel:
            # 合并可见光和红外通道（红外显示为红色增强）
            img_display = image[:3].clone()
            img_display[0] = torch.clamp(img_display[0] + image[3], 0, 1)  # 将红外通道加到红色通道
            img_np = img_display.permute(1, 2, 0).numpy()
        else:
            # 仅显示RGB通道
            img_np = image[:3].permute(1, 2, 0).numpy()

        img_np = (img_np * 255).astype('uint8')
        ax.imshow(img_np)

        # 绘制边界框和标签
        boxes = labels['boxes'].numpy()
        class_ids = labels['labels'].numpy()

        for box, cls_id in zip(boxes, class_ids):
            # 反归一化坐标
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 验证类别ID是否有效
            if cls_id >= len(test_ClASS):
                print(f"警告：无效类别ID {cls_id} (最大应为 {len(test_ClASS) - 1})")
                continue

            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            # 添加类别标签
            label = test_ClASS[cls_id]
            ax.text(
                x1, y1 - 5, f"{label} (ID:{cls_id})",
                color='lime', fontsize=12, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )

        # 添加标题信息
        plt.title(
            f"验证样本 {idx}\n"
            f"文件名: {dataset.image_files[idx]}\n"
            f"目标数: {len(class_ids)} | "
            f"图像尺寸: {image.shape[1]}x{image.shape[2]} | "
            f"有效类别: {np.unique(class_ids)}",
            fontsize=12
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 打印详细坐标信息
        print(f"\n样本 {idx} 详细标注信息:")
        for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
            print(f"  目标 {i + 1}: {test_ClASS[cls_id]} (ID:{cls_id})")
            print(f"    像素坐标: x1={int(box[0])}, y1={int(box[1])}, "
                  f"x2={int(box[2])}, y2={int(box[3])}")
            print(
                f"    框尺寸: {int((box[2] - box[0]))}x{int((box[3] - box[1]))} 像素")


# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset = Loader(
        image_ir_dir='../train/ir',
        image_vis_dir='../train/vis',
        label_dir='../train/label'
    )
    # 验证5个随机样本（显示红外通道）
    validate_annotations(dataset, num_samples=5, show_ir_channel=True)
