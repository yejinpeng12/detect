from torch import nn
import numpy as np
import torch
from torch.amp import autocast,GradScaler
from Image_Loader import DataLoader,Loader,collate_fn
from basebone.ELANNet import ELANNet
from neck.SPP.SPPBlockCSP import SPPFBlockCSP
from neck.PaFPN.PaFPN import YOLOv7PaFPN
from DecoupledHead.DecoupledHead import DecoupledHead
from loss import Criterion
from basebone.ELANNet_Tiny import ELANNet_Tiny
from tqdm import tqdm
import sys
import math
from YOLOv8.basebone import Base_bone
from YOLOv8.neck import PaFPN
from yolov12.backbone import BackBone
from  yolov12.neck import neck
import os
from YOLOv7.basic_block_v7.simAM_module import simam_module

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class YOLO(nn.Module):
    def __init__(self,
                 device = "cuda",
                 num_classes=5,
                 confidence_thresh=0.6,
                 nms_thresh=0.2,
                 topk=1000,
                 trainable=False,
                 depthwise = False):
        super().__init__()
        self.stride = [8,16,32]#网格尺寸分别为(80*80,40*40,20*20)
        self.device = device
        self.nms_thresh = nms_thresh
        self.confidence_thresh = confidence_thresh
        self.num_classes = num_classes
        self.topk = topk
        self.trainable = trainable
        # #主干网络
        self.backbone = ELANNet_Tiny(depthwise=False)
        self.neck_dims = self.backbone.feat_dims[-3:]
        #颈部网络
        self.neck = SPPFBlockCSP(in_dim=self.neck_dims[-1],out_dim=self.neck_dims[-1]//2,depthwise=False)
        self.neck_dims[-1] = self.neck_dims[-1]//2
        #颈部网络：特征金字塔
        self.fpn = YOLOv7PaFPN(self.neck_dims,None,depthwise=depthwise)
        self.head_dims = self.fpn.out_dim

        self.simam = simam_module()

        # self.backbone = Base_bone()
        # self.neck = PaFPN(self.backbone.feat_dim)
        # self.head_dims = self.neck.out_dim

        # self.backbone = BackBone()
        # self.neck = neck(self.backbone.out_dim)
        #
        # self.head_dims = self.neck.out_dim
        #检测头
        self.head = nn.ModuleList([DecoupledHead(head_dim,head_dim,num_classes=self.num_classes,depthwise=depthwise)
                                  for head_dim in self.head_dims])#这里产生了三个网格尺寸的检测头

        #预测层
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 1, kernel_size=1)
                                for head in self.head
                              ])
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes, kernel_size=1)
                                for head in self.head
                              ])
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4, kernel_size=1)
                                for head in self.head])

    # def decode_box(self, reg_pred, anchors, fmp_size):
    #     # 归一化解码
    #     grid_scale = 1.0 / torch.tensor(fmp_size, device=reg_pred.device)
    #     ctr_pred = reg_pred[..., :2].sigmoid() * grid_scale + anchors[..., :2]
    #     wh_pred = torch.exp(reg_pred[..., 2:].clamp(max=5)) * grid_scale
    #     return torch.cat([
    #         (ctr_pred - wh_pred / 2).clamp(0, 1),
    #         (ctr_pred + wh_pred / 2).clamp(0, 1)
    #     ], dim=-1)
    @torch.no_grad()
    def inference_single_image(self,x):
        x1 = self.backbone(x)

        #x2 = self.neck(x1)
        x1[-1] = self.simam(self.neck(x1[-1]))

        x2 = self.fpn(x1)

        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        for level,(feat,head) in enumerate(zip(x2,self.head)):
            cls_feat, reg_feat = head(feat)

            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level,fmp_size)

            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            #decode bbox
            ctr_pred = reg_pred[..., :2] * self.stride[level] + anchors[..., :2]
            wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride[level]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
            #box_pred = self.decode_box(reg_pred,anchors,fmp_size)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        scores ,labels,bboxes= self.post_process(
            all_obj_preds, all_cls_preds, all_box_preds)
        return bboxes, scores, labels
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            x1 = self.backbone(x)

            #x2 = self.neck(x1)
            x1[-1] = self.simam(self.neck(x1[-1]))

            x2 = self.fpn(x1)

            all_anchors = []
            all_strides = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            all_reg_preds = []
            for level, (feat, head) in enumerate(zip(x2, self.head)):
                cls_feat, reg_feat = head(feat)

                # [B, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                b, _, h, w = cls_pred.size()
                fmp_size = [h, w]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
                # decode bbox
                ctr_pred = reg_pred[..., :2] * self.stride[level] + anchors[..., :2]
                wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride[level]
                pred_x1y1 = ctr_pred - wh_pred * 0.5
                pred_x2y2 = ctr_pred + wh_pred * 0.5
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
                #box_pred = self.decode_box(reg_pred,anchors,fmp_size)
                # print("box_pred min:", reg_pred.min().item())
                # print("box_pred max:", reg_pred.max().item())

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_reg_preds.append(reg_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)

            # output dict
            outputs = {"pred_obj": all_obj_preds,  # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,  # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,  # List(Tensor) [B, M, 4](x1,y1,x2,y2)
                       "pred_reg": all_reg_preds,  # List(Tensor) [B, M, 4](tx,ty,w,h)
                       "anchors": all_anchors,  # List(Tensor) [M, 2]
                       "strides": self.stride,  # List(Int) [8, 16, 32]
                       "stride_tensors": all_strides  # List(Tensor) [M, 1]数值上是8或16或32
                       }

            return outputs

    def generate_anchors(self,level,fmp_size):
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h,device=self.device),torch.arange(fmp_w,device=self.device)],indexing='ij')
        anchor_xy = torch.stack([anchor_x,anchor_y],dim=-1).float().view(-1,2)
        anchor_xy += 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)
        return anchors
    # def  generate_anchors(self, level, fmp_size):
    #     fmp_h, fmp_w = fmp_size
    #     # 生成归一化网格坐标 [0,1]
    #     y, x = torch.meshgrid(
    #         torch.linspace(0, 1, fmp_h, device=self.device),
    #         torch.linspace(0, 1, fmp_w, device=self.device),
    #         indexing='ij'
    #     )
    #     anchor_xy = torch.stack([x, y], dim=-1).view(-1, 2)
    #     # 中心对齐补偿
    #     anchor_xy += 0.5 / min(fmp_h, fmp_w)
    #     return anchor_xy

    #非极大值抑制操作
    def nms(self,bounding_box,scores):
        """
        只保留得分最高的边界框，并移除与其重叠度较高的其他边框（针对单个类别）
        :param bounding_box:[13*13,4]
        :param scores:[13*13,1]
        :return:
        """
        x1 = bounding_box[:,0]
        y1 = bounding_box[:,1]
        x2 = bounding_box[:,2]
        y2 = bounding_box[:,3]
        areas = (x2 - x1) * (y2 - y1)
        #将得分进行排序，得分最大的索引值放在最前面,[::-1]表示反转数列
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            #这里要注意order储存的是索引值
            #[start:stop:step],这是索引语法
            #start表示切片的起始索引(包含)
            #stop表示切片的结束索引(不包含)
            #step表示步长，即每次跳过的元素数量
            #[::-1]表示反转序列顺序
            i = order[0]
            keep.append(i)
            #计算交集的左上角点和右下角点的坐标
            #maximum是比较取值，去最大的作为返回值，如果是数与数组比较，则返回数组，数组中比数小的被数取代
            xx1 = np.maximum(x1[i],x1[order[1:]])
            yy1 = np.maximum(y1[i],y1[order[1:]])
            xx2 = np.minimum(x2[i],x2[order[1:]])
            yy2 = np.minimum(y2[i],y2[order[1:]])
            #计算交集的宽和高
            w = np.maximum(1e-10,xx2-xx1)
            h = np.maximum(1e-10,yy2-yy1)
            #计算交集的面积
            inter = w * h
            #计算交并比
            iou = inter/ (areas[i] +areas[order[1:]] - inter)
            #滤除超过NMS阈值的边界框
            #np.where返回一个元组，元组的第一个元素是符合条件的索引值列表
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]#将符合条件的边界框取下来继续选，
            # 为什么加一，是因为iou是除了第一个边界框以外的数组，相比order只少了第一个边界框，加了一后便能把符合条件的索引值全部拿走，丢掉不符合条件的
        return keep
    #后处理
    def post_process(self,obj_preds,cls_preds,box_preds):
        all_scores = []
        all_labels = []
        all_bboxes = []

        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds,cls_preds,box_preds):
            scores_i = (torch.sqrt(obj_pred_i.sigmoid()*cls_pred_i.sigmoid())).flatten()

            num_topk = min(self.topk,64)

            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = topk_scores > self.confidence_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs,self.num_classes,rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)


        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        keep = np.zeros(len(bboxes),dtype=np.int32)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes,c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores,labels,bboxes

#探查gpu内存使用情况
def print_memory_usage():
    print("GPU memory usage:", torch.cuda.memory_allocated() / (1024 * 1024 * 1024), "GB")
    print("GPU memory reserved:", torch.cuda.memory_reserved() / (1024 * 1024 * 1024), "GB")

if __name__ == '__main__':
    model = YOLO(trainable=True,depthwise=True).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)#1e-7
    criterion = Criterion("cuda", num_classes=5)
    model.load_state_dict(torch.load('config/simam78'))
    n_p = sum(x.numel() for x in model.parameters())
    print(n_p/(1024 ** 2))
    #梯度缩放器
    scaler = GradScaler(init_scale=2.0**16,#降低初始缩放因子
                        growth_factor=2.0,#降低增长因子
                        backoff_factor=0.5#提高回退因子
                        )

    annotation_dir = '../train/label'
    image_ir_dir = '../train/ir'
    image_vis_dir = '../train/vis'
    dataset = Loader(image_ir_dir,image_vis_dir,annotation_dir)
    dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                num_workers=4,#根据CPU核心数调整
                pin_memory=True,
                persistent_workers=True,#保持worker进程
                prefetch_factor=2,#预取2个批次
                collate_fn=collate_fn
            )
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(79 ,100):
        total_loss = 0
        for images,targets in tqdm(dataloader,file=sys.stdout,position=0,colour="green",desc=f"Epoch: {epoch}/99"):
            images = images.to("cuda")
            optimizer.zero_grad()
            #启用混合精度上下文
            with autocast("cuda",enabled=True,dtype=torch.float16):
                preds = model(images)
                loss_dict = criterion(preds,targets,epoch)
                losses = loss_dict['losses']
                total_loss += losses
                #print_memory_usage()
            #缩放损失并反向传播
            scaler.scale(losses).backward()
            #梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #更新参数（自动取消缩放）
            scaler.step(optimizer)
            #调整缩放因子
            scaler.update()
            #torch.cuda.empty_cache()
            if epoch >= 49:
                tqdm.write(f"Loss: {loss_dict['losses']},Loss_obj:{loss_dict['loss_obj']},Loss_cls:{loss_dict['loss_cls']},Loss_box:{loss_dict['loss_box']},loss_box_aux:{loss_dict['loss_box_aux']}")
            else:
                tqdm.write(f"Loss: {loss_dict['losses']},Loss_obj:{loss_dict['loss_obj']},Loss_cls:{loss_dict['loss_cls']},Loss_box:{loss_dict['loss_box']}")
        with open('loss.txt', 'a') as f:
            f.write(f"{epoch}_total_mean_loss{total_loss / 6250}\n")
        torch.save(model.state_dict(),os.path.join("config",f"simam{epoch}"))