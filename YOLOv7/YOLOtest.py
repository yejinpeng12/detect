from torch import nn
import numpy as np
import torch
from torch.amp import autocast,GradScaler
from Image_Loader import DataLoader,Loader,collate_fn
from basebone.ELANNet import ELANNet
from neck.SPP.SPPF import SPPF
from neck.PaFPN.PaFPN import YOLOv7PaFPN
from DecoupledHead.DecoupledHead import DecoupledHead
from loss import Criterion


class YOLO(nn.Module):
    def __init__(self,
                 device = "cuda",
                 num_classes=5,
                 confidence_thresh=0.01,
                 nms_thresh=0.5,
                 topk=100,
                 trainable=False,):
        super().__init__()
        self.stride = [8,16,32]
        self.device = device
        self.nms_thresh = nms_thresh
        self.confidence_thresh = confidence_thresh
        self.num_classes = num_classes
        self.topk = topk
        self.trainable = trainable
        #主干网络
        self.backbone = ELANNet()
        self.neck_dims = self.backbone.feat_dim[-3:]
        #颈部网络
        self.neck = SPPF(in_dim=self.neck_dims[-1],out_dim=self.neck_dims[-1]//2)
        self.neck_dims[-1] = self.neck.out_dim
        #颈部网络：特征金字塔
        self.fpn = YOLOv7PaFPN(self.neck_dims,None)
        self.head_dims = self.fpn.out_dim
        #检测头
        self.head = nn.ModuleList([DecoupledHead(head_dim,head_dim,num_classes=self.num_classes)
                                  for head_dim in self.head_dims])

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
                                for head in self.head
                              ])
    @torch.no_grad()
    def inference_single_image(self,x):
        x1 = self.backbone(x)

        x1[-1] = self.neck(x1[-1])

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

            # decode bbox
            ctr_pred = reg_pred[..., :2] * self.stride[level] + anchors[..., :2]
            wh_pred = torch.exp(reg_pred[..., 2:]) * self.stride[level]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        bboxes, scores, labels = self.post_process(
            all_obj_preds, all_cls_preds, all_box_preds)
        return bboxes, scores, labels



    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image
        else:
            x1 = self.backbone(x)

            x1[-1] = self.neck(x1[-1])

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

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_reg_preds.append(reg_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)

            # output dict
            outputs = {"pred_obj": all_obj_preds,  # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,  # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,  # List(Tensor) [B, M, 4]
                       "pred_reg": all_reg_preds,  # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,  # List(Tensor) [M, 2]
                       "strides": self.stride,  # List(Int) [8, 16, 32]
                       "stride_tensors": all_strides  # List(Tensor) [M, 1]
                       }

            return outputs

    def generate_anchors(self,level,fmp_size):
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h),torch.arange(fmp_w)],indexing="xy")
        anchor_xy = torch.stack([anchor_x,anchor_y],dim=-1).float().view(-1,2)
        anchor_xy += 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)
        return anchors

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
            h = np.minimum(1e-10,yy2-yy1)
            #计算交集的面积
            inter = w * h
            #计算交并比
            iou = inter/ (areas[i] +areas[order[1:]] - inter + 1e-14)
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

model = YOLO(trainable=True).cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
criterion = Criterion("cuda",num_classes=5)
#梯度缩放器
scaler = GradScaler()

annotation_dir = '../train/label'
image_ir_dir = '../train/ir'
image_vis_dir = '../train/vis'
dataset = Loader(image_ir_dir,image_vis_dir,annotation_dir)
dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn  # 处理变长标签
        )

for epoch in range(100):
    for images,targets in dataloader:
        images = images.to("cuda")
        optimizer.zero_grad()
        #启用混合精度上下文
        with autocast("cuda",enabled=True):
            preds = model(images)
            loss_dict = criterion(preds,targets,epoch)
        #缩放损失并反向传播
        scaler.scale(loss_dict['losses']).backward()
        #更新参数（自动取消缩放）
        scaler.step(optimizer)
        #调整缩放因子
        scaler.update()

        print(f"Epoch: {epoch}, Loss: {loss_dict['losses']}")
    torch.save(model.state_dict(),"model")
#model.load_state_dict(torch.load('model'))加载参数