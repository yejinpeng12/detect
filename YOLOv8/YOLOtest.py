from torch import nn
import numpy as np
import torch
from head import head
import math
from backbone import Base_bone
from neck import PaFPN
import torch.nn.functional as F
from dfl import DFL

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
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
                 confidence_thresh=0.25,
                 nms_thresh=0.2,
                 topk=100,
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
        self.reg_max = 16

        self.backbone = Base_bone()
        self.neck = PaFPN(self.backbone.feat_dim)
        self.head_dims = self.neck.out_dim
        #检测头
        self.head = nn.ModuleList([head(head_dim,head_dim,num_classes=self.num_classes)
                                  for head_dim in self.head_dims])#这里产生了三个网格尺寸的检测头
        self.dfl = DFL(self.reg_max)
    @torch.no_grad()
    def inference_single_image(self,x):
        x1 = self.backbone(x)

        x2 = self.neck(x1)


        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        for level,(feat,head) in enumerate(zip(x2,self.head)):
            cls_outputs, reg_outputs = head(feat)


            fmp_size = cls_outputs.shape[-2:]
            b,_,h,w = reg_outputs.shape
            anchors = self.generate_anchors(level, fmp_size)


            cls_pred = cls_outputs[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_outputs.permute(0, 2, 3, 1).contiguous().view(b ,-1 , 4 * self.reg_max)

            box_pred = self.bbox_decode(reg_pred).squeeze(0)
            box_pred = dist2bbox(box_pred, anchors, xywh=False) * self.stride[level]

            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        scores , labels, bboxes= self.post_process(
            all_cls_preds, all_box_preds)
        return bboxes, scores, labels
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            x1 = self.backbone(x)

            x2 = self.neck(x1)

            all_anchors = []
            all_strides = []
            all_cls_preds = []
            all_box_preds = []
            all_reg_preds = []
            for level, (feat, head) in enumerate(zip(x2, self.head)):
                cls_outputs, reg_outputs = head(feat)


                b, _, h, w = cls_outputs.size()
                fmp_size = [h, w]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]


                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_outputs.permute(0, 2, 3, 1).contiguous().view(b, -1, self.num_classes)
                reg_pred = reg_outputs.permute(0, 2, 3 ,1).contiguous().view(b, -1, 4 * self.reg_max)

                box_pred = self.bbox_decode(reg_pred)
                box_pred = dist2bbox(box_pred, anchors, xywh=False)

                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_reg_preds.append(reg_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)

            all_cls_preds = torch.cat(all_cls_preds,dim=1)
            all_box_preds = torch.cat(all_box_preds,dim=1)
            all_reg_preds = torch.cat(all_reg_preds,dim=1)
            all_anchors = torch.cat(all_anchors,dim=0)
            all_strides = torch.cat(all_strides,dim=0)
            # output dict
            outputs = {
                       "pred_cls": all_cls_preds,  #  [B, M, C]
                       "pred_box": all_box_preds,  #  [B, M, 4](x1,y1,x2,y2)
                        "pred_dist":all_reg_preds,  # [B, M, 4 * 16]
                       "anchors": all_anchors,  # [M, 2]
                        "strides":all_strides
                       }

            return outputs

    def generate_anchors(self, level, fmp_size):
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h,device=self.device),torch.arange(fmp_w,device=self.device)],indexing='ij')
        anchor_xy = torch.stack([anchor_x,anchor_y],dim=-1).float().view(-1,2)
        anchor_xy += 0.5
        anchors = anchor_xy.to(self.device)
        return anchors

    #非极大值抑制操作
    def nms(self,bounding_box,scores):
        """
        只保留得分最高的边界框，并移除与其重叠度较高的其他边框（针对单个类别）
        """
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
            current_box = bounding_box[i].reshape(1,4)
            other_boxes = bounding_box[order[1:]]
            ciou = bbox_iou(torch.tensor(current_box),torch.tensor(other_boxes),xywh=False,CIoU=True)
            ciou = ciou.cpu().numpy()
            #滤除超过NMS阈值的边界框
            #np.where返回一个元组，元组的第一个元素是符合条件的索引值列表
            inds = np.where(ciou <= self.nms_thresh)[0]
            order = order[inds + 1]#将符合条件的边界框取下来继续选，
            # 为什么加一，是因为iou是除了第一个边界框以外的数组，相比order只少了第一个边界框，加了一后便能把符合条件的索引值全部拿走，丢掉不符合条件的
        return keep
    #后处理
    def post_process(self,cls_preds,box_preds):
        all_scores = []
        all_labels = []
        all_bboxes = []

        for cls_pred_i, box_pred_i in zip(cls_preds,box_preds):
            scores_i = cls_pred_i.sigmoid().flatten()

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

    def bbox_decode(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        b, a, c = pred_dist.shape  # batch, anchors, channels
        proj = torch.arange(self.reg_max, dtype=torch.float, device=pred_dist.device)
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return pred_dist#(b,a,4)