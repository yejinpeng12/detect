import torch
import torch.nn.functional as F
from YOLOv7.simOTA import SimOTA

def get_ious(bboxes1,
             bboxes2,
             box_mode="xyxy",
             iou_type="iou"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        bboxes1 = torch.cat((-bboxes1[..., :2], bboxes1[..., 2:]), dim=-1)
        bboxes2 = torch.cat((-bboxes2[..., :2], bboxes2[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]).clamp_(min=0) \
        * (bboxes1[..., 3] - bboxes1[..., 1]).clamp_(min=0)
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]).clamp_(min=0) \
        * (bboxes2[..., 3] - bboxes2[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(bboxes1[..., 2], bboxes2[..., 2])
                   - torch.max(bboxes1[..., 0], bboxes2[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(bboxes1[..., 3], bboxes2[..., 3])
                   - torch.max(bboxes1[..., 1], bboxes2[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = bboxes2_area + bboxes1_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = torch.max(bboxes1[..., 2], bboxes2[..., 2]) \
            - torch.min(bboxes1[..., 0], bboxes2[..., 0])
        g_h_intersect = torch.max(bboxes1[..., 3], bboxes2[..., 3]) \
            - torch.min(bboxes1[..., 1], bboxes2[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        return gious
    else:
        raise NotImplementedError

class Criterion(object):
    def __init__(self,
                 device,
                 num_classes=80):
        self.device = device
        self.num_classes = num_classes
        self.max_epoch = 20
        self.no_aug_epoch = 10
        self.aux_bbox_loss = False
        # loss weight
        self.loss_obj_weight = 1.0
        self.loss_cls_weight = 1.0
        self.loss_box_weight = 5.0
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=2.5,
            topk_candidate=10
        )

    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj

    def loss_classes(self, pred_cls, gt_label):
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        ious = get_ious(pred_box, gt_box, "xyxy", 'giou')
        loss_box = 1.0 - ious

        return loss_box

    def loss_bboxes_aux(self, pred_reg, gt_box, anchors, stride_tensors):
        # 在训练的第二和第三阶段，增加bbox的辅助损失，直接回归预测的delta和label的delta之间的损失

        # 计算gt的中心点坐标和宽高
        gt_cxcy = (gt_box[..., :2] + gt_box[..., 2:]) * 0.5
        gt_bwbh = gt_box[..., 2:] - gt_box[..., :2]

        # 计算gt的中心点delta和宽高的delta，本质就是边界框回归公式的逆推
        gt_cxcy_encode = (gt_cxcy - anchors) / stride_tensors
        gt_bwbh_encode = torch.log(gt_bwbh / stride_tensors)
        gt_box_encode = torch.cat([gt_cxcy_encode, gt_bwbh_encode], dim=-1)

        # 计算预测的delta和gt的delta指甲的L1损失
        loss_box_aux = F.l1_loss(pred_reg, gt_box_encode, reduction='none')

        return loss_box_aux

    def __call__(self, outputs, targets, epoch=0):
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_reg']: List(Tensor) [B, M, 4]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...],
                                 'labels': [...],
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']#[M,2]
        # preds: [B, M, C]
        #将三个分辨率的锚点融合，M=m1+m2+m3
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # ------------------ 标签分配 ------------------
        cls_targets = []
        box_targets = []
        obj_targets = []
        fg_masks = []
        for batch_idx in range(bs):
            #取出一张图片的labels和boxes
            tgt_labels = targets[batch_idx]["labels"].to(device)#[N]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)#[N,4]

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # There is no valid gt
                cls_target = obj_preds.new_zeros((0, self.num_classes))
                box_target = obj_preds.new_zeros((0, 4))
                obj_target = obj_preds.new_zeros((num_anchors, 1))
                fg_mask = obj_preds.new_zeros(num_anchors).bool()
            else:
                (
                    fg_mask,
                    assigned_labels,
                    assigned_ious,
                    assigned_indexs
                ) = self.matcher(
                    fpn_strides=fpn_strides,
                    anchors=anchors,
                    pred_obj=obj_preds[batch_idx],
                    pred_cls=cls_preds[batch_idx],
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes
                )
                #取出对应的正确答案
                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(assigned_labels.long(), self.num_classes)
                cls_target = cls_target * assigned_ious.unsqueeze(-1)
                box_target = tgt_bboxes[assigned_indexs]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)

        # 将标签的shape处理成和预测的shape相同的形式，以便后续计算损失
        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fgs = fg_masks.sum()

        # ------------------ Objecntness loss ------------------
        loss_obj = self.loss_objectness(obj_preds.view(-1, 1), obj_targets.float())
        loss_obj = loss_obj.sum() / num_fgs

        # ------------------ Classification loss ------------------
        cls_preds_pos = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.loss_classes(cls_preds_pos, cls_targets)
        loss_cls = loss_cls.sum() / num_fgs

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        # ------------------ Aux regression loss ------------------
        loss_box_aux = None
        if epoch >= (self.max_epoch - self.no_aug_epoch - 1):
            ## reg_preds
            reg_preds = torch.cat(outputs['pred_reg'], dim=1)
            reg_preds_pos = reg_preds.view(-1, 4)[fg_masks]
            ## anchor tensors
            anchors_tensors = torch.cat(outputs['anchors'], dim=0)[None].repeat(bs, 1, 1)
            anchors_tensors_pos = anchors_tensors.view(-1, 2)[fg_masks]
            ## stride tensors
            stride_tensors = torch.cat(outputs['stride_tensors'], dim=0)[None].repeat(bs, 1, 1)
            stride_tensors_pos = stride_tensors.view(-1, 1)[fg_masks]
            ## aux loss
            loss_box_aux = self.loss_bboxes_aux(reg_preds_pos, box_targets, anchors_tensors_pos, stride_tensors_pos)
            loss_box_aux = loss_box_aux.sum() / num_fgs

            losses += loss_box_aux

        # Loss dict
        if loss_box_aux is None:
            loss_dict = dict(
                loss_obj=loss_obj,
                loss_cls=loss_cls,
                loss_box=loss_box,
                losses=losses
            )
        else:
            loss_dict = dict(
                loss_obj=loss_obj,
                loss_cls=loss_cls,
                loss_box=loss_box,
                loss_box_aux=loss_box_aux,
                losses=losses
            )

        return loss_dict