from YOLOtest import YOLO
from Image_Loader import DataLoader,Loader,collate_fn
from tqdm import tqdm
import sys
import torch
from torch.amp import autocast,GradScaler
from loss import YOLOv8Loss

if __name__ == '__main__':
    model = YOLO(trainable=True,depthwise=True).cuda()
    loss_fn = YOLOv8Loss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    #model.load_state_dict(torch.load('modelv8_0'))
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
    for epoch in range(0,100):
        for images, targets, mask_gt in tqdm(dataloader,file=sys.stdout,position=0,colour="green",desc=f"Epoch: {epoch}/99"):
            images = images.to("cuda")
            mask_gt = mask_gt.to("cuda")
            optimizer.zero_grad()
            #启用混合精度上下文
            with autocast("cuda",enabled=True,dtype=torch.float16):
                preds = model(images)
                loss_dict = loss_fn(preds, targets, mask_gt)
                losses = loss_dict['total_loss']
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
            tqdm.write(f"Loss: {loss_dict['total_loss']},Loss_cls:{loss_dict['loss_cls']},Loss_box:{loss_dict['loss_box']},loss_dfl:{loss_dict['loss_dfl']},fg_ratio:{loss_dict['fg_ratio']}")
        torch.save(model.state_dict(),f"modelv8_{epoch}")

