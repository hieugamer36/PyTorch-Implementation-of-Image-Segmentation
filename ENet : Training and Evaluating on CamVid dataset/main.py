import torch
from argparse import ArgumentParser
from data.camvid import dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from model.enet import ENet
from utils import weighing_class
from train import train_batch
from eval import eval
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch

#Color for visualization
color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
]) 


def arguments():
    parser=ArgumentParser()
    parser.add_argument("--train_img_root")
    parser.add_argument("--train_label_root")
    parser.add_argument("--val_img_root")
    parser.add_argument("--val_label_root")
    parser.add_argument("--test_img_root")
    parser.add_argument("--test_label_root")
    parser.add_argument("--lr",type=float,default=5e-4) 
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=300)
    parser.add_argument("--weight_decay",type=float,default=2e-4)
    parser.add_argument("--lr_decay_epochs",type=int,default=100)
    parser.add_argument("--lr_decay",type=float,default=0.1)
    parser.add_argument("--num_workers",type=int,default=4)
    return parser.parse_args()

if __name__=='__main__':
    
    """Args"""
    # train_img_root="./CamVid/train/"
    # train_label_root="./CamVid/trainannot/"
    # val_img_root="./CamVid/val/"
    # val_label_root="./CamVid/valannot/"
    # test_img_root="./CamVid/test/"
    # test_label_root="./CamVid/testannot/"
    # batch_size=5
    # num_workers=2
    # lr=5e-4
    # weight_decay=2e-4
    # lr_decay_epochs=100
    # lr_decay=0.1
    # epochs=10
    
    args=arguments()
    train_img_root=args.train_img_root
    train_label_root=args.train_label_root
    val_img_root=args.val_img_root
    val_label_root=args.val_label_root
    test_img_root=args.test_img_root
    test_label_root=args.test_label_root
    batch_size=args.batch_size
    num_workers=args.num_workers
    lr=args.lr
    epochs=args.epochs
    lr_decay_epochs=args.lr_decay_epochs
    weight_decay=args.weight_decay
    lr_decay=args.lr_decay
    print("lr : ",lr)
    print("Epochs : ",epochs)
    print("lr_decay_epochs : ",lr_decay_epochs)
    print("lr_decay : ",lr_decay)
    print("weight_decay : ",weight_decay)
    print("batchsize : ",batch_size)
    
    """Get device"""
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    """Get dataset"""
    train_dataset=dataset(train_img_root,train_label_root)
    val_dataset=dataset(val_img_root,val_label_root)
    test_dataset=dataset(test_img_root,test_label_root)
    print("Number training : ",len(train_dataset))
    print("Number validation :",len(val_dataset))
    print("Number testing : ",len(test_dataset))
    
    """Get dataloader"""
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers)

    """Get Class Weighting"""
    class_weights=torch.from_numpy(weighing_class(train_loader,num_classes=12)).float().to(device)
    unlabeled_idx=list(color_encoding).index('unlabeled')
    class_weights[unlabeled_idx]=0
    print("class_weights : ",class_weights)

    """Get model,loss,optimizer,lr_scheduler"""
    model=ENet(num_classes=12).to(device)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    lr_scheduler=lr_scheduler.StepLR(optimizer,lr_decay_epochs,lr_decay)
    
    """Training"""
    best_val_iou=0
    for epoch in range(epochs):
        #Train
        train_loss=train_batch(train_loader,model,loss_fn,optimizer,device=device)
        lr_scheduler.step()
        # Eval on val set
        val_loss,val_iou=eval(val_loader,model,loss_fn,device=device)
        print("Epoch : {}/{} , val_loss ={:.2f}, val_iou={:.2f}".format(epoch+1,epochs,val_loss,val_iou))
        if best_val_iou < val_iou:
            best_val_iou=val_iou
            checkpoints={
                "state_dict":model.state_dict(),
                "epoch":epoch,
                "lr":lr,
                "val_loss":val_loss,
                "val_iou":val_iou
            }
            torch.save(checkpoints,"my_checkpoints.pth.tar")

