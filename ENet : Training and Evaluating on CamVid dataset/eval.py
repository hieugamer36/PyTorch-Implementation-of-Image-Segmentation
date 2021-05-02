import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def get_cfs_matrix(pred,target):
    """
    target : torch tensor with shape : (batchsize,img_height,img_width)
    pred : torch tensor with shape : (batchsize,num_classes,img_height,img_width)
    """
    pred=pred.cpu()
    target=target.cpu()
    #Flatten
    target = target.view(-1)
    pred=pred.max(1)[1].view(-1)
    cfs=confusion_matrix(target,pred)
    cfs[:,11]=0
    cfs[11,:]=0
    return confusion_matrix(target,pred)

def calculate_mean_iou(cfs_matrix):
    """
    cfs_matrix : confusion matrix : np array with shape (num_classes,num_classes)
    iou in semantic segmentation is calculated : iou=true_positive/(true_positive+false_positive+false_negative)
    true_positive = elements in diagonal of cfs matrix
    false_positive in each row = sum of each row - true_positive in each row
    false_negative in each col = sum of each col - true_positive in each col
    """
    true_positive=np.diag(cfs_matrix)
    false_positive=np.sum(cfs_matrix,1)-true_positive
    false_negative=np.sum(cfs_matrix,0)-true_positive
    iou=true_positive/(false_negative+false_negative+true_positive+1e-5)
    return np.mean(iou) #Get average of ious for entire classes

def eval(dataloader,model,loss_fn,device="cpu",num_classes=12):
    """
    model has been sent before this function is called
    """
    model.eval()
    test_losses=[]
    cfs_matrix=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for data,target in dataloader:
            data,target=data.to(device),target.to(device)
            pred=model(data) #batchsize , num_classes , img_height, img_width
            loss=loss_fn(pred,target)
            test_losses.append(loss.item())
            cfs_matrix += get_cfs_matrix(pred.detach(),target.detach())
    
    mean_iou = calculate_mean_iou(cfs_matrix)
    model.train()
    return sum(test_losses)/len(test_losses), mean_iou

    