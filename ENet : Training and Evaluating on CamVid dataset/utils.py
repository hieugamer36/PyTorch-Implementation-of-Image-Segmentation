import numpy as np
import os
from PIL import Image


def weighing_class(dataloader,num_classes=12,c=1.02):
    """
    calculate weights as ENet paper shown:
    w_class=1/(ln(c+p_class))
    where p_class=num_each_class/total
    """
    num_each_class=0
    total=0
    for _,label in dataloader: #label shape : batchsize,1,360,480
        flat_label=label.numpy().flatten()
        num_each_class += np.bincount(flat_label)
        total+=flat_label.size
    
    p_class=num_each_class/total
    return 1/(np.log(c+p_class))
