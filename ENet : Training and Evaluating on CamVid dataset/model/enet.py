import torch
import torch.nn as nn
from .components import Initial_Block,Bottleneck,Downsampling,Upsampling

class ENet(nn.Module):
    def __init__(self,num_classes):
        """
        num_classes : the number of class needs to be segmented
        As original paper shows , there are 5 stages after initial block
        dropout_prob : p=0.01 before bottleneck2.0 and p=0.1 afterwards
        """
        super(ENet,self).__init__()

        #Initial Block
        self.initial_block=Initial_Block(in_channels=3,out_channels=16)

        #Stage 1 : 1 downsample + 4 regular bottleneck
        self.bottleneck_1_0=Downsampling(in_channels=16,out_channels=64,dropout_prob=0.01)
        self.bottleneck_1_x=nn.Sequential(
            Bottleneck(in_channels=64,dropout_prob=0.01),
            Bottleneck(in_channels=64,dropout_prob=0.01),
            Bottleneck(in_channels=64,dropout_prob=0.01),
            Bottleneck(in_channels=64,dropout_prob=0.01)
        )

        #Stage 2 : 1 downsampling + 2 reg + 4 dil + 2 asymmetric 
        #config format : type_bottleneck,in_channels (= out_channels),kernel_size,padding,dilation
        #Downsampling is defined independently to get max_idxs later
        #type_bottleneck :  "R" ,"A" ,"R" is regular or dilated (based on padding and dilation params) , "A" is asymmetric
        self.bottleneck_2_0=Downsampling(in_channels=64,out_channels=128,dropout_prob=0.1)
        self.stage_2_configs=[
            ["R",128,3,1,1],
            ["R",128,3,2,2],#dilated 2
            ["A",128,5,2,1], #asym 5
            ["R",128,3,4,4],#dilated 4
            ["R",128,3,1,1],
            ["R",128,3,8,8],#dilated 8
            ["A",128,5,2,1], #asym 5
            ["R",128,3,16,16] #dilated 16
        ]
        self.bottleneck_2_1to8=self.create_encoder(self.stage_2_configs)
        

        #Stage 3 is similar to stage 2 without bottleneck2.0
        self.bottleneck_3_1to8=self.create_encoder(self.stage_2_configs)


        #Stage 4 and 5 (DECODER)
        self.bottleneck_4_0=Upsampling(in_channels=128,out_channels=64,dropout_prob=0.1)
        self.bottleneck_4_1=Bottleneck(in_channels=64,dropout_prob=0.1)
        self.bottleneck_4_2=Bottleneck(in_channels=64,dropout_prob=0.1)
        self.bottleneck_5_0=Upsampling(in_channels=64,out_channels=16,dropout_prob=0.1)
        self.bottleneck_5_1=Bottleneck(in_channels=16,dropout_prob=0.1)
        self.fullconv=nn.ConvTranspose2d(in_channels=16,out_channels=num_classes,kernel_size=2,stride=2,bias=False)

    def forward(self,x):
        x=self.initial_block(x)

        #Encoder
        #Stage 1
        x,max_idxs_1=self.bottleneck_1_0(x)
        x=self.bottleneck_1_x(x)

        #Stage 2
        x,max_idxs_2=self.bottleneck_2_0(x)
        x=self.bottleneck_2_1to8(x)

        #Stage 3
        x=self.bottleneck_3_1to8(x)

        #Decoder
        #Stage 4
        x=self.bottleneck_4_0(x,max_idxs_2)
        x=self.bottleneck_4_1(x)
        x=self.bottleneck_4_2(x)

        #Stage 5
        x=self.bottleneck_5_0(x,max_idxs_1)
        x=self.bottleneck_5_1(x)
        
        #fullconv
        x=self.fullconv(x)
        return x
    
    def create_encoder(self,configs):
        layers=nn.ModuleList()
        for config in configs:  
            layers+=[Bottleneck(in_channels=config[1],kernel_size=config[2],padding=config[3],dilation=config[4],isAsymmetric=(config[0]=='A'),dropout_prob=0.1)]
        return nn.Sequential(*layers)
    

def test():
    x=torch.randn(1,3,512,512)
    model=ENet(num_classes=15)
    print(model(x).shape)

if __name__=='__main__':
    test()
        