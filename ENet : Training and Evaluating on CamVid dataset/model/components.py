import torch
import torch.nn as nn


class Initial_Block(nn.Module):
    """
    Initial Block will contain 2 elements:
    1. A conv2d with kernel_size=3 ,stride=2 (main branch)
    2. An extension branch which only has a maxpooling2d
    Because the maxpooling layer follows after the input layer , the out_channels of this layer will be 3
    To make sure that after concatenating the number of channel is 16 ,the output channel number of conv2d will be 13
    """
    def __init__(self,in_channels=3,out_channels=16):
        super(Initial_Block,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels-in_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.prelu=nn.PReLU()
        self.batchnorm=nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out_main_branch=self.conv(x)
        out_ext_branch=self.maxpool(x)
        out=torch.cat([out_main_branch,out_ext_branch],dim=1)
        out=self.batchnorm(out)
        out=self.prelu(out)
        return out


class Bottleneck(nn.Module):
    """
    From the idea of Resnet ,there are 2 branches : main and extension branches
    The element in main branch : (not contain maxpool, maxpool is added for downsampling bottleneck , in_channels=out_channels)
    The conv elements in extension branches : 1x1 conv , regular/dilated/asymmetri conv , 1x1 conv
    the first 1x1 conv is to reduce the dimensionality
    the second 1x1 conv is to expand the dimensionality
    Activation function : PReLU (followed by batchnorm layers)
    Regularizer : Dropout (default : 0)
    """
    def __init__(self,in_channels,internal_ratio=3,kernel_size=3,padding=1,dilation=1,isAsymmetric=False,dropout_prob=0.0):
        """
        kernel_size,padding is parameters of kernel(s) in conv (extension branch) (stride=1)
        """
        super(Bottleneck,self).__init__()
        internal_channels=in_channels//internal_ratio
        self.first_1x1=nn.Conv2d(in_channels,internal_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.batchnorm1=nn.BatchNorm2d(internal_channels)
        self.second_1x1=nn.Conv2d(internal_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.batchnorm2=nn.BatchNorm2d(in_channels)
        self.prelu1=nn.PReLU()
        self.prelu2=nn.PReLU()
        self.out_prelu=nn.PReLU()
        self.dropout=nn.Dropout(dropout_prob)

        if isAsymmetric:
            self.conv=nn.Sequential(
                nn.Conv2d(internal_channels,internal_channels,kernel_size=(kernel_size,1),stride=1,padding=(padding,0),bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(),

                nn.Conv2d(internal_channels,internal_channels,kernel_size=(1,kernel_size),stride=1,padding=(0,padding),bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )

        else:
            self.conv=nn.Sequential(
                nn.Conv2d(internal_channels,internal_channels,kernel_size=kernel_size,stride=1,dilation=dilation,padding=padding,bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        
    def forward(self,x):
        ext=self.prelu1(self.batchnorm1(self.first_1x1(x)))
        ext=self.conv(ext)
        ext=self.prelu2(self.batchnorm2(self.second_1x1(ext)))
        ext=self.dropout(ext)
        return self.out_prelu(x+ext)


class Downsampling(nn.Module):
    """
    The downsampling architecture is based on the bottleneck structure and maxpooling is added to the main branch
    the first 1x1 conv in the bottleneck is replaced by 2x2 with stride 2
    the maxpooling will return indices for maxunpooling later
    """
    def __init__(self,in_channels,out_channels,internal_ratio=3,dropout_prob=0.0):
        super(Downsampling,self).__init__()
        internal_channels=in_channels//internal_ratio
        self.main_branch=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        self.ext=nn.Sequential(

            #2x2 conv
            nn.Conv2d(in_channels,internal_channels,kernel_size=2,stride=2,padding=0,bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(),

            #3x3 conv
            nn.Conv2d(internal_channels,internal_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(),

            #1x1 conv
            nn.Conv2d(internal_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

        #Regularizer
        self.dropout=nn.Dropout(dropout_prob)

        self.out_prelu=nn.PReLU()
    
    def forward(self,x):
        out_main_branch,max_idxs=self.main_branch(x)
        out_ext=self.dropout(self.ext(x))

        #Zero padding to match the depths between out_ext and out_main_branch (depth of out_ext is greater than depth of x)
        zero_pad=torch.zeros(x.shape[0],out_ext.shape[1]-out_main_branch.shape[1],out_main_branch.shape[2],out_main_branch.shape[3])
        if torch.cuda.is_available():
            zero_pad=zero_pad.to("cuda")
        out_main_branch=torch.cat([out_main_branch,zero_pad],dim=1)

        return self.out_prelu(out_main_branch+out_ext),max_idxs


class Upsampling(nn.Module):
    """
    The architecture is still based on the bottleneck but the zero pad in the downsample is replaced by a conv
    """
    def __init__(self,in_channels,out_channels,internal_ratios=3,dropout_prob=0.0):
        super(Upsampling,self).__init__()
        internal_channels=in_channels//internal_ratios
        self.conv_main=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.unpool=nn.MaxUnpool2d(kernel_size=2)
        self.ext=nn.Sequential( 
            #first 1x1 conv
            nn.Conv2d(in_channels,internal_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(),

            #conv transpose for upsample
            nn.ConvTranspose2d(internal_channels,internal_channels,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(),

            #second 1x1conv
            nn.Conv2d(internal_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        #dropout
        self.dropout=nn.Dropout(dropout_prob)

        self.out_prelu=nn.PReLU()
    
    def forward(self,x,max_idxs):
        out_main_branch=self.unpool(self.conv_main(x),indices=max_idxs,output_size=(x.shape[2]*2,x.shape[3]*2))
        out_ext=self.ext(x)
        return self.out_prelu(out_main_branch+out_ext)
    

# def test():
    # x=torch.randn(1,3,512,512)
    # y=torch.randn(1,64,128,128)
    # initialBlock=Initial_Block(3,16)
    # regular_bottleneck=Bottleneck(64,3)
    # dilated_bottleneck=Bottleneck(64,3,3,padding=2,dilation=2)
    # asym_bottleneck=Bottleneck(64,3,kernel_size=5,padding=2)
    # downsample=Downsampling(in_channels=64,out_channels=128,internal_ratio=3)
    # out_downsample,max_idxs=downsample(y)
    # upsample=Upsampling(128,64)
    # out_initial_block=initialBlock(x)
    # print(out_initial_block.shape)
    # print(regular_bottleneck(y).shape)
    # print(dilated_bottleneck(y).shape)
    # print(asym_bottleneck(y).shape)
    # print(out_downsample.shape)
    # print(upsample(out_downsample,max_idxs).shape)
    # return

# if __name__=='__main__':
#     test()