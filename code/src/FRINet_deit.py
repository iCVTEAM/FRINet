import torch
import torch.nn as nn
import torch.nn.functional as F
from laplaciannet import RFAE
Act = nn.ReLU
from DeiT import deit_base_distilled_patch16_384

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList,nn.ModuleDict)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (LayerNorm,nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax,nn.AvgPool2d)):
            pass
        else:
            m.initialize()
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    def initialize(self):
        weight_init(self)
        
def window_partition(x,window_size):
    # input B C H W
    x = x.permute(0,2,3,1)
    B,H,W,C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows #B_ H_ W_ C

def window_reverse(windows,window_size,H,W):
    B=int(windows.shape[0]/(H*W/window_size/window_size))
    x = windows.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x.permute(0,3,1,2)

class MLP(nn.Module):
    def __init__(self, inchannel,outchannel, bias=False):
        super(MLP, self).__init__()
        self.conv1 = nn.Linear(inchannel, outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ln   = nn.LayerNorm(outchannel)
        

    def forward(self, x):
        return self.relu(self.ln(self.conv1(x))+x)
    def initialize(self):
        weight_init(self)


class WAttention(nn.Module): # x hf  y  lf
    def __init__(self, dim, num_heads=8, level=8,qkv_bias=True, qk_scale=None):
        super().__init__()
        self.level = level
        self.mul = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))
        self.add = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))

        self.conv_x = nn.Sequential(ConvBNReLu(dim,dim),ConvBNReLu(dim,dim,kernel_size=1,padding=0))

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        
        self.lnx = nn.LayerNorm(dim)
        self.lny = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)
        
        self.shortcut = nn.Linear(dim,dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3, stride=1, padding=1),
            LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,kernel_size=1, stride=1, padding=0),
            LayerNorm(dim)
        )
        self.mlp = MLP(dim,dim)


    def forward(self, x, y): 
        origin_size = x.shape[2]
        ws = origin_size//self.level//4 
        y = F.interpolate(y,size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv_x(x)

        x = window_partition(x,ws) 
        y = window_partition(y,ws)

        x = x.view(x.shape[0], -1, x.shape[3])
        sc1 = x
        x = self.lnx(x)
        y = y.view(y.shape[0], -1, y.shape[3])
        y = self.lny(y)
        B, N, C = x.shape
        y_kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q= self.q(x).reshape(B,N,1,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q      = x_q[0]   
        y_k, y_v = y_kv[0],y_kv[1]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale # B_ C WW WW
        attn = attn.softmax(dim=-1)
        x = (attn @ y_v).transpose(1, 2).reshape(B, N, C) # B' N C
        x = self.act(x+sc1)
        x = self.act(x+self.mlp(x))
        x = x.view(-1,ws,ws,C)
        x = window_reverse(x,ws,origin_size,origin_size) # B C H W
        x = self.act(self.conv2(x)+x)
        return x
        
    def initialize(self):
        weight_init(self)


class DB1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB1,self).__init__()
        self.squeeze2 = nn.Sequential(nn.Conv2d(outplanes, outplanes, kernel_size=3,stride=1,dilation=2,padding=2), LayerNorm(outplanes), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), LayerNorm(outplanes), nn.ReLU(inplace=True))
        self.relu = Act(inplace=True)

    def forward(self, x,z):
        if(z is not None and z.size()!=x.size()):
            z = F.interpolate(z, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.squeeze1(x)
        z = x+self.squeeze2(x) if z is None else x+self.squeeze2(x+z)       
        return z,z

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanesx,inplanesz,outplanes,head=8):
        super(DB2,self).__init__()
        self.inplanesx=inplanesx
        self.inplanesz=inplanesz

        self.short_cut = nn.Conv2d(inplanesz, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv=(nn.Sequential(
            nn.Conv2d((inplanesx+inplanesz),outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        ))
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z):
        z = F.interpolate(z, size=x.size()[2:], mode='bilinear')
        p = self.conv(torch.cat((x,z),dim=1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)


class ConvBNReLu(nn.Module):
    def __init__(self,inplanes,outplanes,kernel_size=3,dilation=1,padding=1):
        super(ConvBNReLu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes,outplanes,kernel_size=kernel_size, dilation=dilation,stride=1, padding=padding),
            LayerNorm(outplanes),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)

class FuseLayer(nn.Module):
    def __init__(self):
        super(FuseLayer,self).__init__()
        self.fuse_16 = WAttention(384,num_heads=8,level=1)
        self.fuse_8  = WAttention(192,num_heads=8,level=2)
        self.fuse_4  = WAttention(96,num_heads=8,level=4)

        self.sqz0     = nn.Sequential(ConvBNReLu(768,384),ConvBNReLu(384,384,kernel_size=1,padding=0))
        self.sqz_16  = DB1(768,384)
        self.sqz1     = nn.Sequential(ConvBNReLu(384,192),ConvBNReLu(192,192,kernel_size=1,padding=0))
        self.sqz_8   = DB1(768,192)
        self.sqz2     = nn.Sequential(ConvBNReLu(192,96),ConvBNReLu(96,96,kernel_size=1,padding=0))
        self.sqz_4   = DB1(768,96)

        self.R1  = WAttention(192,num_heads=8,level=2)
        self.R2  = WAttention(96,num_heads=8,level=4)
        self.R3  = WAttention(96,num_heads=8,level=4)

    def forward(self,lap_out,vit_out):  # 512 128 32 768     
        vit_16  = self.sqz0(vit_out[11])
        vit_8 = self.sqz1(vit_16)
        vit_4  = self.sqz2(vit_8)

# cross fusion
        fuse_16 = self.fuse_16(lap_out['out1_16'],vit_16)
        fuse_8 = self.fuse_8(lap_out['out2_8'],vit_8)
        fuse_4 = self.fuse_4(lap_out['out3_4'],vit_4)

# up fusion
        fuse_8 = self.R1(lap_out['out1_8'],fuse_8)
        fuse_4= self.R2(lap_out['out2_4'],fuse_4)
        fuse_4 = self.R3(lap_out['out1_4'],fuse_4)

        return fuse_16,fuse_8,fuse_4 
    def initialize(self):
        weight_init(self)

class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fuse0 = DB1(768,384)
        self.fuse1 = DB2(384,384,192)
        self.fuse2 = DB2(192,192,96)
        self.fuse3 = DB2(96,96,48)

    def forward(self,vitf,fuse_16,fuse_8,fuse_4):

        vitf,z = self.fuse0(vitf,None)
        out_16,z = self.fuse1(fuse_16,z)
        out_8,z = self.fuse2(fuse_8,z)
        out_4,z = self.fuse3(fuse_4,z)

        return vitf,out_16,out_8,out_4
    def initialize(self):
        weight_init(self)

class re_head(nn.Module):
    def __init__(self):
        super(re_head,self).__init__()
        self.head_4= nn.Sequential(ConvBNReLu(48,48),nn.Conv2d(48,3,3,1,0))
        self.head_8= nn.Sequential(ConvBNReLu(96,48),nn.Conv2d(48,3,3,1,0))
        self.head_16= nn.Sequential(ConvBNReLu(384,48),nn.Conv2d(48,3,3,1,0))
    def forward(self,fuse_4,fuse_8,fuse_16):
        re_4 = self.head_4(fuse_4)
        re_8 = self.head_8(fuse_8)
        re_16 = self.head_16(fuse_16)

        return re_4, re_8,re_16
    def initialize(self):
        weight_init(self)

class FRINet(nn.Module):
    def __init__(self, cfg=None):
        super(FRINet, self).__init__()
        self.cfg      = cfg
        self.linear1 = nn.Conv2d(48, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1)

        self.fuselayer = FuseLayer()
        self.decoder = decoder()
        self.re_decoder = decoder()
        self.re_head = re_head()

        self.RFAE    = RFAE()
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.bkbone   = deit_base_distilled_patch16_384()
        
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain=torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k,v in pretrain.items():
                new_state_dict[k[7:]] = v  
            self.load_state_dict(new_state_dict, strict=True)  

    def forward(self, img,RFC,shape=None,mask=None):
        shape = img.size()[2:] if shape is None else shape
        img = F.interpolate(img, size=(384,384), mode='bilinear',align_corners=True)
        vitf = self.bkbone(img) # Low Frequency Representation

        cnn_out = self.RFAE(RFC) #High Frequency Representation Array
        fuse_16,fuse_8,fuse_4 = self.fuselayer(cnn_out,vitf) # Progressive Frequency Representation Integration
        vitout16,out_16,out_8,out_4 = self.decoder(vitf[11],fuse_16,fuse_8,fuse_4)  # COD decoder
        re_16 ,_,re_8 ,re_4  = self.re_decoder(vitf[11],cnn_out['out1_16'],cnn_out['out2_8'],cnn_out['out3_4']) # Reconstruct decoder
        
        pred1 = F.interpolate(self.linear1(out_4), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linear2(out_8), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.linear3(out_16), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.linear4(vitout16), size=shape, mode='bilinear')

        re_4,re_8,re_16 = self.re_head(re_4,re_8,re_16)

        return pred1,pred2,pred3,pred4, re_4, re_8,re_16
