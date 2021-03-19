import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class HWHW_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(HWHW_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.sigmoid(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CC_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CC_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.sigmoid(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class HH_Module(nn.Module):
    def __init__(self, in_dim):
        super(HH_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (H) X (H)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.sigmoid(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class WW_Module(nn.Module):
    def __init__(self, in_dim):
        super(WW_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (W) X (W)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.sigmoid(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CHCH_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CHCH_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X CH X CH
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C*height, -1)
        proj_key = x.view(m_batchsize, C*height, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.sigmoid(energy_new)
        proj_value = x.view(m_batchsize, C*height, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CWCW_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CWCW_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X CW X CW
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C*width, -1)
        proj_key = x.view(m_batchsize, C*width, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.sigmoid(energy_new)
        proj_value = x.view(m_batchsize, C*width, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out #+ max_out
        return self.sigmoid(out)

class RowAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(RowAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x.permute(0,2,1,3), dim=1, keepdim=True)
        x = self.conv1(avg_out)
        return self.sigmoid(x)

class ColumnAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ColumnAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x.permute(0,3,2,1), dim=1, keepdim=True)
        x = self.conv1(avg_out)
        return self.sigmoid(x)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # for pytorch 0.3.1
        #nn.init.constant(self.W.weight, 0)
        #nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        tx = self.body(x)

        return tx



## define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MaskBranchDownUp, self).__init__()
        
        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
       
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx
## gpm
class GPM(nn.Module):
    def __init__(self, n_feats):
        super(GPM, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(n_feats//4, n_feats, kernel_size=3, padding=1, bias=False)
        self.convkg = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)
        self.conv31 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        w = x1.shape[2]
        h = x1.shape[3]
        if w // 2 == 0 and h // 2 == 0:
            x_concat = torch.cat([x1[:,:,:w//2,:h//2], x1[:,:,w//2:,:h//2], x1[:,:,:w//2,h//2:], x1[:,:,w//2:,h//2:]],1)
            x2 = self.convkg(x_concat)
            x_fea = torch.zeros([x2.shape[0], x2.shape[1]//4, x.shape[2], x.shape[3]]).cuda()
            x_fea[:,:,:w//2,:h//2] = x2[:,:16,:,:]
            x_fea[:,:,w//2:,:h//2] = x2[:,16:32,:,:]
            x_fea[:,:,:w//2,h//2:] = x2[:,32:48,:,:]
            x_fea[:,:,w//2:,h//2:] = x2[:,48:64,:,:]
            out = self.conv3(x_fea)
        else:
            out = self.conv3(x1)
        return out


## define nonlocal mask branch
class NLMaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLMaskBranchDownUp, self).__init__()
        
        MB_RB1 = []
        MB_RB1.append(GPM(n_feat))
        #MB_RB1.append(NonLocalBlock2D(n_feat, n_feat//2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())
        
        MB_all2c = []
        MB_all2c.append(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False))
        MB_all2h = []
        MB_all2h.append(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False))
        MB_all2w = []
        MB_all2w.append(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False))
        MB_all2 = []
        MB_all2.append(nn.Conv2d(n_feat * 4, n_feat, 1, padding=0, bias=False))
        #MB_all2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
        
        self.cha1 = CC_Module(n_feat)
        self.row1 = HH_Module(n_feat)
        self.col1 = WW_Module(n_feat)
        self.cha2 = ChannelAttention(n_feat)
        self.row2 = RowAttention()
        self.col2 = ColumnAttention()

        self.all2c = nn.Sequential(*MB_all2c)
        self.all2h = nn.Sequential(*MB_all2h)
        self.all2w = nn.Sequential(*MB_all2w)
        self.all2 = nn.Sequential(*MB_all2)
        self.wc1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wh1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.ww1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wc1.data.fill_(0)
        self.wh1.data.fill_(0)
        self.ww1.data.fill_(0)
        self.wc2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wh2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.ww2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wc2.data.fill_(0)
        self.wh2.data.fill_(0)
        self.ww2.data.fill_(0)



    def forward(self, x):
        x_RB1 = self.MB_RB1(x)

        x_cha2 = self.cha2(x_RB1) * x_RB1
        x_row2 = self.row2(x_RB1) * x_RB1.permute(0,2,1,3)
        x_col2 = self.col2(x_RB1) * x_RB1.permute(0,3,2,1)
        mx = self.wc2 * x_cha2 + self.wh2 * x_row2.permute(0,2,1,3) + self.ww2 * x_col2.permute(0,3,2,1) + x_RB1 
        mx = self.all2c(mx) 

        #x_cha1 = self.cha1(mx) 
        #x_row1 = self.row1(mx)
        #x_col1 = self.col1(mx)
        #mx = self.wc1 * x_cha1 + self.wh1 * x_row1 + self.ww1 * x_col1 + mx
        #mx = self.all2w(mx) 


        if x_RB1.shape[2] // 2 ==0 and x_RB1.shape[3] // 2 ==0: 
            x_Down = self.MB_Down(mx)
            x_RB2 = self.MB_RB2(x_Down)
            x_Up = self.MB_Up(x_RB2)
            x_preRB3 = x_RB1 + x_Up  
            x_RB3 = self.MB_RB3(x_preRB3)
            x_1x1 = self.MB_1x1conv(x_RB3)
        #     mx_att = self.weight_c*x_cc + self.weight_h*x_hh + self.weight_w*x_ww
        #     x_1x1 = torch.cat([mx_att,x_RB3], 1)
        #     x_1x1 = self.MB_1x1conv(x_1x1)       
            mx = self.MB_sigmoid(x_1x1)
        else: 
            x_RB2 = self.MB_RB2(mx)
            x_preRB3 = x_RB1 + x_RB2
            x_RB3 = self.MB_RB3(x_preRB3)
            x_1x1 = self.MB_1x1conv(x_RB3)

        #     mx_att = self.weight_c*x_cc + self.weight_h*x_hh + self.weight_w*x_ww
        #     x_1x1 = torch.cat([mx_att,x_RB3], 1)
        #     x_1x1 = self.MB_1x1conv(x_1x1)
            mx = self.MB_sigmoid(x_1x1)

        #mx_att = x_hwhw + x_cc + x_hh + x_ww + x_cwcw + x_chch
        #mx_att = x_ww 
        return mx#, mx_att




## define residual attention module 
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


## define nonlocal residual attention module 
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx
