import common
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

def make_model(args, parent=False):
    return TSAN(args)

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class FeaExtra(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(FeaExtra, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        conv = conv
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1, stride=1)
        self.dilated_conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1, stride=1)
        self.conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1, stride=1)
        self.dilated_conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1, stride=1)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1_1 = self.relu(self.conv_3_1(x))
        input_1_2 = self.relu(self.dilated_conv_3_1(input_1_1))
        input_2_1 = self.relu(self.dilated_conv_3_2(x))
        input_2_2 = self.relu(self.conv_3_2(input_2_1))
        output = torch.cat([input_1_1, input_1_2, input_2_1, input_2_2], 1)
        output = self.relu(self.confusion(output))
        return output

class make_dense(nn.Module):
    def __init__(self, n_channels, n_feats=64, i=0):
        super(make_dense, self).__init__()
        kernel_size_1 = 3
        if i < 3:
            s = i+1
        else:
            s = i - 2*(i-3)
        #print(s)
        conv = common.default_conv
        self.conv_1_1 = nn.Conv2d(n_channels, n_feats, 1, padding=0, stride=1)
        self.conv_1_2 = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=s, stride=1, dilation=s)
        self.dilated_conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=s, stride=1, dilation=s)
        self.conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=s, stride=1, dilation=s)
        self.dilated_conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=s, stride=1, dilation=s)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out1x1 = self.relu(self.conv_1_1(x))
        input_1_1 = self.relu(self.conv_3_1(out1x1))
        input_1_2 = self.relu(self.dilated_conv_3_1(input_1_1))
        input_2_1 = self.relu(self.dilated_conv_3_2(out1x1))
        input_2_2 = self.relu(self.conv_3_2(input_2_1))
        output = torch.cat([input_1_1, input_1_2, input_2_1, input_2_2], 1)
        output = self.relu(self.conv_1_2(output))
        #output = output + out1x1
        output = torch.cat([output, x], 1)
        return output

class DRB(nn.Module):
    def __init__(self, n_channels, n_denselayer=6, n_feats=64):
        super(DRB, self).__init__()
        nChannels_ = n_channels
        modules = []
        for i in range(n_denselayer):    
            modules.append(make_dense(nChannels_, n_feats, i))
            nChannels_ += n_feats 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, n_feats, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = x + out
        return out

class CSB(nn.Module):
    def __init__(self, n_feats):
        super(CSB, self).__init__()
        self.conv1 = nn.Conv2d(n_feats*4, n_feats*4, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # print(x.size(2)//2)
        f1 = x[:, : , 0:x.size(2)//2, 0:x.size(3)//2]
        f2 = x[:, : , x.size(2)//2:, 0:x.size(3)//2]
        f3 = x[:, : , 0:x.size(2)//2, x.size(3)//2:]
        f4 = x[:, : , x.size(2)//2:, x.size(3)//2:]
        # print(f1.size(), f2.size(), f3.size(), f4.size())
        f_all = torch.cat([f1,f2,f3,f4], 1)
        # print(f_all.size())
        f_all = self.conv1(f_all)
        out = x
        out[:,:,0:x.size(2)//2, 0:x.size(3)//2] = f_all[:,0:f_all.size(1)//4,:,:]
        out[:,:,x.size(2)//2:, 0:x.size(3)//2] = f_all[:,f_all.size(1)//4:f_all.size(1)//2,:,:]
        out[:,:,0:x.size(2)//2, x.size(3)//2:] = f_all[:,f_all.size(1)//2:f_all.size(1)*3//4,:,:]
        out[:,:,x.size(2)//2:, x.size(3)//2:] = f_all[:,f_all.size(1)*3//4:,:,:]
        return out

class First_Order(nn.Module):
    def __init__(self):
        super(First_Order, self).__init__()
        self.adaptive = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        first_c = F.sigmoid(self.adaptive(x))
        first_c = first_c * x
        xh = x.permute(0,2,1,3)
        first_h = F.sigmoid(self.adaptive(xh))
        first_h = (first_h * xh).permute(0,2,1,3)
        xw = x.permute(0,3,2,1)
        first_w = F.sigmoid(self.adaptive(xw))
        first_w = (first_w * xw).permute(0,3,2,1)

        return first_c + first_h + first_w + x

class Second_Order(nn.Module):
    def __init__(self):
        super(Second_Order, self).__init__()

    def forward(self, x):
        second_c = F.sigmoid(torch.mean(x, 1).unsqueeze(1))
        second_c = second_c * x
        xh = x.permute(0,2,1,3)
        second_h = F.sigmoid(torch.mean(xh, 1).unsqueeze(1))
        second_h = (second_h * xh).permute(0,2,1,3)
        xw = x.permute(0,3,2,1)
        second_w = F.sigmoid(torch.mean(xw, 1).unsqueeze(1))
        second_w = (second_w * xw).permute(0,3,2,1)

        return second_c + second_h + second_w + x

class MCAB(nn.Module):
    def __init__(self, n_channels, n_denselayer=6, n_feats=64):
        super(MCAB, self).__init__()
        self.drb = DRB(n_channels, n_denselayer, n_feats)
        self.csb = CSB(n_feats)
        self.first = First_Order()
        self.second = Second_Order()
        self.conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        down_b = self.drb(x)
        top_b = self.csb(x)
        top_b = self.second(self.first(top_b))
        top_b = F.sigmoid(self.conv3(top_b))
        return down_b*top_b



class TSAN(nn.Module):
    def __init__(self,args):
        super(TSAN, self).__init__()
        conv=common.default_conv
        n_feats = args.n_feats
        scale = args.scale[0]
        n_colors = args.n_colors
        kernel_size_1 = 3
        n_channels = 64
        n_denselayer = 6
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.first_conv = nn.Conv2d(n_colors, n_feats, 1, padding=0, stride=1)

        modules_body1 = MCAB(n_channels, n_denselayer, n_feats)
        modules_body2 = MCAB(n_channels, n_denselayer, n_feats)
        modules_body3 = MCAB(n_channels, n_denselayer, n_feats)

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * 3, n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size_1)]
        modules_up = [
            common.Upsampler(conv, scale, n_feats, act=False)]
        self.output1 = conv(n_feats, n_colors, 1)
        self.output2 = conv(n_feats, n_colors, 1)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.relu = nn.ReLU(inplace=True)
        self.body1 = modules_body1
        self.body2 = modules_body2
        self.body3 = modules_body3

        self.tail = nn.Sequential(*modules_tail)
        self.up = nn.Sequential(*modules_up)
        self.first2_conv = nn.Conv2d(n_colors, n_feats, 1, padding=0, stride=1)
        self.refine = self.make_layer(Conv_ReLU_Block,5)

    def make_layer(self, block, num_layer):
        layers = []
        for _ in range(num_layer):
            layers.append((block()))
        return nn.Sequential(*layers)   

    def forward(self, x):
        x = self.sub_mean(x)
        first_out = self.relu(self.first_conv(x))
        out_dense1 = self.body1(first_out)
        out_dense2 = self.body2(out_dense1)
        out_dense3 = self.body3(out_dense2)
        dense = torch.cat([out_dense1, out_dense2, out_dense3], 1)

        out = self.tail(dense)
        out = out + first_out
        out = self.up(out)
        out = self.output1(out)
        out1 = self.add_mean(out)
        out2 = self.first2_conv(out)
        out2 = self.refine(out2)
        out2 = self.output2(out2)
        #print(out2.size(), out.size())
        out2 = out2 + out
        out2 = self.add_mean(out2)
        return out1, out2

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == '__main__':
    net = MCAB(n_channels=64,n_denselayer=2,n_feats=64)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # net = net.cuda()

    var1 = torch.FloatTensor(1, 64, 64, 64)
    # print(var1)

    var2 = torch.FloatTensor(1, 10, 32, 32)
    var3 = torch.FloatTensor(1, 10, 32, 32)
    out1 = net(var1)
    # print(out1)
    print('*************')
    print(out1.size()) 
