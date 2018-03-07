## https://github.com/cypw/DPNs
## https://github.com/rwightman/pytorch-dpn-pretrained

from common import*
# if __name__ == '__main__':
#     from adaptive_avgmax_pool import adaptive_avgmax_pool2d
# else:
#     from .adaptive_avgmax_pool import adaptive_avgmax_pool2d
#

# https://github.com/rwightman/pytorch-dpn-pretrained/blob/fc2ca159630c8c190702763613883e34b2274956/model_factory.py
#      mean=[124 / 255, 117 / 255, 104 / 255],
#      std=[1 / (.0167 * 255)] * 3)



def image_to_tensor_transform(image):

    mean=[124 / 255, 117 / 255, 104 / 255]
    std =[1 / (.0167 * 255)] * 3

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor



###  modules  ###-----------------------------

class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


###  network ###-----------------------------

class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128)):
        super(DPN, self).__init__()

        self.b = b
        bw_factor = 1 if small else 4
        blocks = collections.OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

    def forward(self, x):
        x = self.features(x)
        return x



class DPNet92(nn.Module):

    def load_pretrain_file(self,pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.dpn.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if key in skip: continue
            state_dict[key] = pretrain_state_dict[key]

        self.dpn.load_state_dict(state_dict)

        ##fixed  for mihas model
        state_dict = self.fc.state_dict()
        state_dict['weight']= pretrain_state_dict['classifier.weight']
        state_dict['bias'  ]= pretrain_state_dict['classifier.bias'  ]
        self.fc.load_state_dict(state_dict)

        pass

    #-----------------------------------------------------------------------

    def __init__(self, in_shape=(3,180,180), num_classes=5270, type='logits' ):
        super(DPNet92, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes
        self.type = type

        test_time_pool=True
        self.dpn = DPN(
            num_init_features=64, k_r=96, groups=32,
            k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        ) #setting for dpn92

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.fc = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)


    def forward(self, x):
        x = self.dpn(x)  # torch.Size([1, 2688, 6, 6])
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = F.dropout(x, p=0.5, training=self.training)

        if self.type == 'last_feature':
            x = x.view(x.size(0), -1)
            return   x

        if self.type == 'logits':
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return   x






########################################################################################
def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5270
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)

    net = DPNet92(in_shape=(C,H,W), num_classes=num_classes)
    net.load_pretrain_file(
           '/root/share/project/kaggle/cdiscount/model/dpnet92/mihas/dpn92-0066.pth',
            skip=[] #'fc.weight'	,'fc.bias'
        )
    net.cuda()
    net.train()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    print(net)

    print('probs')
    print(probs)

    #merging
    # net.eval()
    # net.merge_bn()



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()



