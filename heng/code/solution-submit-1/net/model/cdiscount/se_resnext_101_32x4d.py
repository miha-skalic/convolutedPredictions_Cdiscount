from common import*
from dataset.transform import *

#  https://github.com/facebookresearch/ResNeXt
#  https://github.com/clcarwin/convert_torch_to_pytorch
#    mean = { 0.485, 0.456, 0.406 },
#     std = { 0.229, 0.224, 0.225 },


image_to_tensor_transform = pytorch_image_to_tensor_transform


###  modules  ###-----------------------------
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

class ConvBn2d(nn.Module):

    def merge_bn(self):
        #raise NotImplementedError
        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat



    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# ----
class SEScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x




class SENextBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, reduction, groups=32, is_down_sample=False, stride=1):
        super(SENextBlock, self).__init__()
        self.is_down_sample = is_down_sample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, groups=groups)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.scale    = SEScale(out_planes, reduction)

        if is_down_sample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        if self.is_down_sample:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv_bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn2(x)
        x = F.relu(x,inplace=True)
        x = self.conv_bn3(x)
        x = self.scale(x)*x + residual
        x = F.relu(x,inplace=True)
        return x


# layers ##---------------------------------------------------------------------

def make_layer(in_planes, planes, out_planes, reduction, num_blocks, stride):
    layers = []
    layers.append(SENextBlock(in_planes, planes, out_planes, reduction, is_down_sample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SENextBlock(out_planes, planes, out_planes, reduction))

    return nn.Sequential(*layers)


def make_layer0(in_planes, out_planes,stride):
    layers = [
        ConvBn2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
    ]
    return nn.Sequential(*layers)



###  network ###-----------------------------
class SEResnext101(nn.Module):

    # def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
    #     #raise NotImplementedError
    #
    #     pytorch_state_dict = torch.load(pytorch_file)
    #     state_dict = self.state_dict()
    #     keys = list(state_dict.keys())
    #     for key in keys:
    #         if any(s in key for s in skip):
    #             continue
    #         #print(key)
    #         state_dict[key] = pytorch_state_dict[key]
    #     self.load_state_dict(state_dict)



    #-----------------------------------------------------------------------
    def __init__(self, in_shape=(3,180,180), num_classes=5270,  type = 'logits' ):

        super(SEResnext101, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes
        self.type = type

        self.layer0 = make_layer0(in_channels, 64, stride=2)
        self.layer1 = make_layer(  64, 128,  256,  16, num_blocks=3 ,stride=1) #4
        self.layer2 = make_layer( 256, 256,  512,  32, num_blocks=4 ,stride=2) #5
        self.layer3 = make_layer( 512, 512, 1024,  64, num_blocks=23,stride=2) #6
        self.layer4 = make_layer(1024,1024, 2048, 128, num_blocks=3 ,stride=2) #7
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) #;print(x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)

        if self.type == 'last_feature':
            return   x

        if self.type == 'logits':
            x = self.fc (x)  #logits
            return   x



#####################################################################################################3

def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5270
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)


    net = SEResnext101(in_shape=(C,H,W), num_classes=num_classes)
    # net.load_pretrain_pytorch_file(
    #         '/root/share/project/kaggle/cdiscount/results/resnext101-180-00c/checkpoint/00117000_model.pth',
    #         skip=['fc.','.scale.']
    #     )
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

def run_check_net_cdiscount():

    checkpoint = '/root/share/project/kaggle/cdiscount/model/se-resnext101-180-03b/00121000_model.pth'
    net = SEResnext101(in_shape=(3,180,180), num_classes=5270)
    net.load_state_dict(torch.load(checkpoint))
    net.cuda()
    net.eval()
    #net.merge_bn()

    scores = np.zeros((12,5270),np.uint8)
    names = [ '10-0', '10-1', '10-2', '14-0', '21-0', '24-0', '27-0', '29-0', '32-0', '32-1', '32-2', '32-3']
    for n,name in enumerate(names):
        image_file = '/media/ssd/data/kaggle/cdiscount/image/test/' + name + '.jpg'
        assert(os.path.exists(image_file))
        image = cv2.imread(image_file)

        logit = net( Variable(image_to_tensor_transform(image).unsqueeze(0) ).cuda() )
        prob  = F.softmax(logit,dim=1)
        prob  = prob.data.cpu().numpy()[0]
        scores[n] = prob*255

    print('')
    product_scores =[]
    product_scores.append( scores[0:3].mean(0)   )
    product_scores.append( scores[3]             )
    product_scores.append( scores[4]             )
    product_scores.append( scores[5]             )
    product_scores.append( scores[6]             )
    product_scores.append( scores[7]             )
    product_scores.append( scores[8:12].mean(0)  )

    for s in product_scores:
        print('%05d,  %0.5f'%(np.argmax(s), np.max(s)/255))


def run_check_net_convert():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 4
    num_classes = 5270
    C,H,W = 3,180,180
    inputs = torch.randn(batch_size,C,H,W)
    x = Variable(inputs).cuda()

    net = SEResnext101(in_shape=(C,H,W), num_classes=num_classes, type = 'logits')
    net.cuda()
    net.train()

    #original
    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)
    print('before -------------------')
    print(type(net))
    print(net)
    print('')
    print('probs')
    print(probs)

    #convert
    net.type = 'last_feature'
    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)
    print('before -------------------')
    print(type(net))
    print(net)
    print('')
    print('probs')
    print(probs)

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_net()
    run_check_net_convert()
    #run_check_net_cdiscount()












