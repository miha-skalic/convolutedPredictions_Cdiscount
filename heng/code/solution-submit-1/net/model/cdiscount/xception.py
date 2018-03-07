# https://arxiv.org/pdf/1610.02357.pdf


# "Xception: Deep Learning with Depthwise Separable Convolutions" - Francois Chollet (Google, Inc), CVPR 2017

# separable conv pytorch
#  https://github.com/szagoruyko/pyinn
#  https://github.com/pytorch/pytorch/issues/1708
#  https://discuss.pytorch.org/t/separable-convolutions-in-pytorch/3407/2
#  https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/3


from common import *
# import pyinn as P
# from pyinn.modules import Conv2dDepthwise


def image_to_tensor_transform(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - 0.5) *2
    tensor[1] = (tensor[1] - 0.5) *2
    tensor[2] = (tensor[2] - 0.5) *2
    return tensor


#----- helper functions ------------------------------

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
class SeparableConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, is_bn=True):
        super(SeparableConvBn2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=False)  #depth_wise
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False) #point_wise
        #self.conv1 = Conv2dDepthwise(in_channels,  kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        #self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_channels)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x

#
class SBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels, is_first_relu=True):
        super(SBlock, self).__init__()
        self.is_first_relu=is_first_relu

        self.downsample = ConvBn2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2)
        self.conv1 = SeparableConvBn2d(in_channels,     channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(   channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self,x):
        residual = self.downsample(x)
        if self.is_first_relu:
            x = F.relu(x,inplace=False)
        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
        x = x + residual

        return x



class XBlock(nn.Module):

    def __init__(self, in_channels):
        super(XBlock, self).__init__()

        self.conv1 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

    def forward(self,x):

        residual = x
        x = F.relu(x,inplace=True)
        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.relu(x,inplace=True)
        x = self.conv3(x)
        x = x + residual

        return x



class EBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(EBlock, self).__init__()

        self.conv1 = SeparableConvBn2d(in_channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(channels,out_channels, kernel_size=3, padding=1, stride=1)


    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.relu(x,inplace=True)

        return x


class Xception(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        pytorch_state_dict = torch.load(pytorch_file,map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            #print(key)
            state_dict[key] = pytorch_state_dict[key]
        self.load_state_dict(state_dict)

    #-----------------------------------------------------------------------

    def __init__(self, in_shape=(3,180,180), num_classes=5270,  type = 'logits' ):

        super(Xception, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes = num_classes
        self.type = type

        self.entry0  = nn.Sequential(
            ConvBn2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.entry1  = SBlock( 64,128,128,is_first_relu=False)
        self.entry2  = SBlock(128,256,256)
        self.entry3  = SBlock(256,728,728)

        self.middle1 = XBlock(728)
        self.middle2 = XBlock(728)
        self.middle3 = XBlock(728)
        self.middle4 = XBlock(728)
        self.middle5 = XBlock(728)
        self.middle6 = XBlock(728)
        self.middle7 = XBlock(728)
        self.middle8 = XBlock(728)

        self.exit1 = SBlock( 728, 728,1024)
        self.exit2 = EBlock(1024,1536,2048)
        self.fc = nn.Linear(2048, num_classes)


    def forward(self,x):

        x = self.entry0(x)    #; print('entry0 ', x.size())
        x = self.entry1(x)    #; print('entry1 ', x.size())
        x = self.entry2(x)    #; print('entry2 ', x.size())
        x = self.entry3(x)    #; print('entry3 ', x.size())
        x = self.middle1(x)   #; print('middle1 ',x.size())
        x = self.middle2(x)   #; print('middle2 ',x.size())
        x = self.middle3(x)   #; print('middle3 ',x.size())
        x = self.middle4(x)   #; print('middle4 ',x.size())
        x = self.middle5(x)   #; print('middle5 ',x.size())
        x = self.middle6(x)   #; print('middle6 ',x.size())
        x = self.middle7(x)   #; print('middle7 ',x.size())
        x = self.middle8(x)   #; print('middle8 ',x.size())
        x = self.exit1(x)     #; print('exit1 ',x.size())
        x = self.exit2(x)     #; print('exit2 ',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        ##x = F.dropout(x, training=self.training, p=0.25)     #

        if self.type == 'last_feature':
            return x

        if self.type == 'logits':
            x = self.fc (x)  #logits
            return   x


########################################################################################################
def check_depthwise_conv():

    #https://github.com/szagoruyko/pyinn

    batch_size  = 1
    C,H,W = 3,180,180
    inputs = torch.randn(batch_size,C,H,W)

    in_channels=C
    kernel_size=3
    padding=1
    stride=2

    conv1a = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                           groups=in_channels, bias=False)  # depth_wise

    conv1b = Conv2dDepthwise(in_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

    x = Variable(inputs).cuda()
    y_a = conv1a(x)

    print(x)
    print(y_a)



def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5000
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]


    net = Xception(in_shape=in_shape, num_classes=num_classes, type = 'logits')
    # net.load_pretrain_pytorch_file(
    #         '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.keras.convert.pth',
    #         skip=['fc.weight'	,'fc.bias']
    #     )
    net.cuda().train()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    logits = net.forward(x)
    probs  = F.softmax(logits,dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    print(net)

    print('probs')
    print(probs)

    #merging
    # net.eval()
    # net.merge_bn()

def run_check_feature_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5000
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]


    net = Xception(in_shape=in_shape, num_classes=num_classes, type = 'last_feature')
    net.cuda().eval()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    features = net.forward(x)

    print(type(net))
    print(net)

    print('features')
    print(features)
    print(features.size())



def run_check_net_cdiscount():

    checkpoint = '/root/share/project/kaggle/cdiscount/model/xception-180-01b/checkpoint/00158000_model.pth'
    net = Xception(in_shape=(3,180,180), num_classes=5270)
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

    for i, s in enumerate(product_scores):
        print('%02d : %05d,  %0.5f'%(i, np.argmax(s), np.max(s)/255))




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_net()
    run_check_feature_net()
    #check_depthwise_conv()
    #run_check_net_cdiscount()

