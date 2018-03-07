from common import*
from dataset.transform  import*
#image_to_tensor_transform = pytorch_image_to_tensor_transform


# squeeze-excite
# https://github.com/titu1994/keras-squeeze-excite-network

# https://github.com/moskomule/senet.pytorch
# https://github.com/hujie-frank/SENet
# https://github.com/ruotianluo/pytorch-resnet



def image_to_tensor_transform(image):
    mean   = [104, 117, 123]
    image  = image.astype(np.float32) - np.array(mean, np.float32)
    image  = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)
    return tensor

def tensor_to_image_transform(tensor):
    image = tensor.numpy()
    image = image.transpose((1,2,0))
    mean   = [104, 117, 123]
    image  = image.astype(np.float32) + np.array(mean, np.float32)
    image  = image.astype(np.uint8)
    return image


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
        std_bn_weight   = (std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
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



class SEBottleneck(nn.Module):
    def __init__(self, in_planes, planes, out_planes, reduction, is_downsample=False, stride=1):
        super(SEBottleneck, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.scale    = SEScale(out_planes, reduction)
        if is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = self.conv_bn1(x)
        z = F.relu(z,inplace=True)
        z = self.conv_bn2(z)
        z = F.relu(z,inplace=True)
        z = self.conv_bn3(z)
        z = z*self.scale(z)
        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x

        z = F.relu(z,inplace=True)
        return z


# layers ##---------------------------------------------------------------------

def make_layer(in_planes, planes, out_planes, reduction, num_blocks, stride):
    layers = []
    layers.append(SEBottleneck(in_planes, planes, out_planes, reduction, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SEBottleneck(out_planes, planes, out_planes, reduction))

    return nn.Sequential(*layers)


def make_layer0(in_planes, out_planes,stride):
    layers = [
        ConvBn2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
    ]
    return nn.Sequential(*layers)


###  network ###-----------------------------
class SEResNet50(nn.Module):

    def load_pretrain_file(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file,map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue

            pretrain_key = key
            #print('%36s'%key, ' ','%-36s'%pretrain_key)
            state_dict[key] = pretrain_state_dict[pretrain_key]
        self.load_state_dict(state_dict)


    def merge_bn(self):
        print ('merging bn ....')
        for name, m in self.named_modules():
            if isinstance(m, (ConvBn2d,)):
                print('\t%s'%name)
                m.merge_bn()
        print('')


    def convert_to_feature_extract(self):
        self.fc = Identity()
        return 2048


    #-----------------------------------------------------------------------
    def __init__(self, in_shape=(3,180,180), num_classes=5270 ):

        super(SEResNet50, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes

        self.layer0 = make_layer0(in_channels, 64, stride=2)
        self.layer1 = make_layer (  64,  64,  256, reduction= 16, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer2 = make_layer ( 256, 128,  512, reduction= 32, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer3 = make_layer ( 512, 256, 1024, reduction= 64, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer4 = make_layer (1024, 512, 2048, reduction=128, num_blocks=3, stride=2)  #out = 512*4 = 2048
        self.fc  = nn.Linear(2048, num_classes)

    def forward(self, x):
                            #; print('input ' ,x.size())
        x = self.layer0(x)  #; print('layer0 ',x.size())
        x = self.layer1(x)  #; print('layer1 ',x.size())
        x = self.layer2(x)  #; print('layer2 ',x.size())
        x = self.layer3(x)  #; print('layer3 ',x.size())
        x = self.layer4(x)  #; print('layer4 ',x.size())

        #x = F.adaptive_avg_pool2d(x, output_size=1)
        x = F.adaptive_avg_pool2d(x, output_size=2)
        x = F.adaptive_max_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc (x)
        return x #logits



########################################################################################################
# test some images
#   https://github.com/soeaver/caffe-model/blob/master/cls/synset.txt
#   https://github.com/ruotianluo/pytorch-resnet/blob/master/synset.py ()
#
#    (441)  810 n02823750 beer glass
#    (  1)  449 n01443537 goldfish, Carassius auratus
#    (  9)  384 n01518878 ostrich, Struthio camelus
#    ( 22)  397 n01614925 bald eagle, American eagle, Haliaeetus leucocephalus
#    (281)  173 n02123045 tabby, tabby cat


def run_check_net_imagenet():
    num_classes = 1000
    C,H,W = 3,224,224
    net = SEResNet50(in_shape=(C,H,W), num_classes=num_classes)
    net.load_pretrain_file(
            '/root/share/data/models/reference/senet/ruotianluo/SE-ResNet-50.convert.pth',
            skip=[]
        )
    #net.cuda()
    net.eval()


    #image = cv2.imread('/root/share/data/imagenet/dummy/256x256/beer_glass.jpg')
    #image = cv2.imread('/root/share/data/imagenet/dummy/256x256/goldfish.jpg')
    #image = cv2.imread('/root/share/data/imagenet/dummy/256x256/blad_eagle.jpg')
    #image = cv2.imread('/root/share/data/imagenet/dummy/256x256/ostrich.jpg')
    image = cv2.imread('/root/share/data/imagenet/dummy/256x256/tabby_cat.jpg')
    #image = cv2.imread('/root/share/data/imagenet/dummy/256x256/bullet_train.jpg')


    #pre process ----
    #  https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNet-50.prototxt
    mean  = [104, 117, 123]
    image = image.astype(np.float32) - np.array(mean, np.float32)
    image = image.transpose((2,0,1))


    #pre process ----


    #run net
    logits = net( Variable(torch.from_numpy(image).unsqueeze(0).float() ) )
    probs  = F.softmax(logits,dim=1).data.numpy().reshape(-1)
    #print('probs\n',probs)

    #check
    print('results ', np.argmax(probs), ' ', probs[np.argmax(probs)])




def run_check_net_cdiscount():

    checkpoint = '/root/share/project/kaggle/cdiscount/results/__submission__/stable-00/excited-resnet50-180-00a/checkpoint/00061000_model.pth'
    net = SEResNet50(in_shape=(3,180,180), num_classes=5270)
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



def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5270
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)



    net = SEResNet50(in_shape=(C,H,W), num_classes=num_classes)
    net.load_pretrain_file(
            '/root/share/project/kaggle/cdiscount/model/se-resnet50-180-00a/checkpoint/00061000_model.pth',
            skip=[] #['fc.']
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
    #print(net)

    print('probs')
    print(probs)

    #merging
    #net.eval()
    #net.merge_bn()


def run_check_net_convert():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 4
    num_classes = 5270
    C,H,W = 3,180,180
    inputs = torch.randn(batch_size,C,H,W)
    x = Variable(inputs).cuda()

    net = SEResNet50(in_shape=(C,H,W), num_classes=num_classes)
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
    net.convert_to_feature_extract()
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


    #run_check_net_cdiscount()
    run_check_net()
    #run_check_net_imagenet()
    #run_check_net_convert()

