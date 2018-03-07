from common import*
from dataset.transform  import*

#https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
image_to_tensor_transform = pytorch_image_to_tensor_transform



###  modules  ###-----------------------------
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#     def forward(self,x):
#         return x


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

        self.is_bn = False
        self.bn = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat


    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1):
        super(ConvBn2d, self).__init__()
        self.is_bn = True

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        if self.is_bn :
            x = self.bn(x)

        return x



class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(Bottleneck, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes,     planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   planes,     planes, kernel_size=3, padding=1, stride=stride)
        self.conv_bn3 = ConvBn2d(   planes, out_planes, kernel_size=1, padding=0, stride=1)

        if self.is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):

        z = self.conv_bn1(x)
        z = F.relu(z,inplace=True)
        z = self.conv_bn2(z)
        z = F.relu(z,inplace=True)
        z = self.conv_bn3(z)

        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x

        z = F.relu(z,inplace=True)
        return z


# layers ##---------------------------------------------------------------------

def make_layer(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(Bottleneck(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(Bottleneck(out_planes, planes, out_planes))

    return nn.Sequential(*layers)

def make_layer0(in_channels, out_planes):
    layers = [
        ConvBn2d(in_channels, out_planes, kernel_size=7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]
    return nn.Sequential(*layers)



###  network ###-----------------------------
class ResNet101(nn.Module):

    # def load_pretrain_file(self,pretrain_file, skip=[]):
    #
    #     pretrain_state_dict = torch.load(pretrain_file)
    #     state_dict = self.state_dict()
    #
    #     keys = list(state_dict.keys())
    #     for key in keys:
    #         if any(s in key for s in skip):
    #             continue
    #
    #         pretrain_key = key
    #         if 'layer0.0.conv.'   in key: pretrain_key=key.replace('layer0.0.conv.',  'conv1.' )
    #         if 'layer0.0.bn.'     in key: pretrain_key=key.replace('layer0.0.bn.',    'bn1.'   )
    #         if '.conv_bn1.conv.'  in key: pretrain_key=key.replace('.conv_bn1.conv.', '.conv1.')
    #         if '.conv_bn1.bn.'    in key: pretrain_key=key.replace('.conv_bn1.bn.',   '.bn1.'  )
    #         if '.conv_bn2.conv.'  in key: pretrain_key=key.replace('.conv_bn2.conv.', '.conv2.')
    #         if '.conv_bn2.bn.'    in key: pretrain_key=key.replace('.conv_bn2.bn.',   '.bn2.'  )
    #         if '.conv_bn3.conv.'  in key: pretrain_key=key.replace('.conv_bn3.conv.', '.conv3.')
    #         if '.conv_bn3.bn.'    in key: pretrain_key=key.replace('.conv_bn3.bn.',   '.bn3.'  )
    #         if '.downsample.conv.'in key: pretrain_key=key.replace('.downsample.conv.',  '.downsample.0.')
    #         if '.downsample.bn.'  in key: pretrain_key=key.replace('.downsample.bn.',    '.downsample.1.')
    #
    #         #print('%36s'%key, ' ','%-36s'%pretrain_key)
    #         state_dict[key] = pretrain_state_dict[pretrain_key]
    #
    #     self.load_state_dict(state_dict)
    #     #torch.save(state_dict,save_model_file)


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
    def __init__(self, in_shape=(3,180,180), num_classes=5270,  type = 'logits' ):

        super(ResNet101, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes=num_classes
        self.type = type

        self.layer0 = make_layer0(in_channels, 64)
        self.layer1 = make_layer(   64,  64,  256, num_blocks= 3, stride=1)  #out =  64*4 =  256
        self.layer2 = make_layer(  256, 128,  512, num_blocks= 4, stride=2)  #out = 128*4 =  512
        self.layer3 = make_layer(  512, 256, 1024, num_blocks=23, stride=2)  #out = 256*4 = 1024
        self.layer4 = make_layer( 1024, 512, 2048, num_blocks= 3, stride=2)  #out = 512*4 = 2048
        self.fc  = nn.Linear(2048, num_classes)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        #x                   #; print('input ' ,x.size())
        x = self.layer0(x)  #; print('layer0 ',x.size())
        x = self.layer1(x)  #; print('layer1 ',x.size())
        x = self.layer2(x)  #; print('layer2 ',x.size())
        x = self.layer3(x)  #; print('layer3 ',x.size())
        x = self.layer4(x)  #; print('layer4 ',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)

        if self.type == 'last_feature':
            return x

        if self.type == 'logits':
            x = self.fc (x)  #logits
            return   x



########################################################################################################


def run_check_net_imagenet():


    net = ResNet101(in_shape=(3,224,224), num_classes=1000)
    net.load_pretrain_file(
            '/root/share/data/models/reference/imagenet/resnet/resnet101-5d3b4d8f.pth',
            skip=[]
        )
    #net.cuda()
    net.eval()


    synset=[]
    for i in range(10001): synset.append('nil')

    if 1: # https://github.com/ruotianluo/pytorch-resnet/blob/master/synset.py
        synset[441] = 'n02823750 beer glass'
        synset[  1] = 'n01443537 goldfish, Carassius auratus'
        synset[ 22] = 'n01614925 bald eagle, American eagle, Haliaeetus leucocephalus'
        synset[  9] = 'n01518878 ostrich, Struthio camelus'
        synset[281] = 'n02123045 tabby, tabby cat'

    if 0: # https://github.com/soeaver/caffe-model/blob/master/cls/synset.txt
        synset[810] = 'n02823750 beer glass'
        synset[449] = 'n01443537 goldfish, Carassius auratus '
        synset[397] = 'n01614925 bald eagle, American eagle, Haliaeetus leucocephalus'
        synset[384] = 'n01518878 ostrich, Struthio camelus'
        synset[173] = 'n02123045 tabby, tabby cat'


    names = ['beer_glass', 'goldfish', 'blad_eagle', 'ostrich', 'tabby_cat', 'bullet_train', 'great_white_shape', 'pickup']
    for name in names:
        image_file = '/root/share/data/imagenet/dummy/256x256/' + name + '.jpg'
        image = cv2.imread(image_file)
        image = cv2.resize(image,(224,224))

        #pre process ----
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224)).astype(np.float32)
        image = image.transpose((2,0,1))
        image = image/255

        mean = [0.485, 0.456, 0.406 ]
        std  = [0.229, 0.224, 0.225 ]
        image[0] = (image[0] - mean[0]) / std[0]
        image[1] = (image[1] - mean[1]) / std[1]
        image[2] = (image[2] - mean[2]) / std[2]
        #pre process ----


        #run net
        logits = net( Variable(torch.from_numpy(image).unsqueeze(0).float() ) )
        probs  = F.softmax(logits,dim=1).data.numpy().reshape(-1)
        i = np.argmax(prob)

        print(' ** %-s *******************'%name)
        #print(prob)
        print('\t','results %5d, %05f, %05f, %s'%(i, logit[i], prob[i],synset[i-1])) # 0 is for background in mobilenet
        print('\t','')


def run_check_net_cdiscount():

    checkpoint = '/root/share/project/kaggle/cdiscount/model/resnet101-160-08b/checkpoint/00280000_model.pth'
    net = ResNet101(in_shape=(3,180,180), num_classes=5270)
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
        image = image[10:170,10:170]  ##crop

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


    net = ResNet101(in_shape=(C,H,W), num_classes=num_classes)
    # net.load_pretrain_file(
    #         '/root/share/data/models/reference/imagenet/resnet/resnet101-5d3b4d8f.pth',
    #         skip=['fc.']
    #     )
    net.cuda()
    net.train()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    logits = net.forward(x)
    probs  = F.softmax(logits,dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)

    print('probs')
    print(probs)

    #merging ----
    # net.eval()
    # net.merge_bn()



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_net()
    #run_check_net_imagenet()
    run_check_net_cdiscount()
