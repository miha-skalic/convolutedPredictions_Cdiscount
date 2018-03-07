from common import*
from dataset.transform import*


class FcNet0(nn.Module):
    def __init__(self, in_shape=1000, num_classes=5270 ):
        super(FcNet0, self).__init__()
        self.num_classes = num_classes
        in_channels = in_shape

        self.linear1 = nn.Linear(in_channels, 7168)
        self.relu1   = nn.PReLU()
        self.linear2 = nn.Linear(7168, 4096)
        self.relu2   = nn.PReLU()
        self.fc = nn.Linear(4096, num_classes)


    def forward(self, x):
                             #; print('input ' ,x.size())
        N,C = x.size()
        x = self.linear1(x)  #; print('linear1 ',x.size())
        x = self.relu1  (x)
        #x = F.dropout(x, p=0.2, training=self.training)

        x = self.linear2(x)  #; print('linear2 ',x.size())
        x = self.relu2  (x)
        #x=F.dropout(x,p=0.2,training=self.training)

        x = self.fc(x)
        return x  #logits



def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 2
    num_classes = 5270
    V = 4
    C = 2048

    labels = torch.randn(batch_size,num_classes)
    inputs = torch.randn(batch_size,V,C)
    inputs = torch.abs(inputs)

    net = FcNet0(in_shape=C, num_classes=num_classes)
    net.cuda()
    net.train()

    y = Variable(labels).cuda()
    x = Variable(inputs).cuda()
    x = F.max_pool1d(x.permute(0, 2, 1), kernel_size=V).squeeze(2)
    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)
    print('probs')
    print(probs)




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()


