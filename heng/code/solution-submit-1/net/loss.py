from common import *


class SoftCrossEntroyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntroyLoss, self).__init__()

    def forward(self, logits, soft_labels):
        #batch_size, num_classes =  logits.size()
        # soft_labels = labels.view(-1,num_classes)
        # logits      = logits.view(-1,num_classes)


        logits = logits - logits.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(logits), 1))
        loss = - (soft_labels*logits).sum(1) + log_sum_exp
        loss = loss.mean()

        return loss



# loss, accuracy -------------------------
def top_accuracy(probs, labels, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""

    probs  = probs.data
    labels = labels.data

    max_k = max(top_k)
    batch_size = labels.size(0)

    values, indices = probs.topk(max_k, dim=1, largest=True,  sorted=True)
    indices  = indices.t()
    corrects = indices.eq(labels.view(1, -1).expand_as(indices))

    accuracy = []
    for k in top_k:
        # https://stackoverflow.com/questions/509211/explain-slice-notation
        # a[:end]      # items from the beginning through end-1
        c = corrects[:k].view(-1).float().sum(0, keepdim=True)
        accuracy.append(c.mul_(1. / batch_size))
    return accuracy


## focal loss ## ---------------------------------------------------
class CrossEntroyLoss(nn.Module):
    def __init__(self):
        super(CrossEntroyLoss, self).__init__()

    def forward(self, logits, labels):
        #batch_size, num_classes =  logits.size()
        # labels = labels.view(-1,1)
        # logits = logits.view(-1,num_classes)

        max_logits  = logits.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(logits-max_logits), 1))
        loss = log_sum_exp - logits.gather(dim=1, index=labels.view(-1,1)).view(-1) + max_logits
        loss = loss.mean()

        return loss

## https://github.com/unsky/focal-loss
## https://github.com/sciencefans/Focal-Loss
## https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/39951

#  https://raberrytv.wordpress.com/2017/07/01/pytorch-kludges-to-ensure-numerical-stability/
#  https://github.com/pytorch/pytorch/issues/1620
class FocalLoss(nn.Module):
    def __init__(self,gamma = 2, alpha=1.2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, logits, labels):
        eps = 1e-7

        # loss =  - np.power(1 - p, gamma) * np.log(p))
        probs = F.softmax(logits)
        probs = probs.gather(dim=1, index=labels.view(-1,1)).view(-1)
        probs = torch.clamp(probs, min=eps, max=1-eps)

        loss = -torch.pow(1-probs, self.gamma) *torch.log(probs)
        loss = loss.mean()*self.alpha

        return loss




# https://arxiv.org/pdf/1511.05042.pdf
class TalyorCrossEntroyLoss(nn.Module):
    def __init__(self):
        super(TalyorCrossEntroyLoss, self).__init__()

    def forward(self, logits, labels):
        #batch_size, num_classes =  logits.size()
        # labels = labels.view(-1,1)
        # logits = logits.view(-1,num_classes)

        talyor_exp = 1 + logits + logits**2
        loss = talyor_exp.gather(dim=1, index=labels.view(-1,1)).view(-1) /talyor_exp.sum(dim=1)
        loss = loss.mean()

        return loss

# check #################################################################
def run_check_focal_loss():
    batch_size  = 64
    num_classes = 15

    logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
    labels = np.random.choice(num_classes,size=(batch_size))

    logits = Variable(torch.from_numpy(logits)).cuda()
    labels = Variable(torch.from_numpy(labels)).cuda()

    focal_loss = FocalLoss(gamma = 2)
    loss = focal_loss(logits, labels)
    print (loss)


def run_check_soft_cross_entropy_loss():
    batch_size  = 64
    num_classes = 15

    logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
    soft_labels = np.random.uniform(-2,2,size=(batch_size,num_classes))

    logits = Variable(torch.from_numpy(logits)).cuda()
    soft_labels = Variable(torch.from_numpy(soft_labels)).cuda()
    soft_labels = F.softmax(soft_labels,1)

    soft_cross_entropy_loss = SoftCrossEntroyLoss()
    loss = soft_cross_entropy_loss(logits, soft_labels)
    print (loss)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_soft_cross_entropy_loss()

    print('\nsucess!')