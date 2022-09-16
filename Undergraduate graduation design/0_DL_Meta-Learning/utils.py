import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import copy

def reproduce(seed):
    # maybe cause some Graphics card erro
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = (device == 'cuda')
    torch.backends.cudnn.benchmark = (device == 'cuda')


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, output, target):
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        loss1 = reduce_loss(-log_preds.sum(dim=-1) / c, self.reduction)
        loss2 = F.nll_loss(log_preds, target, reduction=self.reduction)
        loss = (1. - self.alpha) * loss2 + self.alpha * loss1
        return loss

class EarlyStopping:
    def __init__(self, patience=10, decrease = True, verbose=False, delta=1e-8):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease = decrease

    def call_decrease(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def call_increase(self, val_acc, model, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0

    def __call__(self, val_loss, model, path):
        if self.decrease is True:
            return self.call_decrease(val_loss, model, path)
        else:
            return self.call_increase(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.decrease is True:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(f'Validation acc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

def mix_label(labels, pred, b=0.5, temp=1):
    # temp makes it sharpen?
    lam = np.random.beta(b, b)
    pred = logit_norm(pred)
    pseudo_labels = torch.nn.functional.softmax(pred / temp, dim=1)
    labels = torch.nn.functional.one_hot(labels.long())
    return (1-lam)*labels + lam*pseudo_labels

def mixup(beta, x, y, c_ind, n_ind, device, debug=False):
    lam = np.random.beta(beta, beta)
    x_clean, y_clean = x[c_ind], y[c_ind]

    x_noise, y_noise = x[n_ind], y[n_ind]
    if debug:
        print('x:', x_clean.shape)
        print('y:', y_clean.shape)
    if y_clean.shape[0] < y_noise.shape[0]:
        n = int(y_noise.shape[0] / y_clean.shape[0]) + 1
        xm_clean, ym_clean = x_clean.repeat(n,1,1,1), y_clean.repeat(n)
        if debug:
            print('n', n)
            print('xm:', xm_clean.shape)
            print('ym:', ym_clean.shape)
        rand_index = torch.randperm(ym_clean.shape[0])[:y_noise.shape[0]].to(device)
        if debug:
            print('ind:', rand_index.shape)
        xm_clean, ym_clean = xm_clean[rand_index], ym_clean[rand_index]
    else:
        rand_index = torch.randperm(y_clean.shape[0])[:y_noise.shape[0]].to(device)
        xm_clean, ym_clean = x_clean[rand_index], y_clean[rand_index]

    if debug:
        print('noise:', x_noise.shape)
        print('xm:', xm_clean.shape)
    x_noise = lam * x_noise + (1 - lam) * xm_clean
    return x_noise, y_noise, ym_clean, lam


def cutmix(beta, x, y, c_ind, n_ind, device):
    lam = np.random.beta(beta, beta)
    x_clean, y_clean = x[c_ind], y[c_ind]

    x_noise, y_noise = x[n_ind], y[n_ind]
    if y_clean.shape[0] < y_noise.shape[0]:
        n = int(y_noise.shape[0] / y_clean.shape[0]) + 1
        # print('n', n)
        xm_clean, ym_clean = x_clean.repeat(n,1,1,1), y_clean.repeat(n)
        rand_index = torch.randperm(ym_clean.shape[0])[:y_noise.shape[0]].to(device)
        # print('n', n)
        # print('1:',ym_clean.shape[0])
        # print('2:', y_noise.shape[0])
        # print('xm:', xm_clean.shape)
        # print('ym:', ym_clean.shape)
        xm_clean, ym_clean = xm_clean[rand_index], ym_clean[rand_index]
        # print('n', n)
        # print('xm:', xm_clean.shape)
        # print('ym:', ym_clean.shape)
    else:
        rand_index = torch.randperm(y_clean.shape[0])[:y_noise.shape[0]].to(device)
        xm_clean, ym_clean = x_clean[rand_index], y_clean[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x_noise.size(), lam)
    x_noise[:, :, bbx1:bbx2, bby1:bby2] = xm_clean[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_noise.size()[-1] * x_noise.size()[-2]))

    return x_noise, y_noise, ym_clean, lam


def rand_bbox(size, lam):
    # return the Clipping regional coordinates
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def logit_norm(v, use_logit_norm=True):
    if not use_logit_norm:
        return v
    return v * torch.rsqrt(torch.mean(torch.square(v)) + 1e-8)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_c = copy.deepcopy(tensor)
        for t, m, s in zip(tensor_c, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_c


unorm = UnNormalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261))

def tensor2Image(image_tensor, imtype=np.uint8):
    image_tensor = unorm(image_tensor)
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) )  * 255.0
    return image_numpy.astype(imtype)

def tensor2imgtensor(image_tensor, imtype=torch.uint8):
    image = unorm(image_tensor)
    image = image * 255.0
    return image.type(imtype)

def Randaugment(img):
    # normalize = torchvision.transforms.Normalize(
    #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    # )
    normalize = torchvision.transforms.Normalize(
        mean=[0.491, 0.482, 0.446],
        std=[0.247, 0.243, 0.261],
    )
    img = tensor2imgtensor(img)
    ra = torchvision.transforms.RandAugment()
    img = ra(img).type(torch.float32) / 255
    img = normalize(img)
    return img


class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, y = next(self.iterator)

        return x, y