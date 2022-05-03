import torch

def dice_loss(input, target, mask, reduce=True):
    batch_size = input.size(0)
    input = torch.sigmoid(input)

    input = input.contiguous().view(batch_size, -1)
    target = target.contiguous().view(batch_size, -1).float()
    mask = mask.contiguous().view(batch_size, -1).float()

    # weight = weight.view(weight.size()[0], -1)
    # input = input * weight * mask

    input = input * mask
    target = target * mask

    # input_mean = torch.mean(input, dim=1)
    # target_mean = torch.mean(target, dim=1)

    # for i in range(input.size(0)):
    #     input[i, :] = input[i, :] - input_mean[i] + 1
    #     target[i, :] = target[i, :] - target_mean[i] + 1

    # input = input * input / 4.0
    # target = target * target / 4.0

    a = torch.sum(input * target, dim=1)
    b = torch.sum(input * input, dim=1) + 0.001
    c = torch.sum(target * target, dim=1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    if reduce:
        loss = torch.mean(loss)
    return loss