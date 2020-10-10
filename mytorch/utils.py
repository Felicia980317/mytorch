import torch
import cv2


def save(path, model):
    torch.save(model.state_dict(), path)


def load(path, model):
    model.load_state_dict(torch.load(path))


def getDevice(cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # if cuda:
    #     print("Device:")
    #     for i in range(torch.cuda.device_count()):
    #         print("    {}:".format(i), torch.cuda.get_device_name(i))
    # else:
    #     print("Device: CPU")
    return device
