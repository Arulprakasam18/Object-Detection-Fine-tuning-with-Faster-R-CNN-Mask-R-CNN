import torch
import time
import torchvision
from torch.utils.data import DataLoader

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)

    return metric_logger

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        metric_logger.update(loss=losses, **loss_dict)

    print("Evaluation results:")
    print(f"Loss: {metric_logger.loss.global_avg}")
    return metric_logger

class MetricLogger:
    def __init__(self, delimiter=", "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()
            self.meters[name].update(value)

    def log_every(self, iterable, print_freq, header):
        i = 0
        start_time = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                elapsed_time = time.time() - start_time
                print(f"{header} {i}/{len(iterable)} - {elapsed_time:.2f}s")
                start_time = time.time()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
