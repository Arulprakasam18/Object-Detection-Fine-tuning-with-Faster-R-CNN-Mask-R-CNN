import torch
from torchvision.transforms import functional as F
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def box_cxcywh_to_xyxy(x):
    """Convert bounding box format from center-size to corner coordinates."""
    b = x.clone()
    b[0] = x[0] - x[2] / 2  # xmin
    b[1] = x[1] - x[3] / 2  # ymin
    b[2] = x[0] + x[2] / 2  # xmax
    b[3] = x[1] + x[3] / 2  # ymax
    return b

def rescale_bboxes(out_bbox, size):
    """Rescale bounding boxes to match image size."""
    return box_cxcywh_to_xyxy(out_bbox) * torch.Tensor([size[0], size[1], size[0], size[1]])

def postprocess(outputs, size, score_threshold):
    """Process model outputs for evaluation."""
    results = []
    for output in outputs:
        # Convert boxes to xyxy format
        boxes_scaled = rescale_bboxes(output["boxes"], size)
        labels = output["labels"]
        scores = output["scores"]

        keep = scores >= score_threshold
        boxes_scaled = boxes_scaled[keep]
        labels = labels[keep]
        scores = scores[keep]

        results.append({
            "boxes": boxes_scaled,
            "labels": labels,
            "scores": scores
        })

    return results
