def get_miou(conf_matrix):
    inter = torch.diag(conf_matrix)

    true = conf_matrix.sum(dim=1)
    pred = conf_matrix.sum(dim=0)

    union = (true + pred) - inter

    iou_per_class = inter / (union + 1e-6)
    miou = torch.mean(iou_per_class)
    return miou, iou_per_class
