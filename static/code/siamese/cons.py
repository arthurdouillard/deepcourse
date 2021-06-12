def contrastive_loss(pred_simi, gt_simi, margin=0.20):
  pos_pair =  gt_simi * (1 - pred_simi)
  neg_pair = (1 - gt_simi) * torch.clamp(pred_simi - margin, min=0.)

  pos_pair = pos_pair ** 2
  neg_pair = neg_pair ** 2

  # Note that `gt_simi` acts here as a gate to avoid doing a for-loop and if/else
  return torch.mean(pos_pair + neg_pair)
