# Document problem-solution examples. This is for my personal reference

## Neural Networks

DETR - class loss is constants, the rest improve

*Tried*

- toggle loss model args `{"cls_loss_coef": 1, "dst_loss_coef": 1, "bbs_loss_coef": 1, "cls_bg_wt": 1.0}`
- toggle n_queries

> 2 gives nan, 5 doesn't improve,

*Solution*

Play around with loss model args, separate lr for featurizer and rest of model
