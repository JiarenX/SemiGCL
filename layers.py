import torch
from torch import nn
import torch.nn.functional as F


class AggregatorMixin():
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class DiffusionAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, dropout=0.5, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(DiffusionAggregator, self).__init__()

        self.fc_local = nn.Linear(2 * int(input_dim), output_dim, bias=False)
        self.fc_global = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
        self.dropout = dropout

    def forward(self, x, neibs, weights):
        agg_weight = weights.view(x.shape[0], -1)
        agg_neib = neibs.view(x.shape[0], -1, neibs.shape[1])
        if int(torch.sum(agg_weight[0, :]).item()) == agg_weight.shape[1]:
            agg_neib = agg_neib.mean(dim=1)
            out = self.fc_local(self.combine_fn([x, agg_neib]))
        else:
            agg_neib = torch.sum(agg_neib * agg_weight.unsqueeze(-1), dim=1)
            out = self.fc_global(agg_neib)
        out = F.dropout(out, self.dropout, training=self.training)
        if self.activation:
            out = self.activation(out)

        return out


aggregator_lookup = {
    "diffusion": DiffusionAggregator,
}
