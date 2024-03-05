import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grads):
        dx = ctx.lambda_ * grads.neg()

        return dx, None


def uniform_neib_sampler(diff_idx, diff_val, ids, n_samples, device='cpu'):
    if int(torch.sum(diff_val[0, :]).item()) == diff_val.shape[1]:
        n_samples = int(n_samples/4)
    tmp_idx = diff_idx[ids]
    tmp_val = diff_val[ids]
    perm = torch.randperm(tmp_idx.shape[1]).to(device)
    tmp_idx = tmp_idx[:, perm]
    tmp_val = tmp_val[:, perm]

    return tmp_idx[:, :n_samples], tmp_val[:, :n_samples]


class GraphSAGE(nn.Module):
    def __init__(self, aggregator_class, input_dim, layer_specs, device='cpu'):
        super(GraphSAGE, self).__init__()
        self.sample_fns = [partial(uniform_neib_sampler, n_samples=s['n_sample'], device=device) for s in layer_specs]
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(input_dim=input_dim, output_dim=spec['output_dim'], activation=spec['activation'])
            agg_layers.append(agg)
            input_dim = int(agg.output_dim/2)

        self.agg_layers = nn.Sequential(*agg_layers)

    def forward(self, ids, diff_idx, diff_val, feats):
        tmp_feats = feats[ids]
        all_feats = [tmp_feats]
        all_vals = []
        for _, sampler_fn in enumerate(self.sample_fns):
            ids, vals = sampler_fn(diff_idx=diff_idx, diff_val=diff_val, ids=ids)
            ids = ids.contiguous().view(-1).long()
            vals = vals.contiguous().view(-1)
            tmp_feats = feats[ids]
            all_feats.append(tmp_feats)
            all_vals.append(vals)
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1], all_vals[k]) for k in range(len(all_feats) - 1)]
        assert len(all_feats) == 1, "len(all_feats) != 1"
        out = all_feats[0]

        return out


class Predictor(nn.Module):
    def __init__(self, num_class=5, inc=128, temp=0.05):
        super(Predictor, self).__init__()

        self.fc = nn.Linear(inc, num_class, bias=False)
        self.temp = temp

    def forward(self, x, reverse=False, lamda=0.1):
        if reverse:
            x = GradientReversalFunction.apply(x, lamda)
        x = F.normalize(x, dim=1)
        x_out = self.fc(x) / self.temp

        return x_out
