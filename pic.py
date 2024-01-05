from typing import Optional, List
import torch.nn as nn
import numpy as np
import torch


def zw_quadrature(
    mode: str,
    nip: int,
    a: Optional[float] = -1,
    b: Optional[float] = 1,
    log_weight: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[str] = 'cpu'
):
    if mode == 'leggauss':
        z, w = np.polynomial.legendre.leggauss(nip)
        z = (b - a) * (z + 1) / 2 + a
        w = w * (b - a) / 2
    elif mode == 'midpoint':
        z = np.linspace(a, b, num=nip + 1)
        z = (z[:-1] + z[1:]) / 2
        w = np.full_like(z, (b - a) / nip)
    elif mode == 'trapezoidal':
        z = np.linspace(a, b, num=nip)
        w = np.full((nip,), (b - a) / (nip - 1))
        w[0] = w[-1] = 0.5 * (b - a) / (nip - 1)
    elif mode == 'simpson':
        assert nip % 2 == 1, 'Number of integration points must be odd'
        z = np.linspace(a, b, num=nip)
        w = np.concatenate([np.ones(1), np.tile(np.array([4, 2]), nip // 2 - 1), np.array([4, 1])])
        w = ((b - a) / (nip - 1)) / 3 * w
    elif mode == 'hermgauss':
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
        z, w = np.polynomial.hermite.hermgauss(nip)
    else:
        raise NotImplementedError('Integration mode not implemented.')
    z, w = torch.tensor(z, dtype=dtype), torch.tensor(w, dtype=dtype)
    w = w.log() if log_weight else w
    return z.to(device), w.to(device)


def build_sum_mask(tree, leaf_type):
    sum_mask = np.array([True] * len(tree))
    if leaf_type in ['bernoulli', 'categorical']:
        last_node_visited = None
        stack = [np.argwhere(tree == -1).item()]
        while stack:
            children = list(np.argwhere(tree == stack[-1])[:, 0])
            if len(children) == 0 or last_node_visited in children:
                sum_mask[stack[-1]] = False if len(children) == 0 else True
                last_node_visited = stack.pop()
            else:
                stack.extend(children)
    sum_mask[np.argwhere(tree == -1).item()] = False
    return sum_mask


class FourierLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: Optional[float] = 1.0,
        flatten01: Optional[bool] = False
    ):
        super(FourierLayer, self).__init__()
        assert out_features % 2 == 0, 'Number of output features must be even.'
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.flatten01 = flatten01
        self.register_buffer('coeff', torch.normal(0.0, sigma, (in_features, out_features // 2)))

    def forward(self, x: torch.Tensor):
        x_proj = 2 * torch.pi * x @ self.coeff
        x_ff = torch.cat([x_proj.cos(), x_proj.sin()], dim=-1).transpose(-2, -1)
        return x_ff.flatten(0, 1) if self.flatten01 else x_ff

    def extra_repr(self) -> str:
        return '{}, {}, sigma={}'.format(self.in_features, self.out_features, self.sigma)


class PIC(nn.Module):

    def __init__(
        self,
        tree: List[int],
        leaf_type: str,
        n_units: Optional[int] = 64,
        sigma: Optional[float] = 1.0,
        n_categories: Optional[int] = None
    ):
        super().__init__()
        leaf_type_dict = {'bernoulli': 1, 'binomial': 1, 'categorical': n_categories, 'gaussian': 2}
        assert leaf_type in leaf_type_dict.keys()

        self.n_features = n_features = len(tree)
        self.tree = tree  # list of predecessor
        self.root = np.argwhere(tree == -1).item()
        self.sum_mask = build_sum_mask(tree, leaf_type)  # if sum_mask[i] is True, then i is a non-root sum region
        self.leaf_mask = ~ self.sum_mask.copy()  # if leaf_mask[i] is True, then i is a leaf of the tree
        self.leaf_mask[self.root] = False
        self.n_sum_nets = sum(self.sum_mask)
        self.leaf_type = leaf_type  # if leaf_type is categorical, this specifies the number of categories
        self.n_categories = n_categories
        self.n_units = n_units

        self.root_net = nn.Sequential(
            FourierLayer(1, n_units, sigma),
            nn.Conv1d(n_units, n_units, 1),
            nn.Tanh(),
            nn.Conv1d(n_units, 1, 1),
            nn.Softplus())
        self.sum_nets = nn.Sequential(
            FourierLayer(2, n_units, sigma),
            nn.Conv1d(self.n_sum_nets * n_units, self.n_sum_nets * n_units, 1, groups=self.n_sum_nets),
            nn.Tanh(),
            nn.Conv1d(self.n_sum_nets * n_units, self.n_sum_nets, 1, groups=self.n_sum_nets),
            nn.Softplus())
        self.leaf_nets = nn.Sequential(
            FourierLayer(1, n_units, sigma),
            nn.Conv1d(n_features * n_units, n_features * n_units, 1, groups=n_features),
            nn.SiLU() if leaf_type == 'gaussian' else nn.Tanh(),
            nn.Conv1d(n_features * n_units, n_features * leaf_type_dict[self.leaf_type], 1, groups=n_features))

    def forward(
        self,
        z: torch.Tensor,
        log_w: Optional[torch.Tensor] = None,
        n_chunks: Optional[int] = 1
    ):
        assert z.ndim == 1 or (z.ndim == 2 and z.size(0) == self.n_features)
        nip = z.size(0) if z.ndim == 1 else z.size(1)  # number of integration points

        if z.ndim == 1:
            self.sum_nets[0].flatten01 = self.leaf_nets[0].flatten01 = False
            self.sum_nets[1].groups = self.leaf_nets[1].groups = 1
            z1d = z1d_root = z.unsqueeze(1)
            z2d = torch.stack([z.repeat_interleave(nip), z.repeat(nip)]).t()
        else:
            self.sum_nets[0].flatten01 = self.leaf_nets[0].flatten01 = True
            self.sum_nets[1].groups, self.leaf_nets[1].groups = self.n_sum_nets, self.n_features
            if self.leaf_type in ['bernoulli', 'categorical']: z.data[self.leaf_mask] = z.data[self.tree[self.leaf_mask]]
            z1d_root = z[self.root].unsqueeze(1)
            z1d = z.unsqueeze(2)
            z2d = torch.stack([z[self.tree[self.sum_mask]].repeat_interleave(nip, 1), z[self.sum_mask].repeat(1, nip)], 2)

        sum_logits = torch.eye(nip, device=z.device).log().unsqueeze(0).repeat(self.n_features, 1, 1)
        sum_logits[self.root] = - self.root_net(z1d_root).unsqueeze(0).expand(-1, nip, -1)
        sum_logits[self.sum_mask] = - torch.hstack(
            [self.sum_nets(chunk) for chunk in z2d.chunk(n_chunks, 1)]).view(-1, nip, nip)
        if log_w is not None:
            sum_logits = (sum_logits - (sum_logits + log_w).logsumexp(-1, True)) + log_w
        sum_logits = sum_logits.exp()

        # for the bernoulli/binomial case, no reshaping is needed
        leaf_logits = self.leaf_nets(z1d)
        if self.leaf_type == 'categorical':
            leaf_logits = leaf_logits.view(self.n_features, self.n_categories, -1).transpose(1, 2)
        elif self.leaf_type == 'gaussian':  # stack loc and log_scale
            leaf_logits = torch.stack([leaf_logits[0::2], leaf_logits[1::2]], dim=1)

        return sum_logits, leaf_logits
