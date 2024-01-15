from torch.distributions import Bernoulli, Binomial, Categorical, Dirichlet, Normal
from typing import Union, Optional, List
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LayerIdx:
    leaf_idx: torch.LongTensor
    sum_idx: torch.LongTensor
    prod_idx: torch.LongTensor


def safelog(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(torch.get_default_dtype()).tiny
    return torch.log(torch.clamp(x, min=eps))


def batched_logsumexp(log_prob: torch.Tensor, sum_param: torch.Tensor) -> torch.Tensor:
    # credits to github.com/loreloc
    # log_prob  (batch_size, n_features, hidden_dim), this is in log  domain
    # sum_param (n_features, hidden_dim, hidden_dim), this is in prob domain
    max_log_prob = torch.max(log_prob, dim=-1, keepdim=True).values
    norm_exp_log_prob = torch.exp(log_prob - max_log_prob)
    log_prob = max_log_prob + safelog(norm_exp_log_prob.transpose(0, 1) @ sum_param.transpose(1, 2)).transpose(0, 1)
    return log_prob


class DLTM(torch.nn.Module):
    """ A Discrete Latent Tree Model modelled as a tensorized Probabilistic Circuit """

    def __init__(
        self,
        tree: Union[List, np.array],
        leaf_type: str,
        hidden_dim: Optional[int] = 16,
        n_categories: Optional[int] = None,
        norm_weight: Optional[bool] = True,
        em: Optional[bool] = False,
        learnable: Optional[bool] = True,
        min_std: Optional[float] = 1e-3,
        max_std: Optional[float] = 7.0,
    ):
        super().__init__()
        assert leaf_type in ['bernoulli', 'categorical', 'gaussian', 'binomial'], 'Leaf type not implemented!'
        self.tree = np.array(tree)  # List of predecessors: tree[i] = j if j is parent of i
        self.root = np.argwhere(self.tree == -1).item()  # tree[i] = -1 if i is root
        self.n_features = n_features = len(self.tree)
        self.features = np.arange(n_features)
        self.leaf_type = leaf_type
        self.n_categories = n_categories  # if categorical (binomial), it specifies the number of categories (trials)
        self.norm_weight = norm_weight  # True to perform log softmax in self.sum_param
        self.min_std = min_std
        self.max_std = max_std
        self.em = em  # True if training under EM, False otherwise
        self._build_structure()

        # initialize sum logits with dirichlet allocation
        sum_logits = Dirichlet(torch.ones(hidden_dim)).sample([n_features, hidden_dim]).log()
        if self.em: sum_logits.exp_()
        self.sum_logits = torch.nn.Parameter(sum_logits) if learnable else sum_logits

        if self.leaf_type == 'bernoulli':
            leaf_logits = torch.rand(n_features, hidden_dim)
            if not self.em: leaf_logits.logit_()
        elif self.leaf_type == 'binomial':
            leaf_logits = torch.rand(n_features, hidden_dim)
            if self.em: raise NotImplementedError('EM not implemented for binomials')
        elif self.leaf_type == 'categorical':
            leaf_logits = torch.randn(n_features, hidden_dim, n_categories)
            if self.em: leaf_logits = leaf_logits.softmax(-1)
        elif self.leaf_type == 'gaussian':
            leaf_logits = torch.empty(n_features, 2, hidden_dim).normal_(std=0.05)
            if self.em: leaf_logits[:, 1] = 0.5 + leaf_logits[:, 1].tanh() * 0.1
        self.leaf_logits = torch.nn.Parameter(leaf_logits) if learnable else leaf_logits

    def _build_structure(self):

        self.bfs = {0: [[self.root], [-1]]}
        depths = np.array([0 if node == self.root else None for node in range(self.n_features)])
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            children = list(np.argwhere(self.tree == node)[:, 0])
            depths[children] = depths[node] + 1
            if len(children):
                queue.extend(children)
                self.bfs.setdefault(depths[node] + 1, [[], []])[0].extend(children)
                self.bfs[depths[node] + 1][1].extend([node] * len(children))

        self.post_order = {}
        last_node_visited = None
        layer_idx = np.full(self.n_features, None)
        stack = [self.root]
        while stack:
            children = list(np.argwhere(self.tree == stack[-1])[:, 0])
            if len(children) == 0 or last_node_visited in children:
                layer_idx[stack[-1]] = 0 if len(children) == 0 else 1 + max(layer_idx[children])
                self.post_order.setdefault(layer_idx[stack[-1]], {})[stack[-1]] = children
                last_node_visited = stack.pop()
            else:
                stack.extend(children)

        self.layers = []
        for layer_id in self.post_order:
            in_len = torch.LongTensor(list(map(len, self.post_order[layer_id].values())))
            self.layers.append(LayerIdx(
                leaf_idx=torch.LongTensor(list(self.post_order[layer_id].keys())),
                sum_idx=torch.LongTensor(sum(self.post_order[layer_id].values(), [])),  # try to get this (monoid) :)
                prod_idx=torch.arange(len(in_len)).repeat_interleave(in_len)))

    @property
    def hidden_dim(self):
        return self.sum_logits.size(-1)

    @property
    def n_param(self):
        return (self.n_features - 1) * self.hidden_dim ** 2 + self.hidden_dim + self.leaf_logits.nelement()

    @property
    def leaf_param(self):
        # returns leaf parameters in the prob domain
        # when em is True, leaf_logits already corresponds to valid leaf params
        if self.em and self.leaf_type == 'gaussian':
            scale = self.leaf_logits[:, 1:].clamp(min=self.min_std, max=self.max_std)
            return torch.cat([self.leaf_logits[:, :1], scale], dim=1)
        elif self.em:
            return self.leaf_logits
        elif self.leaf_type == 'bernoulli':
            return self.leaf_logits.sigmoid()
        elif self.leaf_type == 'binomial':
            return self.leaf_logits.sigmoid()
        elif self.leaf_type == 'categorical':
            return self.leaf_logits.softmax(dim=2)
        elif self.leaf_type == 'gaussian':
            scale = (torch.nn.functional.silu(self.leaf_logits[:, 1:]) + 0.279).clamp(min=self.min_std, max=self.max_std)
            return torch.cat([self.leaf_logits[:, :1], scale], dim=1)

    @property
    def sum_param(self):
        # returns sum parameters in the prob domain
        if self.em:
            return self.sum_logits
        else:
            return self.sum_logits.softmax(dim=-1) if self.norm_weight else self.sum_logits

    @property
    def log_norm_constant(self):
        x = torch.zeros(1, self.n_features).to(device=self.sum_logits.device)
        return self.forward(x, has_nan=x == 0).squeeze()

    def leaf_log_prob(
        self,
        x: torch.Tensor,
        has_nan: Optional[Union[bool, torch.Tensor]] = None,
    ):
        assert x.ndim == 2 and x.size(1) == self.n_features
        leaf_param = self.leaf_param
        if self.leaf_type == 'bernoulli':
            leaf_log_prob = Bernoulli(leaf_param, validate_args=False).log_prob(x.unsqueeze(2))
        elif self.leaf_type == 'binomial':
            leaf_log_prob = Binomial(self.n_categories - 1, leaf_param, validate_args=False).log_prob(x.unsqueeze(2))
        elif self.leaf_type == 'categorical':
            index = x if x.dtype == torch.long else x.long()
            leaf_log_prob = leaf_param.log().transpose(1, 2)[range(self.n_features), index]
        elif self.leaf_type == 'gaussian':
            leaf_log_prob = Normal(leaf_param[:, 0], leaf_param[:, 1], validate_args=False).log_prob(x.unsqueeze(2))
        else:
            raise NotImplementedError('Leaf type not implemented.')
        # Handle potential marginalisation
        if isinstance(has_nan, torch.Tensor):
            return leaf_log_prob.masked_fill(has_nan.unsqueeze(2), 0)
        elif has_nan is True or has_nan is None:
            leaf_log_prob.masked_fill(x.isnan().unsqueeze(2), 0)
        else:
            return leaf_log_prob

    def forward(
        self,
        x: torch.Tensor,
        has_nan: Optional[Union[bool, torch.Tensor]] = None,
        normalize: Optional[bool] = False,
        return_lls: Optional[bool] = False,
        return_prod_lls: Optional[bool] = False
    ):
        leaf_log_prob = self.leaf_log_prob(x, has_nan=has_nan)  # (batch_size, n_features, hidden_dim)
        lls = {'leaf': leaf_log_prob, 'sum': torch.zeros_like(leaf_log_prob)}
        if return_prod_lls: lls['prod'] = torch.zeros_like(leaf_log_prob)
        sum_param = self.sum_param  # not a useless instruction, it may avoid the softmax at every layer iteration
        for layer in self.layers:
            prod = torch.index_add(
                source=lls['sum'][:, layer.sum_idx], dim=1, index=layer.prod_idx.to(x.device),
                input=lls['leaf'][:, layer.leaf_idx])
            lls['sum'][:, layer.leaf_idx] = batched_logsumexp(prod, sum_param[layer.leaf_idx])
            if return_prod_lls: lls['prod'][:, layer.leaf_idx] = prod
        root_log_prob = lls['sum'][:, self.layers[-1].leaf_idx, 0] - (self.log_norm_constant if normalize else 0)
        return (root_log_prob, lls) if return_lls else root_log_prob

    @torch.no_grad()
    def backward(
        self,
        n_samples: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
        mpe: Optional[bool] = False,
        mpe_leaf: Optional[bool] = False
    ):
        def sample_or_mode(dist: torch.distributions.Distribution, mode: bool):
            return dist.mode if mode else dist.sample()

        if x is not None:
            # conditional backward
            assert n_samples is None
            prod_prob = self.forward(x, return_lls=True, return_prod_lls=True)[1]['prod'].exp()
        else:
            # unconditional backward
            prod_prob = torch.ones(n_samples, self.n_features, self.hidden_dim, device=self.sum_logits.device)

        sum_param = self.sum_param
        sum_states = torch.full((len(prod_prob), self.n_features), -1, device=self.sum_logits.device, dtype=torch.long)
        sum_states[:, self.bfs[0][0]] = sample_or_mode(
            Categorical(probs=sum_param[self.bfs[0][0], 0] * prod_prob[:, self.bfs[0][0]]), mode=mpe)
        for depth in range(1, len(self.bfs)):
            children, parents = self.bfs[depth]
            sum_states[:, children] = sample_or_mode(
                Categorical(probs=sum_param[children, sum_states[:, parents]] * prod_prob[:, children]), mode=mpe)

        if self.leaf_type == 'bernoulli':
            samples = sample_or_mode(Bernoulli(self.leaf_param[self.features, sum_states]), mode=mpe or mpe_leaf)
        elif self.leaf_type == 'categorical':
            samples = sample_or_mode(Categorical(self.leaf_param[self.features, sum_states]), mode=mpe or mpe_leaf)
        elif self.leaf_type == 'gaussian':
            loc, scale = self.leaf_param.chunk(2, dim=1)
            samples = sample_or_mode(
                Normal(loc[self.features, 0, sum_states], scale[self.features, 0, sum_states]), mode=mpe or mpe_leaf)
        else:
            raise NotImplementedError('Sampling not implemented for %s leaves' % self.leaf_type)

        if x is not None:
            nan_mask = x.isnan()
            samples = x.masked_fill(nan_mask, 0) + nan_mask * samples
        return samples

    def em_step(
        self,
        x: torch.Tensor,
        step_size: float,
        n_chunks: Optional[int] = 1,  # chunked computation to avoid OOM
        alpha: Optional[float] = 1e-5
    ):
        root_log_prob, lls = self.forward(x, has_nan=False, return_lls=True)
        leaf_grad, sum_grad = torch.autograd.grad(root_log_prob.sum(), (lls['leaf'], self.sum_logits))

        unnorm_sum_logits = self.sum_logits * sum_grad + alpha
        sum_logits = unnorm_sum_logits / unnorm_sum_logits.sum(-1, keepdim=True)
        self.sum_logits.data = (1.0 - step_size) * self.sum_logits + step_size * sum_logits

        if self.leaf_type == 'bernoulli':
            leaf_logits = ((x.unsqueeze(dim=-1) * leaf_grad).sum(dim=0) + alpha) / (leaf_grad.sum(dim=0) + 2 * alpha)
            self.leaf_logits.data = (1.0 - step_size) * self.leaf_logits + step_size * leaf_logits
        elif self.leaf_type == 'categorical':
            cat_mask = torch.eq(x.unsqueeze(1), torch.arange(self.n_categories).to(x.device).view(1, -1, 1)).float()
            leaf_logits = []
            for leaf_grad_chunk, cat_mask_chunk in zip(leaf_grad.chunk(n_chunks), cat_mask.chunk(n_chunks)):
                leaf_logits.append(
                    torch.einsum('ijkl, ijkm -> jkl', leaf_grad_chunk.unsqueeze(1), cat_mask_chunk.unsqueeze(-1)))
            leaf_logits = torch.stack(leaf_logits).sum(0).permute(1, 2, 0) + alpha
            leaf_logits = leaf_logits / (leaf_grad.sum(dim=0).unsqueeze(-1) + self.n_categories * alpha)
            self.leaf_logits.data = (1.0 - step_size) * self.leaf_logits + step_size * leaf_logits
        elif self.leaf_type == 'gaussian':
            sum_leaf_grad = leaf_grad.sum(dim=0) + alpha
            loc = (leaf_grad * x.unsqueeze(dim=-1)).sum(dim=0) / sum_leaf_grad
            scale = ((leaf_grad * (x.unsqueeze(dim=-1) - loc) ** 2.0).sum(dim=0) / sum_leaf_grad).sqrt().clamp(min=1e-5)
            self.leaf_logits.data[:, 0] = (1.0 - step_size) * self.leaf_logits[:, 0] + step_size * loc
            self.leaf_logits.data[:, 1] = (1.0 - step_size) * self.leaf_logits[:, 1] + step_size * scale
        else:
            raise NotImplementedError('EM step not implemented for %s leaves' % self.leaf_type)

        return root_log_prob
