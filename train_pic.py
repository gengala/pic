from pic import PIC, zw_quadrature
from data import datasets, trees
from clt import learn_clt
from dltm import DLTM

from sklearn.model_selection import train_test_split
import numpy as np
import functools
import argparse
import torch
import json
import time
import os

print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('-device',                  type=str,   default='cuda',         help='cpu | cuda')
parser.add_argument('-ds',  '--dataset',        type=str,   default='mnist',        help='dataset name')
parser.add_argument('-split',                   type=str,   default=None,           help='dataset split for EMNIST')
parser.add_argument('-vsp',                     type=float, default=0.05,           help='MNIST valid split percentage')
parser.add_argument('-nip',                     type=int,   default=64,             help='number of integration points')
parser.add_argument('-int', '--integration',    type=str,   default='trapezoidal',  help='integration mode')
parser.add_argument('-lt',  '--leaf_type',      type=str,   default=None,           help='leaf distribution type')
parser.add_argument('-nu',  '--n_units',        type=int,   default=64,             help='pic neural net unit num.')
parser.add_argument('-sigma',                   type=float, default=1.0,            help='sigma ff')
parser.add_argument('-bs',  '--batch_size',     type=int,   default=256,            help='batch size during training')
parser.add_argument('-bs2', '--batch_size2',    type=int,   default=1024,           help='batch size during valid/test')
parser.add_argument('-as',  '--accum_steps',    type=int,   default=1,              help='number of accumulation steps')
parser.add_argument('-nc',  '--n_chunks',       type=int,   default=1,              help='num. of chunks to avoid OOM')
parser.add_argument('-ts',  '--train_steps',    type=int,   default=30_000,         help='num. of training steps')
parser.add_argument('-vf',  '--valid_freq',     type=int,   default=250,            help='validation every n steps')
parser.add_argument('-pat', '--patience',       type=int,   default=5,              help='valid ll patience')
parser.add_argument('-lr',                      type=float, default=0.01,           help='initial learning rate')
parser.add_argument('-t0',                      type=int,   default=500,            help='CAWR t0, 1 for fixed lr')
parser.add_argument('-eta_min',                 type=float, default=1e-4,           help='CAWR eta min')
args = parser.parse_args()
dev = args.device
print(args)


#########################################################
################# create logging folder #################
#########################################################

dataset = args.dataset + ('' if args.split is None else ('_' + args.split))
idx = [args.dataset in x for x in [datasets.DEBD_DATASETS, datasets.MNIST_DATASETS, datasets.UCI_DATASETS, ['ptb288']]]
log_dir = 'log/pic/' + ['debd', 'mnist', 'uci', ''][np.argmax(idx)] + '/' + dataset + '/' + str(int(time.time())) + '/'
os.makedirs(log_dir, exist_ok=True)
json.dump(vars(args), open(log_dir + 'args.json', 'w'), sort_keys=True, indent=4)


#########################################################
############ load data & instantiate QPC-PIC ############
#########################################################

if args.dataset == 'ptb288':
    train, valid, test = datasets.load_ptb288()
    qpc = DLTM(trees.TREE_DICT[dataset], 'categorical', n_categories=50, norm_weight=False, learnable=False)
elif args.dataset in datasets.MNIST_DATASETS:
    train, test = datasets.load_mnist(ds_name=args.dataset, split=args.split)
    train_idx, valid_idx = train_test_split(np.arange(len(train)), train_size=1-args.vsp)
    train, valid = train[train_idx], train[valid_idx]
    leaf_type = 'categorical' if args.leaf_type is None else args.leaf_type
    qpc = DLTM(trees.TREE_DICT[dataset], leaf_type, n_categories=256, norm_weight=False, learnable=False)
elif args.dataset in datasets.DEBD_DATASETS:
    train, valid, test = datasets.load_debd(args.dataset)
    qpc = DLTM(learn_clt(train, 'bernoulli', n_categories=2), 'bernoulli', norm_weight=False, learnable=False)
else:
    train, valid, test = datasets.load_uci(args.dataset)
    qpc = DLTM(trees.TREE_DICT[dataset], 'gaussian', norm_weight=False, learnable=False)

pic = PIC(qpc.tree, qpc.leaf_type, args.n_units, sigma=args.sigma, n_categories=qpc.n_categories).to(device=dev)
print('PIC num. param: %d' % sum(param.numel() for param in pic.parameters() if param.requires_grad))


#########################################################
##################### training loop #####################
#########################################################

optimizer = torch.optim.Adam(pic.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)
z, log_w = zw_quadrature(mode=args.integration, nip=args.nip, a=-1, b=1, log_weight=True, device=dev)

train_lls_log, valid_lls_log, batch_time_log, best_valid_ll = [-np.inf], [-np.inf], [], -np.inf
tik_train = time.time()
for train_step in range(1, args.train_steps + 1):
    tik_batch = time.time()
    # materialise pic
    qpc.sum_logits, qpc.leaf_logits = pic(z, log_w=log_w, n_chunks=args.n_chunks)
    # evaluate qpc
    ll, batch_idx = 0, np.random.choice(len(train), args.batch_size * args.accum_steps, replace=False)
    for idx in np.array_split(batch_idx, args.accum_steps):
        ll_accum = (qpc(train[idx].to(dev), has_nan=False, normalize=True)).mean()
        (-ll_accum).backward(retain_graph=True if args.accum_steps > 1 else False)
        ll += float(ll_accum / args.accum_steps)
    # adam step
    lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    # if args.accum_steps > 1: torch.cuda.empty_cache()
    batch_time_log.append(time.time() - tik_batch)
    # validation & logging
    train_lls_log.append(float(ll))
    if max(valid_lls_log[-args.patience:]) < best_valid_ll:
        print('Early stopping: valid LL did not improve over the last %d steps' % (args.patience * args.valid_freq))
        break
    if train_step % args.valid_freq == 0:
        with torch.no_grad():
            valid_lls_log.append(float(torch.cat(
                [qpc(x.to(dev), has_nan=False, normalize=True) for x in valid.split(args.batch_size2)]).mean()))
        if valid_lls_log[-1] > best_valid_ll:
            best_valid_ll = valid_lls_log[-1]
            torch.save(pic, log_dir + 'pic.pt')
    if train_step % 50 == 0:
        print(train_step, dataset, 'LL: %.2f, lr: %.5f (best valid LL: %.2f, bt: %.2fs,  %.2f GiB)' %
              (ll, lr, best_valid_ll, np.mean(batch_time_log), (torch.cuda.max_memory_allocated() / 1024 ** 3)))
tok_train = time.time()


##########################################################
####### compute train-valid-test LLs of best model #######
##########################################################

with torch.no_grad():
    pic = torch.load(log_dir + 'pic.pt').to(dev)
    qpc.sum_logits, qpc.leaf_logits = pic(z, log_w=log_w)
    train_lls = torch.cat([qpc(x.to(dev), has_nan=False, normalize=True).cpu() for x in train.split(args.batch_size2)])
    valid_lls = torch.cat([qpc(x.to(dev), has_nan=False, normalize=True).cpu() for x in valid.split(args.batch_size2)])
    test_lls = torch.cat([qpc(x.to(dev), has_nan=False, normalize=True).cpu() for x in test.split(args.batch_size2)])


##########################################################
################### printing & logging ###################
##########################################################

print('\ndataset: %s' % dataset)
print('train (nats: %.2f, bpd: %.2f)' % (train_lls.mean(), (-train_lls.mean()) / (np.log(2) * train.size(1))))
print('valid (nats: %.2f, bpd: %.2f)' % (valid_lls.mean(), (-valid_lls.mean()) / (np.log(2) * train.size(1))))
print('test  (nats: %.2f, bpd: %.2f)' % (test_lls.mean(),  (-test_lls.mean()) / (np.log(2) * train.size(1))))
print('train time: %.2fs' % (tok_train - tik_train))
print('batch time: %.2fs' % np.mean(batch_time_log))
print('PIC param num: %d' % sum(param.numel() for param in pic.parameters() if param.requires_grad))
print('QPC param number: %d' % qpc.n_param)
print('max reserved  GPU: %.2f GiB' % (torch.cuda.max_memory_reserved() / 1024 ** 3) if dev == 'cuda' else 0)
print('max allocated GPU: %.2f GiB' % (torch.cuda.max_memory_allocated() / 1024 ** 3) if dev == 'cuda' else 0)

results = {
    'train_time': tok_train - tik_train,
    'batch_time': np.mean(batch_time_log),
    'max_reserved_gpu': torch.cuda.max_memory_reserved() if dev == 'cuda' else 0,
    'max_allocated_gpu': torch.cuda.max_memory_allocated() if dev == 'cuda' else 0,
    'train_lls_log': np.array(train_lls_log[1:]),  # [1:] removes -np.inf
    'valid_lls_log': np.array(valid_lls_log[1:]),  # [1:] removes -np.inf
    'train_lls': np.array(train_lls),
    'valid_lls': np.array(valid_lls),
    'test_lls': np.array(test_lls)
}
if args.dataset in datasets.MNIST_DATASETS:
    results['train_valid_idx'] = np.array((train_idx, valid_idx), dtype=tuple)
np.save(log_dir + 'results.npy', results)
