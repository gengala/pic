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
parser.add_argument('-device',                  type=str,   default='cuda', help='cpu | cuda')
parser.add_argument('-ds',  '--dataset',        type=str,   default=None,   help='dataset name')
parser.add_argument('-split',                   type=str,   default=None,   help='dataset split for EMNIST')
parser.add_argument('-hd',  '--hidden_dim',     type=int,   default=128,    help='ltm hidden dim')
parser.add_argument('-lt',  '--leaf_type',      type=str,   default=None,   help='leaf distribution type')
parser.add_argument('-bs',  '--batch_size',     type=int,   default=256,    help='batch size')
parser.add_argument('-as',  '--accum_steps',    type=int,   default=1,      help='number of accumulation steps')
parser.add_argument('-vsp',                     type=float, default=0.05,   help='MNIST validation split percentage')
parser.add_argument('-ts',  '--train_steps',    type=int,   default=30_000, help='number of training steps')
parser.add_argument('-vf',  '--valid_freq',     type=int,   default=250,    help='validation every n train steps')
parser.add_argument('-pf',  '--print_freq',     type=int,   default=250,    help='print every n steps')
parser.add_argument('-pat', '--patience',       type=int,   default=5,      help='valid ll patience')
parser.add_argument('-lr',                      type=float, default=0.01,   help='learning rate, both for Adam and EM')
parser.add_argument('-t0',                      type=int,   default=500,    help='sched CAWR t0, 1 for fixed lr ')
parser.add_argument('-eta_min',                 type=float, default=1e-4,   help='sched CAWR eta min')
parser.set_defaults(em=True)
parser.add_argument('-em',   dest='em',         action='store_true',        help='training with EM')
parser.add_argument('-adam', dest='em',         action='store_false',       help='training with Adam')
parser.set_defaults(normalize=False)
parser.add_argument('-n',   dest='normalize',   action='store_true',        help='normalize HCLT')
parser.add_argument('-nn',  dest='normalize',   action='store_false',       help='do not normalize HCLT')
args = parser.parse_args()
dev = args.device
print(args)


#########################################################
################# create logging folder #################
#########################################################

dataset = args.dataset + ('' if args.split is None else ('_' + args.split))
idx = [args.dataset in x for x in [datasets.DEBD_DATASETS, datasets.MNIST_DATASETS, datasets.UCI_DATASETS, ['ptb288']]]
log_dir = 'log/ltm/' + ['debd', 'mnist', 'uci', ''][np.argmax(idx)] + '/' + dataset + '/' + str(int(time.time())) + '/'
os.makedirs(log_dir, exist_ok=True)
json.dump(vars(args), open(log_dir + 'args.json', 'w'), sort_keys=True, indent=4)

#########################################################
############ load data & instantiate HCLT ############
#########################################################

if args.dataset == 'ptb288':
    train, valid, test = datasets.load_ptb288()
    hclt = DLTM(trees.TREE_DICT[dataset], 'categorical', args.hidden_dim, n_categories=50, em=args.em).to(device=dev)
elif args.dataset in datasets.MNIST_DATASETS:
    train, test = datasets.load_mnist(ds_name=args.dataset, split=args.split)
    train_idx, valid_idx = train_test_split(np.arange(len(train)), train_size=1-args.vsp)
    train, valid = train[train_idx], train[valid_idx]
    leaf_type = 'categorical' if args.leaf_type is None else args.leaf_type
    hclt = DLTM(trees.TREE_DICT[dataset], leaf_type, args.hidden_dim, n_categories=256, em=args.em).to(device=dev)
elif args.dataset in datasets.DEBD_DATASETS:
    train, valid, test = datasets.load_debd(args.dataset)
    hclt = DLTM(learn_clt(train, 'bernoulli', n_categories=2), 'bernoulli', args.hidden_dim, em=args.em).to(device=dev)
else:
    train, valid, test = datasets.load_uci(args.dataset)
    hclt = DLTM(trees.TREE_DICT[dataset], 'gaussian', args.hidden_dim, em=args.em).to(device=dev)
print('HCLT num. param: %d' % sum(param.numel() for param in hclt.parameters() if param.requires_grad))

#########################################################
##################### training loop #####################
#########################################################

# the optimizer won't be used if training with EM, but we make use of the scheduler anyway for the EM step size
optimizer = torch.optim.Adam(hclt.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

train_lls_log, valid_lls_log, batch_time_log, best_valid_ll = [-np.inf], [-np.inf], [], -np.inf
tik_train = time.time()
for train_step in range(1, args.train_steps + 1):
    tik_batch = time.time()
    lr = optimizer.param_groups[0]['lr']
    if args.em:
        batch = train[np.random.choice(len(train), args.batch_size * args.accum_steps, False)].to(args.device)
        ll = hclt.em_step(x=batch, step_size=lr, n_chunks=args.accum_steps).mean()
    else:
        ll, batch_idx = 0, np.random.choice(len(train), args.batch_size * args.accum_steps, replace=False)
        for idx in np.array_split(batch_idx, args.accum_steps):
            ll_accum = hclt(train[idx].to(device=dev), has_nan=False, normalize=args.normalize).mean()
            (-ll_accum).backward(retain_graph=True if args.accum_steps > 1 else False)
            ll += float(ll_accum / args.accum_steps)
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()  # scheduler can be used both for EM step size and Adam learning rate
    batch_time_log.append(time.time() - tik_batch)
    # validation & logging
    train_lls_log.append(float(ll))
    if max(valid_lls_log[-args.patience:]) < best_valid_ll:
        print('Early stopping: valid LL did not improve over the last %d steps' % (args.patience * args.valid_freq))
        break
    if train_step % args.valid_freq == 0:
        with torch.no_grad():
            log_norm_const = hclt.log_norm_constant.cpu() if args.normalize and not args.em else 0
            valid_lls_log.append(float(torch.cat(
                [hclt(x.to(device=dev), has_nan=False).cpu() - log_norm_const for x in valid.split(args.batch_size)]).mean()))
        if valid_lls_log[-1] > best_valid_ll:
            best_valid_ll = valid_lls_log[-1]
            torch.save(hclt, log_dir + 'hclt.pt')
    if train_step % args.print_freq == 0:
        print(train_step, dataset, 'LL: %.2f, lr: %.5f (best valid LL: %.2f, bt: %.2fs,  %.2f GiB)' %
              (ll, lr, best_valid_ll, np.mean(batch_time_log), (torch.cuda.max_memory_allocated() / 1024 ** 3)))
tok_train = time.time()

##########################################################
####### compute train-valid-test LLs of best model #######
##########################################################

with torch.no_grad():
    log_norm_const = hclt.log_norm_constant.cpu() if args.normalize and not args.em else 0
    train_lls = torch.cat([hclt(x.to(dev), has_nan=False).cpu() for x in test.split(args.batch_size)]) - log_norm_const
    valid_lls = torch.cat([hclt(x.to(dev), has_nan=False).cpu() for x in test.split(args.batch_size)]) - log_norm_const
    test_lls = torch.cat([hclt(x.to(dev), has_nan=False).cpu() for x in test.split(args.batch_size)]) - log_norm_const

##########################################################
################### printing & logging ###################
##########################################################

print('\ndataset: %s' % dataset)
print('train (nats: %.2f, bpd: %.2f)' % (train_lls.mean(), (-train_lls.mean()) / (np.log(2) * train.size(1))))
print('valid (nats: %.2f, bpd: %.2f)' % (valid_lls.mean(), (-valid_lls.mean()) / (np.log(2) * train.size(1))))
print('test  (nats: %.2f, bpd: %.2f)' % (test_lls.mean(),  (-test_lls.mean()) / (np.log(2) * train.size(1))))
print('train time: %.2fs' % (tok_train - tik_train))
print('batch time: %.2fs' % np.mean(batch_time_log))
print('HCLT param number: %d' % hclt.n_param)
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
