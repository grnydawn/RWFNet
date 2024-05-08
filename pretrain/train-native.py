# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

#from pytorch_lightning.cli import LightningCLI
import torch
import torch.optim as optim
import torch.distributed as dist
import os
from climax.pretrain.datamodule import MultiSourceDataModule
from climax.arch import ClimaX, ClimaXp0, ClimaXp1, ClimaXp2 
import random
import yaml
import sys
import numpy as np
from climax.utils.metrics import lat_weighted_mse, mse
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

import os
import socket
import psutil
import re
from pickle import dump
import gc

import pynvml
import torch
import climax.utils.tracer as tr
import time

def gpu_memory_usage(device=0):
    return torch.cuda.memory_allocated(device) / 1024.0**3


def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)


def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3


def memory_cleanup():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        
        if (mem := gpu_memory_usage()) > 3.0:
            LOG.warning("GPU memory usage still high!")
            cnt = 0
            for obj in get_tensors():
                obj.detach()
                obj.grad = None
                obj.storage().resize_(0)
                cnt += 1
            gc.collect()
            torch.cuda.empty_cache()
            usage, cache, misc = gpu_memory_usage_all()
            LOG.warning(
                f"  forcibly cleared {cnt} tensors: {mem:.03f}GB -> {usage:.03f}GB (+{cache:.03f}GB cache, +{misc:.03f}GB misc)"       
            )


def get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue
            
            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue

torch.backends.cudnn.enabled = False

print(os.getenv('SLURM_LOCALID', '0'), "device_count:", torch.cuda.device_count())
print("ROCR_VISIBLE_DEVICES:", os.getenv("ROCR_VISIBLE_DEVICES"))
print("PYTORCH_HIP_ALLOC_CONF:", os.getenv("PYTORCH_HIP_ALLOC_CONF"))
# device0 = torch.device('cuda:{}'.format(int(os.environ['SLURM_LOCALID'])*2))
# device1 = torch.device('cuda:{}'.format(int(os.environ['SLURM_LOCALID'])*2+1))
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
device4 = torch.device('cuda:4')
device5 = torch.device('cuda:5')
device6 = torch.device('cuda:6')
device7 = torch.device('cuda:7')

device_list = [device0, device1, device2, device3, device4, device5, device6, device7]
# device_list = [device0, device1, device2, device3]
# device_list = [device0, device1]
nlayer_per_device = None

amp = True
mp = False
# mp = True


_seq = 0
_depth = 0
def backward_hook_wrapper(module, details=None):
    
    # define register_full_backward_pre_hook function
    def bwd_pre_hook_print(self, output):
        global _depth
        global _seq
        _depth = _depth + 1
        message = "  "*_depth + f'before backward of {module.__class__.__qualname__}'
        torch.cuda.reset_peak_memory_stats()
        print (message, torch.cuda.max_memory_allocated()/1024**3, torch.cuda.memory_stats()["allocated_bytes.all.peak"]/1024**3, torch.cuda.memory_stats()["active_bytes.all.peak"]/1024**3)
        # print(torch.cuda.memory_summary())
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return output

    # define register_full_backward_hook function
    def bwd_hook_print(self, input, output):
        global _depth
        global _seq
        message = "  "*_depth + f'after backward of {module.__class__.__qualname__}'
        _depth = _depth - 1
        print (message, _seq, torch.cuda.max_memory_allocated()/1024**3, torch.cuda.memory_stats()["allocated_bytes.all.peak"]/1024**3, torch.cuda.memory_stats()["active_bytes.all.peak"]/1024**3)
        # print(torch.cuda.memory_summary())
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open("hook-%d.pickle"%(_seq), 'wb'))

        _seq = _seq + 1
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return input

    # register hooks
    module.register_full_backward_pre_hook(bwd_pre_hook_print)
    module.register_full_backward_hook(bwd_hook_print)
    return module


"""
Setup DDP routines: Credit to https://github.com/ORNL/HydraGNN
"""

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0
    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        ## CADES, Frontier, Perlmutter
        world_size = int(os.environ["SLURM_NPROCS"])
        world_rank = int(os.environ["SLURM_PROCID"])
    ## Fall back to default
    if world_size is None:
        world_size = 1
    return int(world_size), int(world_rank)


def find_ifname(myaddr):
    """
    Find socket ifname for a given ip adress. This is for "GLOO" ddp setup.
    Usage example:
        find_ifname("127.0.0.1") will return a network interface name, such as "lo". "lo0", etc.
    """
    ipaddr = socket.gethostbyname(myaddr)
    ifname = None
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address == ipaddr:
                ifname = nic
                break
        if ifname is not None:
            break
    return ifname


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)
    return nlist


def setup_ddp():
    """ "Initialize DDP"""
    if os.getenv("DDSTORE_BACKEND") is not None:
        backend = os.environ["DDSTORE_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")
    world_size, world_rank = init_comm_size_and_rank()
    ## Default setting
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8889")
    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES/Frontier/Perlmutter specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]
    try:
        if backend in ["nccl", "gloo"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)
        if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
            ifname = find_ifname(master_addr)
            if ifname is not None:
                os.environ["GLOO_SOCKET_IFNAME"] = ifname
        print("Distributed data parallel: %s master at %s:%s" % (backend, master_addr, master_port))
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")
    return world_size, world_rank


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def configure_optimizers(model,lr,beta_1,beta_2,weight_decay,warmup_steps,max_steps,warmup_start_lr,eta_min):
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
        {
            "params": decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": weight_decay,
            "foreach": False,
        },
        {
            "params": no_decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": 0,
            "foreach": False,
        },
        ]
    )

    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr,
        eta_min,
    )
    scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}



def training_step(batch, batch_idx: int, net: ClimaX, lat):
    x, y, lead_times, variables, out_variables = batch
    x = x.to(device0)
    y = y.to(device0)
    lead_times = lead_times.to(device0)
    # print ("batch size:", x.shape)

    print("")
    print("x:", list(x.shape), x.min().item(), x.max().item())
    print("y:", list(y.shape), y.min().item(), y.max().item())
    print("vars:", variables)
    print("out_variables:", out_variables)

    print("")
    print("model:")
    for k, v in net.state_dict().items():
        print(k, list(v.shape), v.numel(), v.min().item(), v.max().item())

    loss_dict, _ = net.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=lat)
    loss_dict = loss_dict[0]


    return loss


def training_step2(batch, batch_idx: int, net0: ClimaXp0, net1_list: list, net2: ClimaXp2, lat):
    x, y, lead_times, variables, out_variables = batch
    # print ("size (GB):", x.numel()*4/1024**3, y.numel()*4/1024**3)
    x = x.to(device_list[0])
    lead_times = lead_times.to(device_list[0])

    # loss_dict, _ = net.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=lat)
    x = net0.forward(x, lead_times, variables)
    # print ("#1: net0.isnan:", np.array([ p.isnan().any().item() for p in net0.parameters() ]).any())
    # print ("#1: x.isnan:", x.isnan().any().item())
    for net, dev in zip(net1_list, device_list):
        x = x.to(dev)
        x = net.forward(x)
        # print ("#2: net.isnan:", np.array([ p.isnan().any().item() for p in net.parameters() ]).any())
        # print ("#2: x.isnan:", x.isnan().any().item())
    y = y.to(device_list[-1])
    loss_dict, _ = net2.forward(x, y, out_variables, [lat_weighted_mse], lat=lat)

    loss_dict = loss_dict[0]

    loss = loss_dict["loss"]
    # print ("#3: net.isnan:", np.array([ p.isnan().any().item() for p in net2.parameters() ]).any())
    # print ("#3: loss.isnan:", loss.isnan().any().item())

    return loss


def meminfo():
    memlist = list()
    for dev in device_list:
        memlist.append(torch.cuda.max_memory_allocated(dev))
        torch.cuda.reset_peak_memory_stats(dev)
    return memlist


def main():
    tr.initialize(verbose=True)

#    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

#    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())

    # fit() runs the training
#    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 

    # world_rank = dist.get_rank()
    world_rank = int(os.getenv("SLURM_PROCID", "0"))
# Load config file for experiment
#    try:
    config_path = sys.argv[1]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    # if world_rank==0: 
    #     print(conf,flush=True)

    seed = conf['seed_everything']

    precision=conf['trainer']['precision']

    strategy=conf['trainer']['strategy']

    max_epochs=conf['trainer']['max_epochs']

    checkpoint_path = "tmp" ##conf['trainer']['checkpoint_path']
  
    checkpoint_filename = "last" ##conf['trainer']['checkpoint_filename']

    # resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']
    
    lr = float(conf['model']['lr'])

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps =  conf['model']['warmup_steps']

    max_steps =  conf['model']['max_steps']

    warmup_start_lr =  float(conf['model']['warmup_start_lr'])

    eta_min =  float(conf['model']['eta_min'])

    class_path = conf['model']['net']['class_path']

    default_vars =  conf['model']['net']['init_args']['default_vars']

    img_size =  conf['model']['net']['init_args']['img_size']

    patch_size =  conf['model']['net']['init_args']['patch_size']
 
    emb_dim =  conf['model']['net']['init_args']['embed_dim']

    depth =  conf['model']['net']['init_args']['depth']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    num_heads = conf['model']['net']['init_args']['num_heads']

    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    drop_path = conf['model']['net']['init_args']['drop_path']

    drop_rate = conf['model']['net']['init_args']['drop_rate']

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_start_idx = conf['data']['dict_start_idx']

    dict_end_idx = conf['data']['dict_end_idx']

    dict_in_variables = conf['data']['dict_in_variables']

    dict_out_variables = conf['data']['dict_out_variables']

    dict_max_predict_ranges = conf['data']['dict_max_predict_ranges']

    dict_random_lead_time = conf['data']['dict_random_lead_time']

    dict_hrs_each_step = conf['data']['dict_hrs_each_step']

    dict_buffer_sizes = conf['data']['dict_buffer_sizes']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    img_size_x = img_size[0]
    img_size_y = img_size[1]
    
    # if world_rank==0:
    #     print("precision is ",precision,"strategy ",strategy,"max_epochs",max_epochs,flush=True)
    #     print("lr ",lr,"beta_1 ",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"class_path",class_path,"default_vars",default_vars,flush=True)
    #     print("img_size",img_size,"img_size_x",img_size_x,"img_size_y",img_size_y,"patch_size",patch_size,"emb_dim",emb_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,flush=True)
    #     print("dict_root_dirs",dict_root_dirs,"dict_start_idx",dict_start_idx,"dict_end_idx",dict_end_idx,"batch_size",batch_size,"num_workers",num_workers,flush=True)
    #     print("warmup_steps",warmup_steps,"max_steps",max_steps,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,flush=True)
    #     print("checkpoint_path",checkpoint_path,"checkpoint_filename",checkpoint_filename, flush=True)

    assert (img_size_x%patch_size)==0, "image_size_x % patch_size must be 0"
    assert (img_size_y%patch_size)==0, "image_size_y % patch_size must be 0"


    seq_len = (img_size_x // patch_size)*(img_size_y // patch_size)  #SEQ_LEN is the sequence length, which is the number of image patches 

    seed_everything(seed)

    #initialize ClimaX model
    if not mp:
        model = ClimaX(default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=emb_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            parallel_patch_embed=False,
        ).to(device0)

        # for name, module in model.named_modules():
        #     print (name, module.__class__.__qualname__, module)
        #     handle = backward_hook_wrapper(module)

        #set up DDP
        # local_rank = int(os.environ['SLURM_LOCALID'])
        # model = DDP(model, device_ids=[local_rank*2], output_device=[local_rank*2],find_unused_parameters=True)
        model = DDP(model, find_unused_parameters=True)
    else:
        nlayer_per_device = depth // len(device_list)
        assert depth % len(device_list) == 0

        # print("device_count:", torch.cuda.device_count(), os.environ["ROCR_VISIBLE_DEVICES"])
        model0 = ClimaXp0(default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=emb_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            parallel_patch_embed=False,
        ).to(device_list[0])

        model1_list = list()
        is_first = True
        is_last = False
        for i, dev in enumerate(device_list):
            if i == len(device_list) - 1:
                is_last = True
            m = ClimaXp1(default_vars=default_vars,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=emb_dim,
                depth=depth,
                decoder_depth=decoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                drop_rate=drop_rate,
                parallel_patch_embed=False,
                is_first=is_first,
                is_last=is_last,
                irange=(i,i+nlayer_per_device)
            ).to(dev)
            is_first = False
            model1_list.append(m)
        assert len(model1_list) == len(device_list)

        model2 = ClimaXp2(default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=emb_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            parallel_patch_embed=False,
            var_map=model0.var_map,
        ).to(device_list[-1])
        
        print("model device:", next(model0.parameters()).device)
        for m in model1_list:
            print("model device:", next(m.parameters()).device)
        print("model device:", next(model2.parameters()).device)

        model0 = DDP(model0, find_unused_parameters=True)
        for i in range(len(device_list)):
            model1_list[i] = DDP(model1_list[i], find_unused_parameters=True)
        model2 = DDP(model2, find_unused_parameters=True)

        ## print model
        num_params = 0
        print("-" * 50)
        for k, v in model0.state_dict().items():
            print("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
            num_params += v.numel()
        print("-" * 50)
        for i in range(len(device_list)):
            model1 = model1_list[i]
            for k, v in model1.state_dict().items():
                print("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
                num_params += v.numel()
            print("-" * 50)
        for k, v in model2.state_dict().items():
            print("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
            num_params += v.numel()
        print("-" * 50)
        print("%50s\t%20s\t%10d" % ("Total number of model params:", "", num_params))
        print("All (total, MB): %d %g" % (num_params, num_params * 4 / 1024 / 1024))


    data_module = MultiSourceDataModule(dict_root_dirs=dict_root_dirs,
        dict_start_idx=dict_start_idx,
        dict_end_idx=dict_end_idx,
        dict_buffer_sizes=dict_buffer_sizes,
        dict_in_variables=dict_in_variables,
        dict_out_variables=dict_out_variables,
        dict_max_predict_ranges=dict_max_predict_ranges,
        dict_random_lead_time=dict_random_lead_time,
        dict_hrs_each_step=dict_hrs_each_step,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    data_module.setup()

    lat, lon = data_module.get_lat_lon()

    train_dataloader = data_module.train_dataloader()

    if world_rank==0:
        print("lat.shape ",lat.shape,"lon.shape",lon.shape,flush=True)

        # for i, (name, param) in enumerate(model1_list[4].named_parameters()):
        #     print(i, "parameter name ",name," requires_gradient ",param.requires_grad,param.numel(), flush=True)

        # for i, param in enumerate(model1_list[4].parameters()):
        #     print("parameter name ",i," requires_gradient ",param.requires_grad,param.numel(), flush=True)


    if not mp:
        optimizer_scheduler = configure_optimizers(model,lr,beta_1,beta_2,weight_decay,warmup_steps,max_steps,warmup_start_lr,eta_min)
        optimizer = optimizer_scheduler['optimizer']
        scheduler = optimizer_scheduler['lr_scheduler']
    else:
        optimizer_scheduler0 = configure_optimizers(model0,lr,beta_1,beta_2,weight_decay,warmup_steps,max_steps,warmup_start_lr,eta_min)
        optimizer_scheduler1_list = list()
        for m in model1_list:
            x = configure_optimizers(m,lr,beta_1,beta_2,weight_decay,warmup_steps,max_steps,warmup_start_lr,eta_min)
            optimizer_scheduler1_list.append(x)
        optimizer_scheduler2 = configure_optimizers(model2,lr,beta_1,beta_2,weight_decay,warmup_steps,max_steps,warmup_start_lr,eta_min)

        optimizer0 = optimizer_scheduler0['optimizer']
        optimizer1_list = list()
        for m in optimizer_scheduler1_list:
            x = m['optimizer']
            optimizer1_list.append(x)
        optimizer2 = optimizer_scheduler2['optimizer']

        scheduler0 = optimizer_scheduler0['lr_scheduler']
        scheduler1_list = list()
        for m in optimizer_scheduler1_list:
            x = m['lr_scheduler']
            scheduler1_list.append(x)
        scheduler2 = optimizer_scheduler2['lr_scheduler']

    epoch_start = 0

    # if resume_from_checkpoint:
    #     if os.path.exists(checkpoint_path+"/"+checkpoint_filename+".ckpt"):
    #         checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename+".ckpt")
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         epoch_start = checkpoint['epoch']
    #     else:
    #         if world_rank==0:
    #             print("resume from checkpoint was set to True. But the checkpoint path does not exist. Pretrain from scratch.",flush=True)
    # else:
    #     if world_rank==0:
    #         print("resume from checkpoint was set to False. Pretrain from scratch.",flush=True)
    # if True:
    #     # checkpoint = torch.load(checkpoint_path+"/"+"checkpoint_%d_%d.ckpt"%(0, 3))
    #     checkpoint = torch.load("tmp/checkpoint_0_2.ckpt")
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scaler = torch.cuda.amp.GradScaler(growth_interval=100, enabled=amp)
    min_scale = 128

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print('saving allocated state during OOM')
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open('oom_snapshot.pickle', 'wb'))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=3, active=3, repeat=1),
        record_shapes=True,
        with_stack=True,
        # profile_memory=True, 
        # record_shapes=True, 
        # with_flops=True, 
        # with_modules=True,
        )
    prof = None
    if prof is not None:
        prof.start()

    for epoch in range(epoch_start,epoch_start+max_epochs):
        tr.start("epoch")
        t0 = time.time()

        #tell the model that we are in train mode. Matters because we have the dropout
        if not mp:
            model.train()
        else:
            model0.train()
            for m in model1_list:
                m.train()
            model2.train()
        loss = 0.0

        if world_rank==0:
            print("epoch ",epoch,flush=True)

        # torch.cuda.memory._record_memory_history(enabled=True)
        # torch.cuda.memory._record_memory_history(enabled='all')

        for batch_idx, batch in enumerate(train_dataloader):

            if batch_idx > 10:
                break
            
            tr.start("batch")

            # # print("#0:")
            # # save a snapshot of the memory allocations
            # with open(f"snapshot-%d-%d-%d.pickle"%(epoch, batch_idx, 0), "wb") as f:
            #     dump(s, f)

            with torch.cuda.amp.autocast(enabled=amp, dtype=torch.float16):
                if not mp:
                    loss = training_step(batch, batch_idx,model,lat)
                else:
                    loss2 = training_step2(batch, batch_idx,model0,model1_list,model2,lat)

 
            if world_rank==0:
                if not mp:
                    print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss.item(),time.time()-t0,flush=True)
                else:
                    print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss2.item(),time.time()-t0,flush=True)

            # # print("#1:")
            # # save a snapshot of the memory allocations
            # s = torch.cuda.memory._snapshot()
            # with open(f"snapshot-%d-%d-%d.pickle"%(epoch, batch_idx, 1), "wb") as f:
            #     dump(s, f)

            if not mp:
                optimizer.zero_grad()
            else:
                optimizer0.zero_grad()
                for opt in optimizer1_list:
                    opt.zero_grad()
                optimizer2.zero_grad()

            # # print("#2:")
            # # save a snapshot of the memory allocations
            # s = torch.cuda.memory._snapshot()
            # with open(f"snapshot-%d-%d-%d.pickle"%(epoch, batch_idx, 2), "wb") as f:
            #     dump(s, f)

            # loss.backward()
            # gc.collect()
            # torch.cuda.empty_cache()
            # memory_cleanup()
            if not mp:
                scaler.scale(loss).backward()
            else:
                scaler.scale(loss2).backward()

            # # print("#3:")
            # # save a snapshot of the memory allocations
            # s = torch.cuda.memory._snapshot()
            # with open(f"snapshot-%d-%d-%d.pickle"%(epoch, batch_idx, 3), "wb") as f:
            #     dump(s, f)

            if not mp:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                scaler.step(optimizer0)
                for opt in optimizer1_list:
                    scaler.step(opt)
                scaler.step(optimizer2)
                scaler.update()

            print ("scaler._scale:", scaler._scale)
            if scaler._scale < min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler._scale)

            # # print("#4:")
            # # save a snapshot of the memory allocations
            # s = torch.cuda.memory._snapshot()
            # with open(f"snapshot-%d-%d-%d.pickle"%(epoch, batch_idx, 4), "wb") as f:
            #     dump(s, f)
            tr.stop("batch")
 
            if prof is not None:
                prof.step()

            # torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         }, checkpoint_path+"/"+"checkpoint_%d_%d.ckpt"%(epoch, batch_idx))

        # if loss==0.0 and epoch==0:
        #     raise NotImplementedError(
        #         "train data loader is empty. Consider reducing batch size."
        #     )

        if not mp:
            scheduler.step()
        else:
            scheduler0.step()
            for x in scheduler1_list:
                x.step()
            scheduler2.step()

        # save a snapshot of the memory allocations
        s = torch.cuda.memory._snapshot()
        with open(f"snapshot.pickle", "wb") as f:
            dump(s, f)


        # tell CUDA to stop recording memory allocations now
        # torch.cuda.memory._record_memory_history(enabled=False)
        # torch.cuda.memory._record_memory_history(enabled=None)

        tr.stop("epoch")
        break

    # with open(f"snapshot.pickle", "wb") as f:
    #     dump(s, f)

    if prof is not None:
        prof.stop()
        prof.export_chrome_trace('profile.json')


    # # Check whether the specified checkpointing path exists or not
    # isExist = os.path.exists(checkpoint_path)
    # if not isExist:
    #     # Create a new directory because it does not exist
    #     os.makedirs(checkpoint_path)
    #     print("The new checkpoint directory is created!")        

    # torch.save({
    #         'epoch': epoch_start+max_epochs,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         }, checkpoint_path+"/"+checkpoint_filename+".ckpt")

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        if world_rank == 0:
            gp.pr_file("gp_timing.p%d" % world_rank)
        gp.pr_summary_file("gp_timing.summary")
        gp.finalize()

if __name__ == "__main__":
    world_size = int(os.getenv("SLURM_NTASKS", "0"))
    world_rank = int(os.getenv("SLURM_PROCID", "0"))
    local_rank = int(os.getenv("SLURM_LOCALID", "0"))

    setup_ddp()
    print("dist.get_backend return ",dist.get_backend(),flush=True)

    print("we are here after dist.init_process_group. world_size ",world_size,flush=True)

    main()




