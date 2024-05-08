import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from pytorch_lightning.cli import LightningCLI

from climax.pretrain.datamodule import MultiSourceDataModule
from climax.pretrain.module import PretrainModule

from mpi4py import MPI
import climax.utils.tracer as tr
from climax.utils.callback import MyTracerCallback
import argparse

import torch
torch.backends.cudnn.enabled = False

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    comm.Barrier()

    tr.initialize(verbose=True)

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    tr.start("init")
    cli = LightningCLI(
        model_class=PretrainModule,
        datamodule_class=MultiSourceDataModule,
        seed_everything_default=42,
        # save_config_overwrite=True,
        run=False,
        # auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
        trainer_defaults={"callbacks": [MyTracerCallback()]},
        # trainer_defaults={"callbacks": [MyTracerCallback()], "profiler": "simple"},
    )
    tr.stop("init")
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())

    # fit() runs the training
    tr.start("train")
    ckpt_path = os.environ.get("CKPT_PATH", None) ## TODO: simple fix
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    tr.stop("train")

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        log_dir = cli.trainer.logger.log_dir

        if rank == 0:
            gp.pr_file(os.path.join(log_dir, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join(log_dir, "gp_timing.summary"))
        gp.finalize()


if __name__ == "__main__":
    main()
