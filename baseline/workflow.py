#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
Script for training and inference of the baseline model on CityFlow-NL.
"""
import json
import math
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from siamese_baseline_model import SiameseBaselineModel
from utils import TqdmToLogger, get_logger
from vehicle_retrieval_dataset import CityFlowNLDataset
from vehicle_retrieval_dataset import CityFlowNLInferenceDataset

torch.multiprocessing.set_sharing_strategy('file_system')

flags.DEFINE_integer("num_machines", 1, "Number of machines.")
flags.DEFINE_integer("local_machine", 0,
                     "Master node is 0, worker nodes starts from 1."
                     "Max should be num_machines - 1.")

flags.DEFINE_integer("num_gpus", 2, "Number of GPUs per machines.")
flags.DEFINE_string("config_file", "baseline/default.yaml",
                    "Default Configuration File.")

flags.DEFINE_string("master_ip", "127.0.0.1",
                    "Master node IP for initialization.")
flags.DEFINE_integer("master_port", 12000,
                     "Master node port for initialization.")

FLAGS = flags.FLAGS


def train_model_on_dataset(rank, train_cfg):
    _logger = get_logger("training")
    dist_rank = rank + train_cfg.LOCAL_MACHINE * train_cfg.NUM_GPU_PER_MACHINE
    dist.init_process_group(backend="nccl", rank=dist_rank,
                            world_size=train_cfg.WORLD_SIZE,
                            init_method=train_cfg.INIT_METHOD)
    dataset = CityFlowNLDataset(train_cfg.DATA)
    model = SiameseBaselineModel(train_cfg.MODEL).cuda()
    model = DistributedDataParallel(model, device_ids=[rank],
                                    output_device=rank,
                                    broadcast_buffers=train_cfg.WORLD_SIZE > 1)
    train_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=train_cfg.TRAIN.BATCH_SIZE,
                            num_workers=train_cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=train_cfg.TRAIN.LR.BASE_LR,
        momentum=train_cfg.TRAIN.LR.MOMENTUM)
    lr_scheduler = StepLR(optimizer,
                          step_size=train_cfg.TRAIN.LR.STEP_SIZE,
                          gamma=train_cfg.TRAIN.LR.WEIGHT_DECAY)
    if rank == 0:
        if not os.path.exists(
                os.path.join(train_cfg.LOG_DIR, train_cfg.EXPR_NAME)):
            os.makedirs(
                os.path.join(train_cfg.LOG_DIR, train_cfg.EXPR_NAME, "summary"))
            os.makedirs(os.path.join(train_cfg.LOG_DIR, train_cfg.EXPR_NAME,
                                     "checkpoints"))
        with open(os.path.join(train_cfg.LOG_DIR, train_cfg.EXPR_NAME,
                               "config.yaml"), "w") as f:
            f.write(train_cfg.dump())

    global_step = 0
    for epoch in range(train_cfg.TRAIN.START_EPOCH, train_cfg.TRAIN.EPOCH):
        if rank == 0:
            pbar = tqdm(total=len(dataloader), leave=False,
                        desc="Training Epoch %d" % epoch,
                        file=TqdmToLogger(),
                        mininterval=1, maxinterval=100, )
        for data in dataloader:
            optimizer.zero_grad()
            loss = model.module.compute_loss(data)
            if not (math.isnan(loss.data.item())
                    or math.isinf(loss.data.item())
                    or loss.data.item() > train_cfg.TRAIN.LOSS_CLIP_VALUE):
                loss.backward()
                optimizer.step()
            if rank == 0:
                pbar.update()
            if global_step % train_cfg.TRAIN.PRINT_FREQ == 0:
                if rank == 0:
                    _logger.info("EPOCH\t%d; STEP\t%d; LOSS\t%.4f" % (
                        epoch, global_step, loss.data.item()))
            global_step += 1
        if rank == 0:
            checkpoint_file = os.path.join(train_cfg.LOG_DIR,
                                           train_cfg.EXPR_NAME, "checkpoints",
                                           "CKPT-E%d-S%d.pth" % (
                                               epoch, global_step))
            torch.save(
                {"epoch": epoch, "global_step": global_step,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}, checkpoint_file)
            _logger.info("CHECKPOINT SAVED AT: %s" % checkpoint_file)
            pbar.close()
        lr_scheduler.step()

    dist.destroy_process_group()


def eval_model_on_dataset(rank, eval_cfg, queries):
    _logger = get_logger("evaluation")
    if rank == 0:
        if not os.path.exists(
                os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME)):
            os.makedirs(
                os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME, "logs"))
        with open(os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME,
                               "config.yaml"), "w") as f:
            f.write(eval_cfg.dump())
    dataset = CityFlowNLInferenceDataset(eval_cfg.DATA)
    model = SiameseBaselineModel(eval_cfg.MODEL)
    ckpt = torch.load(eval_cfg.EVAL.RESTORE_FROM,
                      map_location=lambda storage, loc: storage.cpu())
    restore_kv = {key.replace("module.", ""): ckpt["state_dict"][key] for key in
                  ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)
    model = model.cuda(rank)
    dataloader = DataLoader(dataset,
                            batch_size=eval_cfg.EVAL.BATCH_SIZE,
                            num_workers=eval_cfg.EVAL.NUM_WORKERS)

    for idx, query_id in enumerate(queries):
        if idx % eval_cfg.WORLD_SIZE != rank:
            continue
        _logger.info("Evaluate query %s on GPU %d" % (query_id, rank))
        track_score = dict()
        q = queries[query_id]
        for track in dataloader:
            lang_embeds = model.compute_lang_embed(q, rank)
            s = model.compute_similarity_on_frame(track, lang_embeds, rank)
            track_id = track["id"][0]
            track_score[track_id] = s
        top_tracks = sorted(track_score, key=track_score.get, reverse=True)
        with open(os.path.join(eval_cfg.LOG_DIR, eval_cfg.EXPR_NAME, "logs",
                               "%s.log" % query_id), "w") as f:
            for track in top_tracks:
                f.write("%s\n" % track)
    _logger.info("FINISHED.")


if __name__ == "__main__":
    FLAGS(sys.argv)
    cfg = get_default_config()
    cfg.merge_from_file(FLAGS.config_file)
    cfg.NUM_GPU_PER_MACHINE = FLAGS.num_gpus
    cfg.NUM_MACHINES = FLAGS.num_machines
    cfg.LOCAL_MACHINE = FLAGS.local_machine
    cfg.WORLD_SIZE = FLAGS.num_machines * FLAGS.num_gpus
    cfg.EXPR_NAME = cfg.EXPR_NAME + "_" + datetime.now().strftime(
        "%m_%d.%H:%M:%S.%f")
    cfg.INIT_METHOD = "tcp://%s:%d" % (FLAGS.master_ip, FLAGS.master_port)
    if cfg.TYPE == "TRAIN":
        mp.spawn(train_model_on_dataset, args=(cfg,),
                 nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
    elif cfg.TYPE == "EVAL":
        with open(cfg.EVAL.QUERY_JSON_PATH, "r") as f:
            queries = json.load(f)
        if os.path.isdir(cfg.EVAL.CONTINUE):
            files = os.listdir(os.path.join(cfg.EVAL.CONTINUE, "logs"))
            for q in files:
                del queries[q.split(".")[0]]
            cfg.EXPR_NAME = cfg.EVAL.CONTINUE.split("/")[-1]
        mp.spawn(eval_model_on_dataset, args=(cfg, queries),
                 nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
