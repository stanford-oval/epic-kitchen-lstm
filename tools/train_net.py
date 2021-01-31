#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import datetime
import os
import random

import numpy as np
from scipy.stats import gmean
import pprint
import torch
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader, Epickitchens
from slowfast.models import build_model, SlowFast
from slowfast.models.action_predictor import ActionPredictor
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter, TestMeter, EPICTestMeter
from tools.test_net import perform_test

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.freeze_fn('bn_statistics')
    if cur_epoch >= cfg.SOLVER.FREEZE_EPOCH:
        model.freeze_fn('none')
    else:
        model.freeze_fn('all_but_last')

    train_meter.iter_tic()
    if cur_epoch >= cfg.SOLVER.FREEZE_EPOCH:
        data_size = len(train_loader)
    else:
        data_size = len(train_loader) // 10

    for cur_iter, (inputs_img, inputs_index, labels, _, meta) in enumerate(train_loader):
        if cur_iter == data_size:
            break

        # construct input_label from input_index
        history_label_verb = torch.zeros(inputs_index.shape + (cfg.MODEL.NUM_CLASSES[0],))
        history_label_noun = torch.zeros(inputs_index.shape + (cfg.MODEL.NUM_CLASSES[1],))
        for i in range(inputs_index.shape[0]):
            for j, index in enumerate(inputs_index[i]):
                if index != -1:
                    # noinspection PyTypeChecker
                    dataset: Epickitchens = train_loader.dataset
                    # noinspection PyProtectedMember
                    cur_record = dataset._video_records[index]
                    if random.random() > dataset.sample_rate:
                        history_label_verb[i, j, cur_record.label['verb']] = 1
                        history_label_noun[i, j, cur_record.label['noun']] = 1
                    else:
                        history_label_verb[i, j, cur_record.temp_label['verb']] = 1
                        history_label_noun[i, j, cur_record.temp_label['noun']] = 1
        inputs_label = torch.cat((history_label_noun, history_label_verb), dim=2)

        # Transfer the data to the current GPU device.
        if isinstance(inputs_img, (list,)):
            for i in range(len(inputs_img)):
                inputs_img[i] = inputs_img[i].cuda(non_blocking=True)
        else:
            inputs_img = inputs_img.cuda(non_blocking=True)
        inputs_label = inputs_label.cuda()
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                pass
                # for i in range(len(val)):
                #     val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if not cfg.MODEL.LSTM:
            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs_img, meta["boxes"])

            else:
                # Perform the forward pass.
                preds = model(inputs_img)
        else:
            preds = model([inputs_img, inputs_label])

        if isinstance(labels, (dict,)):
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss_verb = loss_fun(preds[0], labels['verb'])
            loss_noun = loss_fun(preds[1], labels['noun'])
            loss = 0.5 * (loss_verb + loss_noun)

            # check Nan Loss.
            misc.check_nan_losses(loss)
        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            # check Nan Loss.
            misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            if isinstance(labels, (dict,)):
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [loss_verb, verb_top1_acc, verb_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_verb, verb_top1_acc, verb_top5_acc = (
                    loss_verb.item(),
                    verb_top1_acc.item(),
                    verb_top5_acc.item(),
                )

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [loss_noun, noun_top1_acc, noun_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_noun, noun_top1_acc, noun_top5_acc = (
                    loss_noun.item(),
                    noun_top1_acc.item(),
                    noun_top5_acc.item(),
                )

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                     (labels['verb'], labels['noun']),
                                                                                     (1, 5))
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, action_top1_acc, action_top5_acc = du.all_reduce(
                        [loss, action_top1_acc, action_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, action_top1_acc, action_top5_acc = (
                    loss.item(),
                    action_top1_acc.item(),
                    action_top5_acc.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (loss_verb, loss_noun, loss),
                    lr, inputs_img[0].size(0) * cfg.NUM_GPUS
                )
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    top1_err, top5_err, loss, lr, inputs_img[0].size(0) * cfg.NUM_GPUS
                )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs_img, inputs_index, labels, _, meta) in enumerate(val_loader):

        # construct input_label from input_index
        history_label_verb = torch.zeros(inputs_index.shape + (cfg.MODEL.NUM_CLASSES[0],))
        history_label_noun = torch.zeros(inputs_index.shape + (cfg.MODEL.NUM_CLASSES[1],))
        for i in range(inputs_index.shape[0]):
            for j, index in enumerate(inputs_index[i]):
                if index != -1:
                    # noinspection PyTypeChecker
                    dataset: Epickitchens = val_loader.dataset
                    # noinspection PyProtectedMember
                    cur_record = dataset._video_records[index]
                    if random.random() > dataset.sample_rate:
                        history_label_verb[i, j, cur_record.label['verb']] = 1
                        history_label_noun[i, j, cur_record.label['noun']] = 1
                    else:
                        history_label_verb[i, j, cur_record.temp_label['verb']] = 1
                        history_label_noun[i, j, cur_record.temp_label['noun']] = 1
        inputs_label = torch.cat((history_label_noun, history_label_verb), dim=2)

        # Transferthe data to the current GPU device.
        if isinstance(inputs_img, (list,)):
            for i in range(len(inputs_img)):
                inputs_img[i] = inputs_img[i].cuda(non_blocking=True)
        else:
            inputs_img = inputs_img.cuda(non_blocking=True)
        inputs_label = inputs_label.cuda()
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                pass
                # for i in range(len(val)):
                #     val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs_img, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
        else:
            if cfg.MODEL.LSTM:
                preds = model([inputs_img, inputs_label])
            else:
                preds = model(inputs_img)
            if isinstance(labels, (dict,)):
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                     (labels['verb'], labels['noun']),
                                                                                     (1, 5))
                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    inputs_img[0].size(0) * cfg.NUM_GPUS
                )
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err, top5_err, inputs_img[0].size(0) * cfg.NUM_GPUS
                )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return is_best_epoch


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs_img, inputs_label, _, _, _ in loader:
            if isinstance(inputs_img, (list,)):
                for i in range(len(inputs_img)):
                    inputs_img[i] = inputs_img[i].cuda(non_blocking=True)
            else:
                inputs_img = inputs_img.cuda(non_blocking=True)
            inputs_label = inputs_label.cuda()
            yield [inputs_img, inputs_label]

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model: SlowFast = build_model(cfg)

    original_cfg_model_lstm = cfg.MODEL.LSTM
    original_cfg_model_sep_pred = cfg.MODEL.SEP_PRED
    cfg.MODEL.LSTM = False
    cfg.MODEL.SEP_PRED = False
    non_lstm_model = build_model(cfg)
    cu.load_checkpoint("./pretrained/SlowFast.pyth", non_lstm_model, cfg.NUM_GPUS > 1)
    model_lstm = ActionPredictor.load_from_checkpoint("./pretrained/" + cfg.MODEL.PRETRAINED_NAME, cfg=cfg)
    model.load_from_non_lstm(non_lstm_model.cuda())
    model.load_from_lstm(model_lstm.cuda())
    cfg.MODEL.LSTM = original_cfg_model_lstm
    cfg.MODEL.SEP_PRED = original_cfg_model_sep_pred


    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=True)

    if cfg.BN.FREEZE:
        model.freeze_fn('bn_parameters')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and not cfg.TRAIN.FINETUNE:
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and cfg.TRAIN.FINETUNE:
        logger.info("Load from given checkpoint file. Finetuning.")
        _ = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = 0
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    if cfg.TRAIN.DATASET != 'epickitchens' or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")

    if cfg.EPICKITCHENS.TB_DIR == "":
        current_datetime = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        writer_dir = os.path.join("./runs/", current_datetime)
    else:
        writer_dir = cfg.EPICKITCHENS.TB_DIR

    tb_writer = SummaryWriter(writer_dir)
    tb_writer.global_iter = 0

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader) // 10, cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        if cfg.TRAIN.DATASET == 'epickitchens':
            train_meter = EPICTrainMeter(tb_writer, len(train_loader) // 10, cfg)
            val_meter = EPICValMeter(tb_writer, len(val_loader), cfg)
        else:
            train_meter = TrainMeter(len(train_loader) // 10, cfg)
            val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        if cur_epoch == cfg.SOLVER.FREEZE_EPOCH:
            print("Update meter epoch_iter")
            train_meter.update_epoch_iters(len(train_loader))
        if cur_epoch >= cfg.SOLVER.FREEZE_EPOCH:
            train_loader.dataset.sample_rate = \
                (cur_epoch - cfg.SOLVER.FREEZE_EPOCH + 1) / (cfg.SOLVER.MAX_EPOCH - cfg.SOLVER.FREEZE_EPOCH)
            train_in_order_loader = loader.construct_loader(cfg, "train-in-order", train_loader.dataset)
            if cfg.TEST.DATASET == 'epickitchens':
                test_meter = EPICTestMeter(
                    len(train_in_order_loader.dataset),
                    1,
                    cfg.MODEL.NUM_CLASSES,
                    len(train_in_order_loader),
                )
            else:
                test_meter = TestMeter(
                    len(train_in_order_loader.dataset),
                    1,
                    cfg.MODEL.NUM_CLASSES,
                    len(train_in_order_loader),
                )
            perform_test(train_in_order_loader, model, test_meter, cfg)
            train_loader = loader.construct_loader(cfg, "train", train_in_order_loader.dataset)
        else:
            # Shuffle the dataset.
            loader.shuffle_dataset(train_loader, cur_epoch)
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            is_best_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch)

