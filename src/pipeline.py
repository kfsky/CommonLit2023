import gc
import json
import math
import os
import pickle
import re
import time
import warnings

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict

from criterion import RMSELoss
from models import CustomModel
from utils import AverageMeter, asMinutes, get_score, log_params_from_omegaconf_dict, seed_everything, timeSince

# 設定
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HYDRA_FULL_ERROR"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_input(tokenizer, max_len, text):
    inputs = tokenizer.encode_plus(
        text, return_tensors=None, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, tokenizer, max_len, df):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df["full_text"].values
        self.labels = df[["content", "wording"]].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.tokenizer, self.max_len, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def create_data(input_path: str, is_train: bool = True) -> pd.DataFrame:
    """
    データを結合する

    :param input_path:
        データpath
    :param is_train:
        trainかtestか
    :return:
        結合したdataframe
    """
    if is_train:
        prompt_df = pd.read_csv(os.path.join(input_path, "prompts_train.csv"))
        summary_df = pd.read_csv(os.path.join(input_path, "summaries_train.csv"))

    else:
        prompt_df = pd.read_csv(os.path.join(input_path, "prompts_test.csv"))
        summary_df = pd.read_csv(os.path.join(input_path, "summaries_test.csv"))

    output_df = pd.merge(prompt_df, summary_df, on="prompt_id", how="left")

    return output_df


def create_text(input_df, tokenizer):
    output_df = input_df.copy()
    sep = tokenizer.sep_token
    output_df["full_text"] = output_df["prompt_question"] + sep + output_df["text"]

    return output_df


def train_fn(train_loader, model, criterion, optimizer, scheduler, epoch, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)

        labels = labels.to(DEVICE)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        log_scaler("train_loss", loss.item(), step)

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()

        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

    return losses.avg


def valid_fn(valid_loader, model, criterion, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()

    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)

        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        log_scaler("valid_loss", loss.item(), step)

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to("cpu").numpy())

        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step, len(valid_loader), loss=losses, remain=timeSince(start, float(step + 1) / len(valid_loader))
                )
            )
    predictions = np.concatenate(preds)

    return losses.avg, predictions


def train_loop(folds, fold, cfg, tokenizer):
    print(f"========== fold: {fold} training ==========")

    # loader
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds[["content", "wording"]].values

    # dataset
    train_dataset = TrainDataset(
        tokenizer,
        cfg.dataset.params.max_len,
        train_folds,
    )
    valid_dataset = TrainDataset(
        tokenizer,
        cfg.dataset.params.max_len,
        valid_folds,
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.loader.train.batch_size,
        shuffle=cfg.loader.train.shuffle,
        num_workers=cfg.loader.train.num_workers,
        pin_memory=cfg.loader.train.pin_memory,
        drop_last=cfg.loader.train.drop_last,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.loader.valid.batch_size,
        shuffle=cfg.loader.valid.shuffle,
        num_workers=cfg.loader.valid.num_workers,
        pin_memory=cfg.loader.valid.pin_memory,
        drop_last=cfg.loader.valid.drop_last,
    )

    model = CustomModel(cfg, config_path=None, pretrained=True)
    torch.save(model.config, os.path.join(cfg.output_dir, "config.pth"))
    model.to(DEVICE)

    # optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model, cfg.optimizer.params.lr, cfg.optimizer.params.lr, cfg.optimizer.params.weight_decay
    )

    optimizer = AdamW(
        optimizer_parameters,
        lr=cfg.optimizer.params.lr,
        eps=cfg.optimizer.params.eps,
        betas=(cfg.optimizer.params.betas_min, cfg.optimizer.params.betas_max),
    )

    # scheduler
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler.name == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.scheduler.params.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler.name == "get_cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.scheduler.params.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.scheduler.params.num_cycles,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.scheduler.params.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.scheduler.params.num_cycles,
            )

        return scheduler

    num_train_steps = int(len(train_folds) / cfg.loader.train.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    # loop
    if cfg.loss.name == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss(reduction="mean")  # RMSELoss(reduction="mean")
    elif cfg.loss.name == "RMSELoss":
        criterion = RMSELoss(reduction="mean")
    best_score = np.inf

    for epoch in range(cfg.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, epoch, cfg)

        # valid
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, cfg)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        print(f"Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}")

        if best_score > score:
            best_score = score
            print(f"Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                os.path.join(cfg.output_dir, f"{cfg.model_name.replace('/', '-')}_fold{fold}_best.pth"),
            )

    predictions = torch.load(
        os.path.join(cfg.output_dir, f"{cfg.model_name.replace('/', '-')}_fold{fold}_best.pth"),
        map_location=torch.device("cpu"),
    )["predictions"]
    valid_folds[[f"pred_{c}" for c in ["content", "wording"]]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


def log_scaler(name, value, step):
    mlflow.log_metric(name, value, step=step)


def get_result(oof_df):
    labels = oof_df[["content", "wording"]].values
    preds = oof_df[[f"pred_{c}" for c in ["content", "wording"]]].values
    score, scores = get_score(labels, preds)

    return score, scores


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    if mlflow.active_run():
        mlflow.end_run()
    # create output directory
    cwd = hydra.utils.get_original_cwd()
    os.makedirs(os.path.join(cwd, "outputs", cfg.experiment_name), exist_ok=True)
    output_dir = os.path.join(cwd, "outputs", cfg.experiment_name)

    # logging model name
    mlflow.log_param("model_name", cfg.model_name)

    # temporarily disable struct mode
    with open_dict(cfg):
        cfg.output_dir = output_dir

    # set mlflow
    # windows環境ではこのようにしないといけない
    mlflow.set_tracking_uri("file://" + cwd + "\mlruns")
    mlflow.set_experiment(cfg.competition_name)

    with mlflow.start_run(run_name=cfg.experiment_name, nested=True):
        exp_params = log_params_from_omegaconf_dict(cfg)
        mlflow.log_params(exp_params)

        # set seed
        seed_everything(cfg.globals.seed)

        # load data
        train_df = create_data(os.path.join(cwd, "inputs"), is_train=True)
        if cfg.globals.debug:
            print("Use Debug Dataset")
            train_df = train_df.sample(1000, random_state=cfg.globals.seed).reset_index(drop=True)

        print(train_df.shape)

        # load_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        tokenizer.save_pretrained(os.path.join(cfg.output_dir, "tokenizer"))

        # create_text
        train_df = create_text(train_df, tokenizer)

        # create_folds
        train_df["fold"] = -1
        if cfg.split.name == "StratifiedKFold":
            fold = StratifiedKFold(
                n_splits=cfg.split.params.n_splits, shuffle=cfg.split.params.shuffle, random_state=cfg.globals.seed
            )
            for n, (train_index, val_index) in enumerate(fold.split(train_df, train_df["prompt_id"])):
                train_df.loc[val_index, "fold"] = n

        elif cfg.split.name == "Content_StratifiedKFold":
            train_df["bin10_content"] = pd.cut(train_df["content"], bins=10, labels=list(range(10)))
            fold = StratifiedKFold(n_splits=cfg.split.params.n_splits, shuffle=cfg.split.params.shuffle, random_state=cfg.globals.seed)
            for fold, (train_index, val_index) in enumerate(fold.split(train_df, train_df["bin10_content"])):
                train_df.loc[val_index, "fold"] = fold

        train_df["fold"] = train_df["fold"].astype(int)

        oof_df = pd.DataFrame()

        # スコアを保存
        cv_scores_content_dict = {}
        cv_scores_wording_dict = {}
        total_cv_scores_dict = {}

        for fold in range(cfg.split.params.n_splits):
            if fold in cfg.trn_fold:
                _oof_df = train_loop(train_df, fold, cfg, tokenizer)
                oof_df = pd.concat([oof_df, _oof_df])
                score, scores = get_result(_oof_df)
                cv_scores_content_dict[f"fold{fold}_content"] = scores[0]
                cv_scores_wording_dict[f"fold{fold}_wording"] = scores[1]
                total_cv_scores_dict[f"fold{fold}_score"] = score

        # save mlflow
        mlflow.log_metrics(cv_scores_content_dict)
        mlflow.log_metrics(cv_scores_wording_dict)
        mlflow.log_metrics(total_cv_scores_dict)

        oof_df = oof_df.reset_index(drop=True)
        score, scores = get_result(oof_df)

        print(f"CV score: {score}")
        mlflow.log_metric("CV score", score)

        oof_df.to_csv(os.path.join(cfg.output_dir, "oof_df.csv"), index=False)

    mlflow.end_run()


if __name__ == "__main__":
    main()
