import gc
import json
import math
import os
import pickle
import random
import re
import string
import time
import warnings

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from bs4 import BeautifulSoup
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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

from criterion import MCRMSELoss, RMSELoss, WeightedMSELoss, WeightedRMSELoss, WeightedSmoothL1Loss
from models import CustomModel
from utils import AWP, AverageMeter, asMinutes, get_score, log_params_from_omegaconf_dict, seed_everything, timeSince

# 設定
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HYDRA_FULL_ERROR"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decontraction(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"he's", "he is", phrase)
    phrase = re.sub(r"there's", "there is", phrase)
    phrase = re.sub(r"We're", "We are", phrase)
    phrase = re.sub(r"That's", "That is", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"they're", "they are", phrase)
    phrase = re.sub(r"Can't", "Cannot", phrase)
    phrase = re.sub(r"wasn't", "was not", phrase)
    phrase = re.sub(r"don\x89Ûªt", "do not", phrase)
    phrase = re.sub(r"donãât", "do not", phrase)
    phrase = re.sub(r"aren't", "are not", phrase)
    phrase = re.sub(r"isn't", "is not", phrase)
    phrase = re.sub(r"What's", "What is", phrase)
    phrase = re.sub(r"haven't", "have not", phrase)
    phrase = re.sub(r"hasn't", "has not", phrase)
    phrase = re.sub(r"There's", "There is", phrase)
    phrase = re.sub(r"He's", "He is", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"You're", "You are", phrase)
    phrase = re.sub(r"I'M", "I am", phrase)
    phrase = re.sub(r"shouldn't", "should not", phrase)
    phrase = re.sub(r"wouldn't", "would not", phrase)
    phrase = re.sub(r"i'm", "I am", phrase)
    phrase = re.sub(r"I\x89Ûªm", "I am", phrase)
    phrase = re.sub(r"I'm", "I am", phrase)
    phrase = re.sub(r"Isn't", "is not", phrase)
    phrase = re.sub(r"Here's", "Here is", phrase)
    phrase = re.sub(r"you've", "you have", phrase)
    phrase = re.sub(r"you\x89Ûªve", "you have", phrase)
    phrase = re.sub(r"we're", "we are", phrase)
    phrase = re.sub(r"what's", "what is", phrase)
    phrase = re.sub(r"couldn't", "could not", phrase)
    phrase = re.sub(r"we've", "we have", phrase)
    phrase = re.sub(r"it\x89Ûªs", "it is", phrase)
    phrase = re.sub(r"doesn\x89Ûªt", "does not", phrase)
    phrase = re.sub(r"It\x89Ûªs", "It is", phrase)
    phrase = re.sub(r"Here\x89Ûªs", "Here is", phrase)
    phrase = re.sub(r"who's", "who is", phrase)
    phrase = re.sub(r"I\x89Ûªve", "I have", phrase)
    phrase = re.sub(r"y'all", "you all", phrase)
    phrase = re.sub(r"can\x89Ûªt", "cannot", phrase)
    phrase = re.sub(r"would've", "would have", phrase)
    phrase = re.sub(r"it'll", "it will", phrase)
    phrase = re.sub(r"we'll", "we will", phrase)
    phrase = re.sub(r"wouldn\x89Ûªt", "would not", phrase)
    phrase = re.sub(r"We've", "We have", phrase)
    phrase = re.sub(r"he'll", "he will", phrase)
    phrase = re.sub(r"Y'all", "You all", phrase)
    phrase = re.sub(r"Weren't", "Were not", phrase)
    phrase = re.sub(r"Didn't", "Did not", phrase)
    phrase = re.sub(r"they'll", "they will", phrase)
    phrase = re.sub(r"they'd", "they would", phrase)
    phrase = re.sub(r"DON'T", "DO NOT", phrase)
    phrase = re.sub(r"That\x89Ûªs", "That is", phrase)
    phrase = re.sub(r"they've", "they have", phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"should've", "should have", phrase)
    phrase = re.sub(r"You\x89Ûªre", "You are", phrase)
    phrase = re.sub(r"where's", "where is", phrase)
    phrase = re.sub(r"Don\x89Ûªt", "Do not", phrase)
    phrase = re.sub(r"we'd", "we would", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"weren't", "were not", phrase)
    phrase = re.sub(r"They're", "They are", phrase)
    phrase = re.sub(r"Can\x89Ûªt", "Cannot", phrase)
    phrase = re.sub(r"you\x89Ûªll", "you will", phrase)
    phrase = re.sub(r"I\x89Ûªd", "I would", phrase)
    phrase = re.sub(r"let's", "let us", phrase)
    phrase = re.sub(r"it's", "it is", phrase)
    phrase = re.sub(r"can't", "cannot", phrase)
    phrase = re.sub(r"don't", "do not", phrase)
    phrase = re.sub(r"you're", "you are", phrase)
    phrase = re.sub(r"i've", "I have", phrase)
    phrase = re.sub(r"that's", "that is", phrase)
    phrase = re.sub(r"i'll", "I will", phrase)
    phrase = re.sub(r"doesn't", "does not", phrase)
    phrase = re.sub(r"i'd", "I would", phrase)
    phrase = re.sub(r"didn't", "did not", phrase)
    phrase = re.sub(r"ain't", "am not", phrase)
    phrase = re.sub(r"you'll", "you will", phrase)
    phrase = re.sub(r"I've", "I have", phrase)
    phrase = re.sub(r"Don't", "do not", phrase)
    phrase = re.sub(r"I'll", "I will", phrase)
    phrase = re.sub(r"I'd", "I would", phrase)
    phrase = re.sub(r"Let's", "Let us", phrase)
    phrase = re.sub(r"you'd", "You would", phrase)
    phrase = re.sub(r"It's", "It is", phrase)
    phrase = re.sub(r"Ain't", "am not", phrase)
    phrase = re.sub(r"Haven't", "Have not", phrase)
    phrase = re.sub(r"Could've", "Could have", phrase)
    phrase = re.sub(r"youve", "you have", phrase)
    phrase = re.sub(r"donå«t", "do not", phrase)
    return phrase


def remove_punctuations(text):
    for punctuation in list(string.punctuation):
        text = text.replace(punctuation, "")
    return text


def clean_number(text):
    text = re.sub(r"(\d+)([a-zA-Z])", "\g<1> \g<2>", text)
    text = re.sub(r"(\d+) (th|st|nd|rd) ", "\g<1>\g<2> ", text)
    text = re.sub(r"(\d+),(\d+)", "\g<1>\g<2>", text)
    return text


def clean_text(text):
    text = decontraction(text)
    # text = text.lower()
    # text = re.sub(r"[^\w\s]", "", text, re.UNICODE)
    return text


def get_additional_special_tokens():
    special_tokens_replacement = {
        "\n": "[BR]",
        "Paragraph": "[PARAGRAPH]",
        "Author": "[AUTHOR]",
        "misspell": "[MISSPELL]",
    }
    return special_tokens_replacement


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

    # jsonファイル（typoの修正）を読み込む
    # 改善しなかったので一旦コメントアウト
    # with open("conf/typo.json", "r") as f:
    #     typo_dict = json.load(f)

    # typoの修正
    # 改善しなかったので一旦コメントアウト
    # print("modify typo")
    # output_df["text"] = output_df["text"].replace(typo_dict, regex=True)
    # # check
    # if len(output_df[output_df["text"].str.contains("ineretsed")]) > 0:
    #     print("fixed typo")

    return output_df


def len2text(text: str):
    words = text.split()
    word_count = len(words)

    if word_count < 50:
        return "Quite short"
    if word_count < 100:
        return "Short"
    if word_count < 250:
        return "Middle"
    else:
        return "Long"


def text_cleaning(text):
    template = re.compile(r"https?://\S+|www\.\S+")  # Removes website links
    text = template.sub(r"", text)

    soup = BeautifulSoup(text, "lxml")  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub("\n+", "\n", text)
    text = re.sub("\.+", ".", text)
    text = re.sub(" +", " ", text)  # Remove Extra Spaces
    text = re.sub(r"\d+", "", text)  # Remove numbers

    return text


def create_text(input_df, tokenizer, cfg):
    output_df = input_df.copy()
    sep = tokenizer.sep_token
    print(sep)

    output_df["text_len_type"] = output_df["text"].apply(len2text)

    # 前処理を行う
    print("clean text")
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("     ", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("  ", " "))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("&", "and"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace('"(', '" ('))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("...", " "))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace(".....", " "))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace(" ,", ","))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("--", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace(" ", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[i]", "i"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[t]", "t"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[A]", "A"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[...]", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[. . .]", ""))
    # output_df["text"] = output_df["text"].apply(lambda x: x.replace("(CommonLit Staff)", ""))
    # output_df["text"] = output_df["text"].apply(lambda x: x.replace("(CommonLit)", ""))
    # output_df["text"] = output_df["text"].apply(lambda x: x.replace("(commonlit.org)", ""))
    # output_df["text"] = output_df["text"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
    # output_df["text"] = output_df["text"].str.replace(r"\(\d+(-\d+)?\)", "", regex=True).str.strip()
    # output_df["text"] = output_df["text"].str.replace(r"\[\d+\]", "", regex=True).str.strip()
    output_df["text"] = output_df["text"].str.replace(r"\[([^\]]+)\]", r"\1", regex=True)
    # output_df["text"] = output_df["text"].str.replace(r"\(Commonlit \d+\)", "", regex=True)
    # output_df["text"] = output_df["text"].str.replace(r"\(Commonlit Staff \d+\)", "", regex=True)

    output_df["text"] = output_df["text"].apply(lambda x: x.replace("[", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("]", ""))

    # パラグラフを参照している部分を特殊トークンに置換
    output_df["text"] = output_df["text"].str.replace(r"\(\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(\d+(-\d+)?\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\[\d+\]", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(paragraph \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Paragraph \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Paragraph \d+ and \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Paragraph\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(par \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(para \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(par.\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(par. \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Par \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Paragraph \d+, lines \d+-\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Paragraphs \d+-\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(paragraphs \d+,\d+,\d+\)", "[PARAGRAPH]", regex=True)

    # 文章中を参照している部分を特殊トークンに（パラグラフと同じ扱いでいいのか？）
    output_df["text"] = output_df["text"].str.replace(r"\(line \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Line \d+-\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Lines \d+-\d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(paragraph \d+-\d+\)", "[PARAGRAPH]", regex=True)

    # ラインとパラグラフを参照している部分を特殊トークンに（パラグラフと同じ扱いでいいのか？）
    output_df["text"] = output_df["text"].str.replace(r"\(paragraph \d+ line \d+\)", "[PARAGRAPH]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(PA \d+ L \d+\)", "[PARAGRAPH]", regex=True)

    # 著者情報を特殊トークンにする
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(CommonLit Staff)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("CommonLit Staff", "[AUTHOR]"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(CommonLit)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(commonlit.org)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].str.replace(r"\(Commonlit \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Commonlit Staff \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(commonlit.org)", "[AUTHOR]"))

    output_df["text"] = output_df["text"].str.replace(r"\(The Third Wave \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(The Third Wave \d+\)", "[AUTHOR]", regex=True)

    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(UShistory.org)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("( UShistory.org)", "[AUTHOR]"))

    output_df["text"] = output_df["text"].str.replace(r"\(UShistory.org \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(USHistory.org paragraph \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("UShistory.org", "[AUTHOR]"))

    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(Aristotle)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].str.replace(r"\(Aristotle \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Aristotle, \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Aristotle \d+-\d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("Aristotle", "[AUTHOR]"))

    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(Upton Sinclair)", "[AUTHOR]"))

    output_df["text"] = output_df["text"].apply(lambda x: x.replace("(Sinclair)", "[AUTHOR]"))
    output_df["text"] = output_df["text"].str.replace(r"\(Sinclair \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].str.replace(r"\(Sinclair, para. \d+\)", "[AUTHOR]", regex=True)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("Upton Sinclair", "[AUTHOR]"))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("Sinclair", "[AUTHOR]"))

    # カンマのあとにスペース
    output_df["text"] = output_df["text"].str.replace(r"(?<=[.,])(?=[^\s])", " ", regex=True)

    output_df["text"] = output_df["text"].apply(clean_text)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("  ", " "))  # もう一度

    # output_df["text"] = output_df["text"].apply(remove_punctuations)
    output_df["text"] = output_df["text"].apply(clean_number)
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("— ", ""))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace("; ", ","))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace(": ", ","))
    output_df["text"] = output_df["text"].apply(lambda x: x.replace(" .", "."))

    output_df["prompt_text"] = output_df["prompt_text"].apply(lambda x: x.replace("  ", " "))
    output_df["prompt_text"] = output_df["prompt_text"].apply(clean_text)
    # output_df["prompt_text"] = output_df["prompt_text"].apply(remove_punctuations)
    output_df["prompt_text"] = output_df["prompt_text"].apply(clean_number)
    output_df["prompt_text"] = output_df["prompt_text"].apply(lambda x: x.replace("— ", ""))
    output_df["prompt_text"] = output_df["prompt_text"].apply(lambda x: x.replace("; ", ","))
    output_df["prompt_text"] = output_df["prompt_text"].apply(lambda x: x.replace(": ", ","))

    # テキストクリーニング
    output_df["prompt_text"] = output_df["prompt_text"].apply(text_cleaning)

    # スペルミスの単語をここで特殊トークンとして認識させることでwordingの予測精度を向上できないか？
    with open("conf/typo.json", "r") as f:
        typo_dict = json.load(f)

    # dictのキーのみに対してループ
    # for k in typo_dict.keys():
    #     output_df["text"] = output_df["text"].apply(lambda x: x.replace(k + " ", "[MISSPELL] "))

    output_df["summarized_prompt_text"] = ""

    # 要約文を追加
    output_df.loc[
        output_df["prompt_id"] == "39c16e", "summarized_prompt_text"
    ] = "In discussing the construction of plots in Tragedy, Aristotle posits that an ideal tragedy should be complex and imitate actions that evoke both pity and fear. The central character should neither be wholly virtuous nor entirely wicked; instead, their downfall should arise from a significant error or frailty, like figures such as Oedipus. Additionally, while some tragedies might use a double plot thread for the sake of audience appeal, true tragic pleasure arises from single-threaded narratives where the change in fortune is from good to bad due to the character's error, rather than vice."
    output_df.loc[
        output_df["prompt_id"] == "3b9047", "summarized_prompt_text"
    ] = "Egyptian society was hierarchically structured, with gods and pharaohs at the apex, believed to control the universe and possessing absolute power, respectively. The pharaoh's administrative responsibilities were overseen by a vizier and scribes, followed by powerful nobles and priests in status, who managed tributes and religious ceremonies. Soldiers, skilled workers, and merchants constituted the middle tiers, while farmers and slaves, who endured high taxes and labor demands, formed the base; yet, social mobility existed, allowing some to rise through education and bureaucratic roles."

    output_df.loc[
        output_df["prompt_id"] == "814d6b", "summarized_prompt_text"
    ] = 'In 1967, history teacher Ron Jones conducted "The Third Wave" experiment at Cubberley High School in Palo Alto to demonstrate how individuals follow the crowd even when it leads to harmful actions. He introduced strict discipline and authoritarian rules, and within days, the movement grew from 30 to over 200 students, displaying extreme loyalty and discipline. However, sensing it was spiraling out of control, Jones ended the experiment by revealing its true purpose, emphasizing the dangers of blind obedience and superiority complexes.'

    output_df.loc[
        output_df["prompt_id"] == "ebad26", "summarized_prompt_text"
    ] = "The family had direct knowledge of the meat industry's malpractices from working in Packingtown, revealing that spoiled meat was often canned or turned into sausage, and every part of a pig was used except its squeal. They learned of various deceitful methods such as using chemicals to alter the meat's appearance and taste, reprocessing rejected sausages from Europe, and creating hams with discarded parts. Extremely unsanitary conditions prevailed, with meat often contaminated by rat feces, poisoned rats, and other debris; this contaminated meat, combined with other waste, was regularly repackaged and sold to consumers after being chemically treated."

    # テキストのみ
    if cfg.use_text == "text":
        output_df["full_text"] = output_df["text"]

    # 質問とテキスト
    elif cfg.use_text == "prompt_question_and_text":
        output_df["full_text"] = output_df["prompt_question"] + sep + output_df["text"]

    # 質問とタイトルとテキスト
    elif cfg.use_text == "prompt_question_title_text":
        output_df["full_text"] = (
            output_df["prompt_question"]
            + " "
            + sep
            + " "
            + output_df["prompt_title"]
            + " "
            + sep
            + " "
            + output_df["text"]
        )

    # 質問とタイトルとテキスト
    elif cfg.use_text == "type_prompt_question_title_text":
        output_df["full_text"] = (
            output_df["text_len_type"]
            + " "
            + sep
            + " "
            + output_df["prompt_question"]
            + " "
            + sep
            + " "
            + output_df["prompt_title"]
            + " "
            + sep
            + " "
            + output_df["text"]
        )

    # 目的変数を先頭に追加
    elif cfg.use_text == "target_prompt_question_title_text":
        output_df["full_text"] = (
            "content wording"
            + " "
            + sep
            + " "
            + output_df["prompt_question"]
            + " "
            + sep
            + " "
            + output_df["prompt_title"]
            + " "
            + sep
            + " "
            + output_df["text"]
        )
    # すべてのテキスト情報
    elif cfg.use_text == "full_text":
        output_df["full_text"] = (
            output_df["text_len_type"]
            + " "
            + sep
            + " "
            + output_df["text"]
            + " "
            + sep
            + " "
            + output_df["prompt_question"]
            + " "
            + sep
            + " "
            + output_df["prompt_title"]
            + " "
            + sep
            + " "
            + output_df["prompt_text"]
        )
        # 改行文は文章の区切りのように使用している可能性があるので、別文字で置換する（本来は別トークンを用意するべき）
        output_df["full_text"] = output_df["full_text"].str.replace("\n\n", "| ")
        output_df["full_text"] = output_df["full_text"].str.replace("\r\n", "| ")

        # 一部の改行\nは削除するようにする
        output_df["full_text"] = output_df["full_text"].str.replace("\n", "")

    elif cfg.use_text == "full_text2":
        output_df["full_text"] = (
            output_df["text"]
            + " "
            + sep
            + " "
            + output_df["prompt_question"]
            + " "
            + sep
            + " "
            + output_df["prompt_title"]
            + " "
            + sep
            + " "
            + output_df["summarized_prompt_text"]
        )

    # テキストとpromot
    elif cfg.use_text == "text_prompt":
        output_df["full_text"] = (
            output_df["text_len_type"]
            + " "
            + sep
            + " "
            + output_df["text"]
            + " "
            + sep
            + " "
            + output_df["prompt_text"]
        )
        # 改行文は文章の区切りのように使用している可能性があるので、別文字で置換する（本来は別トークンを用意するべき）
        output_df["full_text"] = output_df["full_text"].str.replace("\n\n", "| ")
        output_df["full_text"] = output_df["full_text"].str.replace("\r\n", "| ")

        # 一部の改行\nは削除するようにする
        output_df["full_text"] = output_df["full_text"].str.replace("\n", "")

    # テキストと質問（順番入れ替え）
    elif cfg.use_text == "text_question":
        # テキストを先頭にしてみる
        output_df["full_text"] = output_df["text"] + sep + output_df["prompt_question"]

    elif cfg.use_text == "text_question_title":
        output_df["full_text"] = (
            output_df["text"] + sep + output_df["prompt_question"] + sep + output_df["prompt_title"]
        )
    else:
        output_df["full_text"] = output_df["text"]

    print("prepare input. sample text:\n")
    print(f"{output_df['full_text'][0]}")
    print("")
    print(f"{output_df[output_df['student_id']=='1853d9257219']['full_text'].values[0]}")
    print("")
    print(f"{output_df[output_df['student_id'] == '9b8236dd1e52']['full_text'].values[0]}")
    print("")
    print(f"{output_df[output_df['student_id'] == '2c868d9bd1e7']['full_text'].values[0]}")
    print("")
    print(f"{output_df[output_df['student_id'] == '574369ff8f20']['full_text'].values[0]}")
    print("")
    print(f"{output_df[output_df['student_id'] == '4f63273f4aa9']['full_text'].values[0]}")
    print("")
    print(f"{output_df[output_df['student_id'] == 'bce3dd3877d2']['full_text'].values[0]}")

    return output_df


def train_fn(train_loader, model, criterion, optimizer, scheduler, epoch, cfg, log_var_l=None, weighted_criterion=None):
    model.train()
    # AWP
    if cfg.use_awp:
        awp = AWP(model, criterion, optimizer, adv_lr=cfg.awp.params.lr, adv_eps=cfg.awp.params.eps)
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

            if cfg.loss.name == "WeightedMSELoss":
                loss = criterion(y_preds, labels)
                myloss = weighted_criterion(y_preds, labels, log_var_l[0]["params"])
            else:
                loss = criterion(y_preds, labels)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        log_scaler("train_loss", loss.item(), step)

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        # AWP attack
        if cfg.use_awp:
            if cfg.awp.start_epoch <= epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()  # WPする前のモデルに復元

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()

        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )

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
            {
                "params": [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    # llrd
    def get_optimizer_grouped_parameters(model, layerwise_lr, layerwise_weight_decay, layerwise_lr_decay):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "weight_decay": 0.0,
                "lr": layerwise_lr,
            },
        ]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": layerwise_weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            lr *= layerwise_lr_decay
        return optimizer_grouped_parameters

    # optimizer
    if cfg.use_llrd:
        from transformers import AdamW

        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            cfg.llrd.params.layerwise_lr,
            cfg.llrd.params.layerwise_weight_decay,
            cfg.llrd.params.layerwise_lr_decay,
        )

        optimizer = AdamW(
            grouped_optimizer_params,
            lr=cfg.llrd.params.layerwise_lr,
            eps=cfg.llrd.params.layerwise_adam_epsilon,
            correct_bias=not cfg.llrd.params.layerwise_use_bertadam,
        )

    else:
        from torch.optim import AdamW

        optimizer_parameters = get_optimizer_params(
            model, cfg.optimizer.params.lr, cfg.optimizer.params.lr, cfg.optimizer.params.weight_decay
        )

        log_var_l = [
            {
                "params": [torch.zeros((1,), requires_grad=True, device=DEVICE) for i in range(2)],
                "lr": cfg.optimizer.params.lr,
                "weight_decay": 0.0,
            },
        ]
        if cfg.loss.name == "WeightedMSELoss":
            optimizer_parameters += log_var_l
        else:
            optimizer_parameters = optimizer_parameters

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

    # set criterion
    if cfg.loss.name == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss(reduction="mean")
    elif cfg.loss.name == "RMSELoss":
        criterion = RMSELoss(reduction="mean")
    elif cfg.loss.name == "MCRMSELoss":
        criterion = MCRMSELoss()
    elif cfg.loss.name == "WeightedMSELoss":
        criterion = RMSELoss(reduction="mean")
        weighted_criterion = WeightedMSELoss()
    elif cfg.loss.name == "WeightedSmoothL1Loss":
        criterion = WeightedSmoothL1Loss()
    elif cfg.loss.name == "WeightedRMSELoss":
        criterion = WeightedRMSELoss()
    else:
        criterion = nn.MSELoss(reduction="mean")

    # train loop
    best_score = np.inf

    for epoch in range(cfg.epochs):
        start_time = time.time()

        # train
        if cfg.loss.name == "WeightedMSELoss":
            avg_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                cfg,
                log_var_l=log_var_l,
                weighted_criterion=weighted_criterion,
            )
        else:
            avg_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                cfg,
                log_var_l=None,
                weighted_criterion=None,
            )

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
        # 追加トークンを入れる
        special_tokens_replacement = get_additional_special_tokens()
        all_special_tokens = list(special_tokens_replacement.values())

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=True, additional_special_tokens=all_special_tokens
        )
        tokenizer.save_pretrained(os.path.join(cfg.output_dir, "tokenizer"))
        # 追加したトークンが入っているか確認
        print(tokenizer.additional_special_tokens)
        # 特殊トークンでテキストが変換されているか確認
        sample_text = "Finally, they reused old hog skins that [SEP] people would not eat until they chopped it up and false advertised it by relabeling it as head cheese [PARAGRAPH]."
        sample_input_ids = tokenizer.encode(sample_text)
        # Print the IDs
        print("Token IDs:", sample_input_ids)
        # Decode the IDs back to text
        decoded_text = tokenizer.decode(sample_input_ids)

        # Print the decoded text
        print("\nDecoded Text:", decoded_text)

        # create_text
        train_df = create_text(train_df, tokenizer, cfg)

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
            fold = StratifiedKFold(
                n_splits=cfg.split.params.n_splits, shuffle=cfg.split.params.shuffle, random_state=cfg.globals.seed
            )
            for fold, (train_index, val_index) in enumerate(fold.split(train_df, train_df["bin10_content"])):
                train_df.loc[val_index, "fold"] = fold

        elif cfg.split.name == "MultilabelStratifiedKFold":
            fold = MultilabelStratifiedKFold(
                n_splits=cfg.split.params.n_splits, shuffle=cfg.split.params.shuffle, random_state=cfg.globals.seed
            )
            for n, (train_index, val_index) in enumerate(fold.split(train_df, train_df[["content", "wording"]])):
                train_df.loc[val_index, "fold"] = int(n)

        elif cfg.split.name == "Bins_MultilabelStratifiedKFold":
            train_df["bin10_content"] = pd.cut(train_df["content"], bins=10, labels=list(range(10)))
            train_df["bin10_wording"] = pd.cut(train_df["wording"], bins=10, labels=list(range(10)))
            fold = MultilabelStratifiedKFold(
                n_splits=cfg.split.params.n_splits, shuffle=cfg.split.params.shuffle, random_state=cfg.globals.seed
            )
            for n, (train_index, val_index) in enumerate(
                fold.split(train_df, train_df[["bin10_content", "bin10_wording"]])
            ):
                train_df.loc[val_index, "fold"] = int(n)

        elif cfg.split.name == "GroupKFold":
            # prompt_idごとに分ける
            fold = GroupKFold(n_splits=4)

            for n, (train_index, val_index) in enumerate(fold.split(train_df, groups=train_df["prompt_id"])):
                train_df.loc[val_index, "fold"] = int(n)

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
