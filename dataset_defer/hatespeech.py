import os
import re
import sys
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.utils
import requests

from sentence_transformers import SentenceTransformer

sys.path.append("../")

from .basedataset import BaseDataset
from .human import *


def check_embeddings(file_path, train_x, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Load embeddings from file if they exist, otherwise generate and save them.

    Args:
        file_path (str): Path to save/load the embeddings.
        train_x (list of str): Text data to embed.
        model_name (str): SentenceTransformer model name.

    Returns:
        np.ndarray: Embeddings as a NumPy array.
    """
    if os.path.exists(file_path):
        logging.info(f"Loading cached embeddings from {file_path}")
        embeddings = np.load(file_path)
    else:
        logging.info("Generating embeddings")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(train_x)
        logging.info(f"Saving embeddings to {file_path}")
        np.save(file_path, embeddings)
    return embeddings


class ModelPredictAAE:
    def __init__(self, modelfile, vocabfile):
        """
        Edited from https://github.com/slanglab/twitteraae
        """
        self.vocabfile = vocabfile
        self.modelfile = modelfile
        self.load_model()

    def load_model(self):
        self.N_wk = np.loadtxt(self.modelfile)
        self.N_w = self.N_wk.sum(1)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)
        self.wordprobs = (self.N_wk + 1) / self.N_k
        self.vocab = [
            L.split("\t")[-1].strip() for L in open(self.vocabfile, encoding="utf8")
        ]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.vocab) == self.N_wk.shape[0]

    def infer_cvb0(self, invocab_tokens, alpha, numpasses):
        doclen = len(invocab_tokens)

        Qs = np.zeros((doclen, self.K))
        for i in range(doclen):
            w = invocab_tokens[i]
            Qs[i, :] = self.wordprobs[self.w2num[w], :]
            Qs[i, :] /= Qs[i, :].sum()

        lik = Qs.copy()
        Q_k = Qs.sum(0)

        for itr in range(1, numpasses):
            for i in range(doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                Qs[i, :] /= Qs[i, :].sum()
                Q_k += Qs[i, :]

        Q_k /= Q_k.sum()
        return Q_k

    def predict_lang(self, tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        if len(invocab_tokens) < thresh1:
            return 0
        elif len(invocab_tokens) / len(tokens) < thresh2:
            return 0
        else:
            posterior = self.infer_cvb0(invocab_tokens, alpha, numpasses)
            return (np.argmax(posterior) == 0) * 1


def custom_split(tweet):
    tweet = tweet.replace('"', ' " ')
    tweet = re.sub(r'[^A-Za-z0-9\s]', ' ', tweet)
    return tweet.split()


class HateSpeech(BaseDataset):
    """ Hatespeech dataset from Davidson et al. 2017 """

    def __init__(
        self,
        data_dir,
        embed_texts,
        include_demographics,
        expert_type,
        device,
        synth_exp_param=[0.7, 0.7],
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        self.embed_texts = embed_texts
        self.include_demographics = include_demographics
        self.expert_type = expert_type
        self.synth_exp_param = synth_exp_param
        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        if not os.path.exists(self.data_dir + "/hatespeech_labeled_data.csv"):
            logging.info("Downloading HateSpeech dataset")
            r = requests.get(
                "https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv",
                allow_redirects=True,
            )
            with open(self.data_dir + "/hatespeech_labeled_data.csv", "wb") as f:
                f.write(r.content)
            hatespeech_data = pd.read_csv(self.data_dir + "/hatespeech_labeled_data.csv")
        else:
            hatespeech_data = pd.read_csv(self.data_dir + "/hatespeech_labeled_data.csv")

        if not os.path.exists(self.data_dir + "/model_count_table.txt"):
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_count_table.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_count_table.txt", "wb") as f:
                f.write(r.content)

        if not os.path.exists(self.data_dir + "/model_vocab.txt"):
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_vocab.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_vocab.txt", "wb") as f:
                f.write(r.content)

        self.model_aae = ModelPredictAAE(
            self.data_dir + "/model_count_table.txt",
            self.data_dir + "/model_vocab.txt",
        )

        hatespeech_data["demographics"] = hatespeech_data["tweet"].apply(custom_split).apply(
            lambda x: self.model_aae.predict_lang(x)
        )

        distribution_over_labels = []
        for i in range(len(hatespeech_data)):
            label_counts = [
                hatespeech_data.iloc[i]["hate_speech"],
                hatespeech_data.iloc[i]["offensive_language"],
                hatespeech_data.iloc[i]["neither"],
            ]
            distribution_over_labels.append(np.array(label_counts) / sum(label_counts))
        hatespeech_data["label_distribution"] = distribution_over_labels

        human_prediction = []
        if self.expert_type == "synthetic":
            for i in range(len(hatespeech_data)):
                if hatespeech_data.iloc[i]["demographics"] == 0:
                    correct = np.random.choice([0, 1], p=[1 - self.synth_exp_param[0], self.synth_exp_param[0]])
                else:
                    correct = np.random.choice([0, 1], p=[1 - self.synth_exp_param[1], self.synth_exp_param[1]])
                human_prediction.append(
                    hatespeech_data.iloc[i]["class"] if correct else np.random.choice([0, 1, 2])
                )
        else:
            for i in range(len(hatespeech_data)):
                dist = hatespeech_data.iloc[i]["label_distribution"]
                human_prediction.append(np.random.choice([0, 1, 2], p=dist))

        hatespeech_data["human_prediction"] = human_prediction

        train_x = hatespeech_data["tweet"].to_numpy()
        train_y = torch.tensor(hatespeech_data["class"].to_numpy())
        train_h = torch.tensor(hatespeech_data["human_prediction"].to_numpy())
        train_d = torch.tensor(hatespeech_data["demographics"].to_numpy())

        random_seed = 42
        logging.info(f"Using random seed: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        embeddings = check_embeddings("data/hatespeech_embeddings.npy", train_x)
        train_x = torch.from_numpy(np.array(embeddings)).float()

        self.d = train_x.shape[1]

        test_size = int(self.test_split * len(train_x))
        val_size = int(self.val_split * len(train_x))
        train_size = len(train_x) - test_size - val_size

        train_x, val_x, test_x = torch.utils.data.random_split(
            train_x, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(train_y, [train_size, val_size, test_size])
        train_h, val_h, test_h = torch.utils.data.random_split(train_h, [train_size, val_size, test_size])
        train_d, val_d, test_d = torch.utils.data.random_split(train_d, [train_size, val_size, test_size])

        self.data_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_x.dataset[train_x.indices],
                train_y.dataset[train_y.indices],
                train_h.dataset[train_h.indices],
                train_d.dataset[train_d.indices],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )

        self.data_val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                val_x.dataset[val_x.indices],
                val_y.dataset[val_y.indices],
                val_h.dataset[val_h.indices],
                val_d.dataset[val_d.indices],
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.data_test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                test_x.dataset[test_x.indices],
                test_y.dataset[test_y.indices],
                test_h.dataset[test_h.indices],
                test_d.dataset[test_d.indices],
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
