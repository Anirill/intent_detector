import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from app.main import EMBEDDING_SIZE, LSTM_HIDDEN_SIZE, DROPOUT
# from app.main import device
# from app.main import word_dict
from app.utils import eval_preprocess

EMBEDDING_SIZE = 256
LSTM_HIDDEN_SIZE = 256
DROPOUT = 0
VOCAB_SIZE = 2415
device = torch.device("cpu")


class IntentEnc(nn.Module):
    def __init__(self, embedding_size, lstm_hidden_size, vocab_size=VOCAB_SIZE):
        super(IntentEnc, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=DROPOUT
                            )

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, DROPOUT)
        x, _ = self.lstm(x)
        x = F.dropout(x, DROPOUT)
        return x


class IntentDec(nn.Module):
    # def __init__(self, lstm_hidden_size, label_size=len(label_num)):
    def __init__(self, lstm_hidden_size, label_size=54):
        super(IntentDec, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_hidden_size * 2,
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            num_layers=1
                            )  # , dropout=DROPOUT)
        self.fc = nn.Linear(lstm_hidden_size, label_size)

    def forward(self, x, real_len):
        batch = x.size()[0]
        real_len = torch.tensor(real_len).to(device)
        x = F.dropout(x, DROPOUT)
        x, _ = self.lstm(x)
        x = F.dropout(x, DROPOUT)

        index = torch.arange(batch).long().to(device)
        state = x[index, real_len - 1, :]

        res = self.fc(state.squeeze())
        return res


class Intent(nn.Module):
    def __init__(self):
        super(Intent, self).__init__()
        self.enc = IntentEnc(EMBEDDING_SIZE, LSTM_HIDDEN_SIZE).to(device)
        self.dec = IntentDec(LSTM_HIDDEN_SIZE).to(device)

    def detection(self, sentence, size=3):
        x = torch.tensor([eval_preprocess(sentence)]).to(device)
        real_len = len(x)
        h = self.enc(x)
        intent_logits = self.dec(h, real_len)
        log_intent_logits = F.log_softmax(intent_logits, dim=-1)
        top_n = torch.argsort(log_intent_logits, dim=-1, descending=True)[:size]
        top_n = np.array(top_n)
        #         float(log_intent_logits_test[i])
        return top_n, np.array([log_intent_logits[i].item() for i in top_n])
