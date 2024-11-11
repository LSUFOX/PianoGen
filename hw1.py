import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random

from google_drive_downloader import GoogleDriveDownloader as gdd
import pretty_midi
from torch.utils.data import DataLoader, TensorDataset

from midi2seq import process_midi_seq, seq2piano
from midi2seq import dim
from model_base import ComposerBase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is", device)

class Composer(ComposerBase, nn.Module):

    def __init__(self, seq_len=50, d_model=256, num_heads=8, num_layers=6, dim_ff=512, vocab_size=382, load_trained=False):
        # Initialize
        ComposerBase.__init__(self, load_trained)
        nn.Module.__init__(self)

        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer Encoder with batch_first=True for better performance
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.loss_function = nn.CrossEntropyLoss()

        if load_trained:
            file_id = '1BPZmsIz_5GNgf0uxa8cykklnF3C8ifXB'
            dest_path = './trained_transformer_weights.pth'
            # https://drive.google.com/file/d/1BPZmsIz_5GNgf0uxa8cykklnF3C8ifXB/view?usp=sharing
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, unzip=False)
            print('Trained Model Loaded')
            self.load_state_dict(torch.load(dest_path, map_location=torch.device(device)))
        else:
            print("Let's train a new model")

    def forward(self, x):
        # Embedding and positional encoding
        emb = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(emb)
        
        # Predict the next token for each position in the sequence
        logits = self.fc(transformer_out)
        
        return logits

    def train(self, x):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
        optimizer.zero_grad()

        # Make sure input size is [seq_len, batch_size]
        x = x.to(next(self.parameters()).device).long()  # make sure same device
        x = x.permute(1, 0)  # Adjust to [seq_len, batch_size]

        output = self.forward(x)
        
        loss = self.loss_function(output.reshape(-1, output.size(-1)), x.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)  # 增加 max_norm 以减少梯度裁剪的影响
        optimizer.step()

        print(f"Training Loss: {loss.item()}")
        return loss

    def save_model(self, file_name='trained_weights.pth'):
        torch.save(self.state_dict(), file_name)
        print(f"Model weights succcesfully saved at {file_name}")

    def load_model(self, file_name='trained_weights.pth'):
        self.load_state_dict(torch.load(file_name))
        print(f"Model weights succcesfully loaded from {file_name}")

    def compose(self, n):
        generated_seq = []

        shift_tokens = list(range(128*2, 128*2 + 100))
        generated_seq.append(random.choice(shift_tokens))

        for i in range(1, n):
            x = torch.tensor(generated_seq, dtype=torch.long).unsqueeze(1).to(device)
            x = self.embedding(x)
            x = self.pos_encoding[:, :x.size(1), :]
            transformer_out = self.transformer_encoder(x)
            next_token_logits = self.fc(transformer_out[-1])
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1).item()

            # Make sure token size
            next_token = max(0, min(next_token, self.vocab_size - 1))
            generated_seq.append(next_token)

        return np.array(generated_seq)
