import os

import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
utils.set_seed(42)
N_TOKENS = 5
RESOLUTION = 14
train_tokens, train_labels, test_tokens, test_labels, token_vals = utils.load_tokenized_mnist(N_TOKENS, RESOLUTION)

class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Batch First, unnecassarily convoluted but works as expected
        x = x + self.pe[: x.size(1), :].view(1, x.size(1), -1)
        return self.dropout(x)

class Unmasker(nn.Module):
    def __init__(self, vocab_size,n_dim, mask_id=None, embed_dim=128, num_heads=4, num_layers=4, hidden_dim=256, dropout=0.1, ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size - 1 if mask_id is None else mask_id
        self.neg_infinity = -1e9

        self.n_dim = n_dim
        self.tim_embed = TimestepEmbedder(embed_dim, n_dim)
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, dropout=dropout, max_len=n_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, xt, t):
        """
        x: (batch_size, n_dim) Torch.LongTensor
        t: (batch_size) Torch.FloatTensor

        returns: (batch_size, n_dim, vocab_size)
        """
        xt_emb = self.pos_embed(self.tok_embed(xt)) + self.tim_embed(t)[:,None]
        xt_emb = self.encoder(xt_emb)
        logits = self.output_layer(xt_emb) # (batch_size, n_dim, vocab_size)
        return self._subs_param(logits, xt)
    
    def _subs_param(self, logits, xt):
        logits[:,:,self.mask_id] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1,keepdim=True) # idk why is this required
        unmasked_indices = (xt != self.mask_id)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

class MaskingScheduler:
    def __init__(self, mode = "linear"):
        self.mode = mode
        self.eps = 1e-6
        if mode == "linear":
            self.init_linear_scheduler()
        else:
            raise NotImplementedError
        
    def init_linear_scheduler(self):
        self.alpha = lambda t : 1 - t
        self.ce_weight = lambda t : -1/(t + self.eps) 

    @torch.no_grad()
    def get_alpha(self, t):
        return self.alpha(t)
    
    @torch.no_grad()
    def get_ce_weight(self, t):
        return self.ce_weight(t)


def compute_loss(x, model, masking_sch, device):
    t = torch.rand(x.shape[0], dtype=torch.float, device=device)
    alpha_t = masking_sch.get_alpha(t).to(device)
    ce_weight_t = masking_sch.get_ce_weight(t).to(device)
    masks = torch.rand_like(x, dtype=torch.float, device=device) < 1 - alpha_t[:,None]
    masked_x = torch.where(masks, torch.tensor(model.mask_id),x)
    logits = model(masked_x, t)
    logprobs = torch.log_softmax(logits, dim=-1)
    B_select = torch.arange(logprobs.shape[0]).reshape(-1,1)
    L_select = torch.arange(logprobs.shape[1])
    out_select = logprobs[B_select,L_select, x]
    loss = (out_select.sum(dim=-1) * ce_weight_t).mean()
    return loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
model = Unmasker(N_TOKENS + 1, embed_dim=32, num_heads=8, num_layers=8, hidden_dim=16, dropout=0.3, n_dim = RESOLUTION**2).to(device)
# model = Unmasker(N_TOKENS + 1, embed_dim=64, num_heads=8, num_layers=16, hidden_dim=62, dropout=0.3, n_dim = RESOLUTION**2).to(device)
masking_sch = MaskingScheduler()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
N_EPOCHS = 800 #400
BATCH_SIZE = 2048 #1024
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tokens), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_tokens), batch_size=BATCH_SIZE, shuffle=False)
losses = {
    "train": [],
    "test" : []
}
for epoch in range(N_EPOCHS):
    model.train()
    epoch_train_loss = 0 
    batch_count = 0
    for i, (tokens,) in enumerate(train_loader):
        tokens = tokens.to(device)
        loss = compute_loss(tokens, model, masking_sch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * len(tokens)
        batch_count += len(tokens)

    losses["train"].append(epoch_train_loss / batch_count)
    epoch_test_loss = 0
    test_batch_count = 0
    eval_every_epoch = 1
    if epoch % eval_every_epoch == 0:
        with torch.no_grad():
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/unmasker_{N_TOKENS}_{RESOLUTION}x{RESOLUTION}.pt")
            model.eval()
            for i, (tokens,) in enumerate(test_loader):
                tokens = tokens.to(device)
                loss = compute_loss(tokens, model, masking_sch, device)
                epoch_test_loss += loss.item() * len(tokens)
                test_batch_count += len(tokens)
            losses["test"].append(epoch_test_loss / test_batch_count)
    print(f"Epoch {epoch: 03}, Train {losses['train'][-1]:.03f}, Test {losses['test'][-1]:.03f}")


        # Save losses
torch.save(losses, 'saved_models/losses.pt')
# Save the final model
torch.save(model.state_dict(), f"saved_models/final_unmasker_{N_TOKENS}_{RESOLUTION}x{RESOLUTION}.pt")




