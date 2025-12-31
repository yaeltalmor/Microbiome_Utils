import torch
print(torch.cuda.is_available())
import torch.nn as nn
import math
import torch.nn.functional as F
from enum import Enum

class TransformerTestUtils:
    """Static utility functions for validating Transformer building blocks."""

    @staticmethod
    def validate_value_mutex(value_type_mask, value_idx, e_v_cat, e_v_num, eps=1e-9):
        """
        Checks mutual exclusivity between categorical and numeric embeddings,
        while allowing a 'missing' state.
        In case I still plan to actively mask instead of relying on embeddings zeros add in the forward method:
        # value_type_mask = value_type_mask.unsqueeze(-1).float()
        # e_v = (1 - value_type_mask) * e_v_cat + value_type_mask * e_v_num
        """
        # 1) Sanity: mask must be binary
        assert ((value_type_mask == 0) | (value_type_mask == 1)).all(), \
            "value_type_mask must be binary (0 or 1)."

        is_numeric = (value_type_mask == 1)              # (B, T)
        has_cat    = (value_idx != 0)                    # (B, T)
        is_missing = (~is_numeric) & (~has_cat)          # (B, T)
        is_categ   = (~is_numeric) & (has_cat)           # (B, T)

        # 2) Detect nonzeros
        cat_nonzero = (e_v_cat.abs().sum(dim=-1) > eps)  # (B, T)
        num_nonzero = (e_v_num.abs().sum(dim=-1) > eps)  # (B, T)

        # 3) Rules per state
        # Numeric: only numeric nonzero
        assert (~cat_nonzero[is_numeric]).all(), \
            "Categorical embedding must be zero where mask==1 (numeric)."
        assert (num_nonzero[is_numeric]).all(), \
            "Numeric embedding must be nonzero where mask==1 (numeric)."

        # Categorical: only categorical nonzero
        assert (cat_nonzero[is_categ]).all(), \
            "Categorical embedding must be nonzero where mask==0 and value_idx>0."
        assert (~num_nonzero[is_categ]).all(), \
            "Numeric embedding must be zero where mask==0 and value_idx>0."

        # Missing: both zero
        assert (~cat_nonzero[is_missing]).all() and (~num_nonzero[is_missing]).all(), \
            "Both embeddings must be zero for missing (mask==0 and value_idx==0)."

# ------------------------- TRANSFORMER BUILDING BLOCKS ------------------
# ------------------ EVENT EMBEDDING LAYER ------------------

class ContinuousValueEmbedding(nn.Module):
    """
    STraTS Continuous Value Embedding (CVE) technique -
    FFN for numeric values initial embeddings: 1 input neuron → cve_hidden_size → d output, tanh activation.
    if cve_hidden_size is 0, use a single linear layer 1 → d_model without hidden layer or activation.
    """
    def __init__(self, d_model: int, cve_hidden_size: int = None):
        super().__init__()
        if cve_hidden_size>0:
            self.ffn = nn.Sequential(
                nn.Linear(1, cve_hidden_size),
                nn.Tanh(),
                nn.Linear(cve_hidden_size, d_model)
            )
        if cve_hidden_size == 0:
            self.ffn = nn.Sequential(
            nn.Linear(1, d_model)
        )

    def forward(self, numeric_values, mask=None):
        """
        numeric_values: (batch, seq_len) float numerical event values tensor, with zero-padded missing values (for either categorical values or padded events in sequence)
        mask:           (batch, seq_len) binary mask for numerical entries
        """
        numeric_values = numeric_values.unsqueeze(-1)  # (batch, seq_len, 1)
        out = self.ffn(numeric_values)
        # explicitly enforce zero embeddings for missing numeric values.
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out

class EventEmbAggregateMethod(Enum):
    CONCAT = "concat"
    SUM = "sum"

class HybridEventEmbedding(nn.Module):
    def __init__(self, num_event_tokens, num_value_tokens, d_model, cve_hidden_size,
                 event_emb_agg_method: EventEmbAggregateMethod = EventEmbAggregateMethod.SUM):
        super().__init__()
        # In PyTorch, you can set padding_idx in nn.Embedding so that the missing index always maps to a zero vector.
        self.event_emb = nn.Embedding(num_event_tokens, d_model, padding_idx=0)
        self.value_emb_cat = nn.Embedding(num_value_tokens, d_model, padding_idx=0)
        self.value_emb_num = ContinuousValueEmbedding(d_model, cve_hidden_size)
        self.event_emb_agg_method = event_emb_agg_method

    def forward(self, event_idx, value_idx, numeric_value, value_type_mask=None):
        """
        event_idx:     categorical event IDs        (batch, seq_len)
        value_idx:     categorical value IDs        (batch, seq_len): [[None:0, drugA=1, drugB=2, ..],...,] - a dedicated "missing value" token.
        numeric_value: numerical values             (batch, seq_len)
        value_type_mask: mask for numeric values    (batch, seq_len) binary
        """
        e_f = self.event_emb(event_idx)  # (batch, seq_len, d_model)
        # any 0 in the value_idx tensor will produce a zero vector for e_v_cat
        e_v_cat = self.value_emb_cat(value_idx)  # categorical value embedding
        e_v_num = self.value_emb_num(numeric_value, mask=value_type_mask)


        # Ensure exactly one nonzero source per position -
        # Assert values categorical and numerical embeddings are mutually exclusive
        TransformerTestUtils.validate_value_mutex(value_type_mask, value_idx, e_v_cat, e_v_num)

        # Combine categorical and numeric embeddings
        e_v = e_v_cat + e_v_num  # sum where one operand is zero, by value type
        if self.event_emb_agg_method == EventEmbAggregateMethod.CONCAT:
            e_i = torch.cat([e_f, e_v], dim=-1)
        elif self.event_emb_agg_method == EventEmbAggregateMethod.SUM:
            e_i = e_f + e_v
        return e_i

# ------------------ TEMPORAL POSITIONAL EMBEDDING LAYER ------------------
class TimeEmbeddingType(Enum):
    REL_POS_ENC = "rel_pos_enc"
    CVE = "cve"
    TIME2VEC = "time2vec"  # Kazemi et al. 2019

class TimePositionalEncoding(nn.Module):
    """
    Positional Encoding that uses real time values (t_values in months) instead of just indices,
    and concatenates to the original event embeddings.
    """
    def __init__(self, temp_d_model: int, emb_type: TimeEmbeddingType = TimeEmbeddingType.REL_POS_ENC,
                 add_nonperiodic: bool = True):
        super().__init__()
        self.temp_d_model = temp_d_model # The concatenated temporal encodings dimension
        self.emb_type = emb_type
        self.add_nonperiodic = add_nonperiodic

        if emb_type == TimeEmbeddingType.REL_POS_ENC and self.add_nonperiodic:
            assert self.temp_d_model > 2, "temp_d_model must be >2 to add non-periodic encodings."
            self.temp_d_model -= 2

        if emb_type == TimeEmbeddingType.TIME2VEC:
            # 1 non-periodic + (d-1) periodic
            self.omega = nn.Parameter(torch.randn(temp_d_model - 1))  # frequencies
            self.phi = nn.Parameter(torch.randn(temp_d_model - 1))  # phases
            self.omega_lin = nn.Parameter(torch.randn(1))  # linear freq
            self.phi_lin = nn.Parameter(torch.randn(1))  # linear phase

    def forward(self, t_values: torch.Tensor, mask: torch.Tensor):
        """
        t_values: (batch_size, seq_len) - actual months since diagnosis
        src_mask: (B, L) boolean mask (True = valid, False = padding)
        returns: (B, L, d_model) temporal embeddings, with padding zeroed out
        """
        mask = mask.float()  # (B, L) 1=keep, 0=pad
        if self.emb_type == TimeEmbeddingType.REL_POS_ENC:
            # Expand t_values to shape (batch, seq_len, 1)
            position = t_values.unsqueeze(-1)  # now (batch, seq_len, 1)
            # Compute div_term for sinusoidal encoding
            div_term = torch.exp(
                torch.arange(0, self.temp_d_model, 2, dtype=torch.float, device=t_values.device)
                * -(math.log(10000.0) / self.temp_d_model)
            )

            # Initialize positional encoding tensor - add temp_d_model dimension
            pe = torch.zeros(t_values.size(0), t_values.size(1), self.temp_d_model, device=t_values.device)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)

            if self.add_nonperiodic:
                t_norm = (t_values / (t_values.max() + 1e-6)).unsqueeze(-1)  # normalize 0–1
                t2_norm = (t_norm ** 2)
                pe = torch.cat([pe, t_norm, t2_norm], dim=-1)  # (B, L, d_model+2)
                assert pe.shape[-1] == self.temp_d_model + 2, "Temporal PE shape mismatch after adding non-periodic."

            # zero out padded positions
            pe = pe * mask.unsqueeze(-1)
            return pe  # shape: (batch, seq_len, d_model)

        elif self.emb_type == TimeEmbeddingType.TIME2VEC:
            # ----- Time2Vec -----
            t = t_values.unsqueeze(-1)  # (B,L,1)
            # Non-periodic (linear)
            lin = self.omega_lin * t + self.phi_lin  # (B,L,1)
            # Periodic (learnable sinusoids)
            per = torch.sin(t * self.omega + self.phi)  # (B,L,d-1)
            assert per.shape[-1] == self.temp_d_model - 1, "Time2Vec periodic shape mismatch."
            out = torch.cat([lin, per], dim=-1)  # (B,L,d)
            # zero out padded positions
            out = out * mask.unsqueeze(-1)

            return out


# ------------------ METADATA EMBEDDING LAYER ------------------

class MetadataEmbedding(nn.Module):
    """
    2-layer tanh FFN for static metadata features (as in STraTS).
    Maps raw tokenized MD d ∈ R^{input_dim} → embedding e_d ∈ R^{d_model}.
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        hidden_dim = 2 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model),
            nn.Tanh()
        )

    def forward(self, d):
        """
        d: (B, input_dim) raw metadata features (already numericized / one-hot)
        """
        return self.ffn(d)   # (B, d_model)


# ------------------ MULTI-HEAD ATTENTION EMBEDDING LAYER ------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_head."
        self.num_heads = num_heads  # nu. of attention heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads  # head_dim is the embedding dimensions each head will process.
        # bias=False: no impact on performance while reducing complexity (only for inputs)
        self.query_linear = nn.Linear(d_model, d_model, bias=False)  # W_Q
        self.key_linear = nn.Linear(d_model, d_model, bias=False)  # W_K
        self.value_linear = nn.Linear(d_model, d_model, bias=False)  # W_V
        self.output_linear = nn.Linear(d_model, d_model)
        # dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        '''splits the query, key, and values tensors
        between the heads and transforms them into shape:
        batch_size, num_heads, seq_length, head_dim.'''
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None):
        """
        Computes scaled dot-product attention with optional masking.
        query, key, value: (B, n_heads, L, head_dim)
        mask: (B, L) where 1=keep, 0=pad
        """
        # query <batch, n_heads, L, head_dim>
        # key.transpose(-2,-1) <batch, n_heads, head_dim, L>
        # scores  <batch, n_heads, L, L> = attention weights between every pair of tokens for each head.
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: (B, L) → (B, 1, 1, L) → (B, n_heads, L, L)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            mask_expanded = mask_expanded.expand(-1, self.num_heads, query.size(2), -1)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # --- Softmax ---
        attention_weights = F.softmax(scores, dim=-1)

        # Fix NaNs: happens if an entire row was masked (all -inf → NaN)
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )

        # --- Sanity checks ---
        assert attention_weights.shape == scores.shape, "Shape mismatch in attention"
        assert not torch.isnan(attention_weights).any(), "NaN in attention weights"
        assert not torch.isinf(attention_weights).any(), "Inf in attention weights"

        # Dropout for regularization
        attention_weights = self.attn_dropout(attention_weights)

        # weighted sum of value vectors based on how much attention each query pays to each key,
        # in parallel for every head and every batch.
        return torch.matmul(attention_weights, value)  # (B, n_heads, L, head_dim)

    def combine_heads(self, x, batch_size):
        '''transforms the attention weights back into the original embedding shape.'''
        x = x.permute(0, 2, 1, 3).contiguous()  # <batch,n_seq, n_heads, head_dim>
        return x.view(batch_size, -1, self.d_model)  # <batch, n_seq, d_model>

    def forward(self, query, key, value, mask=None):
        '''In the forward method, we split the query, key, and value tensors across the heads
        and compute the attention weights. The weights are combined and passed through
        the output layer to obtain the updated token embeddings projected into the original dimensionality.'''
        batch_size = query.size(0)

        # apply the attention weights on input to generate Q,K,V, and split them between the heads
        query = self.split_heads(self.query_linear(query), batch_size)  # Q=X*W_Q
        key = self.split_heads(self.key_linear(key), batch_size)  # K=X*W_K
        value = self.split_heads(self.value_linear(value), batch_size)  # V=X*W_V

        # weighted values
        attn_output = self.compute_attention(query, key, value, mask)

        output = self.combine_heads(attn_output, batch_size)

        return self.output_linear(output)  # concatenate and project head outputs

# ------------------------- ENCODER-CLASSIFIER TRANSFORMER ------------------
# ------------------ ENCODER LAYER ------------------
class FeedForwardSubLayer(nn.Module):
    '''
    Our FeedForwardSublayer class contains two fully connected linear layers separated by a ReLU activation.
    This is a position-wise feedforward network — it processes each position in the sequence independently
    (same weights applied across all positions).
    expanding each token's representation into a higher-dimensional space (d_ff) to capture richer patterns,
    then projecting it back to d_model. transforming the embeddings before pooling.
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # we use a dimension d_ff between linear layers, to further facilitate capturing complex patterns.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        # applies the forward pass to the attention mechanism outputs, passing them through the layers.
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff, dropout=dropout)
        # layer normalizations for keeping the scales and variances of
        # input embeddings consistent before and after the feed forward layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropouts are also used to regularize and stabilize training.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        '''the input embeddings are passed as the query, key, and value matrices,
        and a mask is used to prevent the processing of padding tokens in the input sequence.'''
        attn_output = self.self_attn(x, x, x, src_mask)  # (B, L, d_model)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)                  # (B, L, d_model)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# ------------------ CLASSIFIER HEAD ------------------
class ClassifierHead(nn.Module):
    """
    MLP head for binary classification.  Learns non-linear combinations of the pooled features.
    - Returns logits during training (use BCEWithLogitsLoss)
    - Optionally returns probabilities at inference (sigmoid)
    """
    def __init__(self, d_model, hidden_dim=64, dropout=0.1):
        "recommended hidden_dim = d_model//2"
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)   # single logit for binary task
        )

    def forward(self, x, return_probs: bool = False):
        logits = self.net(x)  # (B, 1)
        if return_probs:
            return torch.sigmoid(logits)  # (B, 1) probabilities in [0,1]
        return logits  # raw logits for BCEWithLogitsLoss

class EventTimeEmbAggregateMethod(Enum):
    CONCAT = "concat"
    SUM = "sum"

# ------------------ TEMPORAL TRANSFORMER CLASSIFIER ------------------
class TemporalTransformerClassifier(nn.Module):
    """
    Unites: HybridEventEmbedding + TimePositionalEncoding + Transformer Encoder + Pooling + MLP head.
    """
    def __init__(self,
                 num_event_tokens: int,
                 num_value_tokens: int,
                 event_d_model: int = 64,       # event embedding size
                 temp_d_model: int = 32,        # temporal encoding size
                 num_layers: int = 1,           # keep it light by default
                 num_heads: int = 1,            # single head by default
                 d_ff: int = 128,               # FF width per encoder layer
                 dropout: float = 0.1,
                 pooling: str = "max",          # "max" | "mean" | "attention"
                 cls_hidden: int = 64,          # MLP head hidden size
                 cve_num_val_hidden: int = 4,      # CVE hidden layer size
                 metadata_dim: int = None,
                 md_token: bool = False,          # if True → prepend metadata embedding as [MD] token to sequence
                 event_emb_agg_method: EventEmbAggregateMethod = EventEmbAggregateMethod.SUM,
                 event_time_emb_agg_method: EventTimeEmbAggregateMethod = EventTimeEmbAggregateMethod.CONCAT,
                 time_emb_type: TimeEmbeddingType = TimeEmbeddingType.REL_POS_ENC): # if provided → enable metadata embedding
        super().__init__()

        # Combined model dimension after concat
        if event_time_emb_agg_method == EventTimeEmbAggregateMethod.CONCAT:
            self.d_model = event_d_model + temp_d_model
        elif event_time_emb_agg_method == EventTimeEmbAggregateMethod.SUM:
            self.d_model = event_d_model if event_emb_agg_method == EventEmbAggregateMethod.SUM \
                else 2 * event_d_model
        self.event_time_emb_agg_method = event_time_emb_agg_method

        self.hybrid_emb = HybridEventEmbedding(num_event_tokens, num_value_tokens, event_d_model,
                                               cve_num_val_hidden, event_emb_agg_method)

        if time_emb_type == TimeEmbeddingType.REL_POS_ENC:
            self.time_emb = TimePositionalEncoding(temp_d_model, emb_type=TimeEmbeddingType.REL_POS_ENC,
                                                   add_nonperiodic=True)
        elif time_emb_type == TimeEmbeddingType.CVE:
            self.time_emb = ContinuousValueEmbedding(temp_d_model, int(math.sqrt(self.d_model)))
        elif time_emb_type == TimeEmbeddingType.TIME2VEC:
            self.time_emb = TimePositionalEncoding(temp_d_model, emb_type=TimeEmbeddingType.TIME2VEC)

        self.time_emb_type = time_emb_type



        self.layers = nn.ModuleList(
            [EncoderLayer(self.d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        assert pooling in ["max", "mean", "attention"], "Invalid pooling type"
        self.pooling_type = pooling
        if pooling == "attention":
            # Learnable vector to score each time step; mask will be applied before softmax
            self.attn_vector = nn.Parameter(torch.randn(self.d_model))

        # demographics branch (optional)
        if metadata_dim is not None:
            # TODO: if used in concat instead of sum extend classifier input dim
            self.md_emb = MetadataEmbedding(metadata_dim, self.d_model//2 if not md_token else self.d_model)
            self.md_token = md_token
        else:
            self.md_emb = None

        self.classifier = ClassifierHead(self.d_model if md_token else self.d_model+self.d_model//2, hidden_dim=cls_hidden, dropout=dropout)

    def _pool(self, x: torch.Tensor, src_mask: torch.Tensor):
        """
        x: (B, L, d_model)
        src_mask: (B, L) boolean; True=valid token, False=padding
        returns: (B, d_model)
        """
        if self.pooling_type == "max":
            # For max pooling, mask paddings by setting them to a very low value
            masked_x = x.masked_fill(~src_mask.unsqueeze(-1), float('-inf'))
            pooled = torch.max(masked_x, dim=1).values
            # In case a row is all -inf (fully padded), replace with zeros
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled

        if self.pooling_type == "mean":
            mask_exp = src_mask.unsqueeze(-1).float()             # (B, L, 1)
            summed = torch.sum(x * mask_exp, dim=1)               # (B, d_model)
            counts = mask_exp.sum(dim=1).clamp(min=1e-6)          # avoid div-by-zero
            return summed / counts

        # attention pooling
        # scores: (B, L) = (B, L, d_model) · (d_model,)
        scores = torch.matmul(x, self.attn_vector)                # (B, L)
        scores = scores.masked_fill(~src_mask, float('-inf'))     # ignore paddings
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)      # (B, L, 1)
        return torch.sum(x * weights, dim=1)                      # (B, d_model)

    def forward(self,
                event_idx: torch.Tensor,
                value_idx: torch.Tensor,
                numeric_value: torch.Tensor,
                value_type_mask: torch.Tensor,
                t_values: torch.Tensor,
                src_mask: torch.Tensor,
                metadata_idx: torch.Tensor = None,
                inference: bool = False,):
        """
        Inputs:
          - event_idx, value_idx: (B, L)
          - numeric_value: (B, L)
          - value_type_mask: (B, L)
          - t_values: (B, L)
          - src_mask: (B, L) boolean mask (True = valid, False = pad)
          - metadata_idx: (B, D) optional static metadata vector tokens per patient
          - inference: if True → return probabilities; else logits (for BCEWithLogitsLoss)
        Returns:
          - logits (B,1) if inference=False
          - probabilities (B,1) if inference=True
        """
        # (1) Embed events/values
        e_fv = self.hybrid_emb(event_idx, value_idx, numeric_value, value_type_mask)  # (B, L, event_d_model)

        # (2) Temporal encodings
        # e_t = self.time_emb(t_values, mask= src_mask)                                                  # (B, L, temp_d_model)

        # (3) Concatenate
        if self.event_time_emb_agg_method == EventTimeEmbAggregateMethod.CONCAT:
            # x = torch.cat([e_fv, e_t], dim=-1)
            x = e_fv
        elif self.event_time_emb_agg_method == EventTimeEmbAggregateMethod.SUM:
            x = e_fv# + e_t
        # (3b) Optionally prepend metadata as [MD] token
        if (self.md_emb is not None) and self.md_token and (metadata_idx is not None):
            e_md = self.md_emb(metadata_idx)  # (B, d_model)
            e_md = e_md.unsqueeze(1)  # (B, 1, d_model) to act as token
            x = torch.cat([e_md, x], dim=1)  # (B, L+1, d_model)
            # update mask to cover new token
            src_mask = torch.cat(
                [torch.ones_like(src_mask[:, :1], dtype=src_mask.dtype), src_mask],
                dim=1
            )

        # (4) Transformer encoder
        for layer in self.layers:
            x = layer(x, src_mask)                                                    # (B, L, d_model)

        # (5) Pool to patient-level
        pooled = self._pool(x, src_mask)                                              # (B, d_model)

        # # (6) Add Metadata embedding
        if (self.md_emb is not None) and not self.md_token and (metadata_idx is not None):
            e_md = self.md_emb(metadata_idx)                                  # (B, d_model)
            # pooled = pooled + e_md                                          # Option A: additive
            pooled = torch.cat([pooled, e_md], dim=-1)                # Option B: concat #TODO: that's the Strats option
        # (7) Classifier head
        out = self.classifier(pooled, return_probs=inference)                          # (B,1) logits or probs
        return out

# # ---------------EXAMPLE USAGE---------------
# model = TemporalTransformerClassifier(
#     num_event_tokens=4, num_value_tokens=3,
#     event_d_model=64, temp_d_model=32,
#     num_layers=1, num_heads=1, d_ff=256,
#     dropout=0.1, pooling="attention",  # or "max"/"mean"
#     cls_hidden=64, metadata_dim=None
# )
#
# # ==== Toy test ====
#
# # 1. Build vocab for events and categorical values
# event_vocab = {None: 0, "drug_A": 1, "drug_B": 2, "lab_test": 3}
# value_vocab = {None: 0, "aspirin": 1, "paracetamol": 2}  # categorical values only
#
# # 2. Fake batch of token indices
# event_idx = torch.tensor([
#     [1, 2, 3, 0],   # patient 1
#     [2, 3, 0, 0]    # patient 2
# ], dtype=torch.long)
#
# value_idx = torch.tensor([
#     [1, 2, 0, 0],   # categorical values for patient 1
#     [1, 0, 0, 0]    # categorical values for patient 2
# ], dtype=torch.long)
#
# # 3. Numeric values (aligned with events)
# numeric_value = torch.tensor([
#     [0.0, 0.0, 1.2, 0.0],
#     [0.0, 0.0, 0.0, 0.0]
# ], dtype=torch.float32)
#
# # 4. Mask: 1 if numeric, 0 if categorical/missing
# value_type_mask = torch.tensor([
#     [0, 0, 1, 0],
#     [0, 1, 0, 0]
# ], dtype=torch.float32)
#
# # 5. Temporal indices (months since diagnosis)
# t_values = torch.tensor([
#     [-8, 11, 26, 0],
#     [-1, 100, 0, 0]
# ], dtype=torch.long)
#
# # 6. metadata encodings
# # 1-hot vector: [<main_diagnosis_Ulcerative_Colitis, gender_female, visit_child_department_Yes, smoking_No, smoking_missing>]
# metadata_idx = torch.tensor([
#     [0, 1, 1, 1, 0], # non-smoking crohns girl
#     [1, 0, 0, 1, 1]  # smoking colitis male adult
# ], dtype=torch.float32)
#
# # 6. Padding mask (1 = keep, 0 = pad)
# src_mask = (event_idx != 0).bool()   # shape (B, L)
#
# # 7. Dummy binary labels (drug switch yes/no)
# labels = torch.tensor([1, 0], dtype=torch.float32)  # (B,)
#
# # --------------TRAINING LOOP EXAMPLE--------------
# # 7. Dummy binary labels (drug switch yes/no)
# labels = torch.tensor([1, 0], dtype=torch.float32)  # (B,)
#
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # optimizer for toy run - faster learning rate
# criterion = nn.BCEWithLogitsLoss() # loss function
#
# # ==== Training loop (toy test: 3 epochs) ====
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train() # put model in training mode (activates dropout etc.)
#     optimizer.zero_grad()  # reset gradients from last step
#
#     logits = model(event_idx, value_idx, numeric_value, value_type_mask, t_values, src_mask, metadata_idx, inference=False)
#
#     # --- Core asserts (shape + NaN/Inf) ---
#     assert logits.shape == (event_idx.size(0), 1), f"Expected (B,1), got {logits.shape}"
#     assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
#
#     # compute loss
#     loss = criterion(logits.squeeze(-1), labels)
#
#     # backpropagate gradients
#     loss.backward()
#
#     # quick gradient sanity check
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             assert param.grad is not None, f"No gradient for {name}"
#             assert torch.isfinite(param.grad).all(), f"Bad gradient in {name}"
#     # update weights
#     optimizer.step()
#
#     print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
#
# # ==== Inference (probabilities) ====
# model.eval()
# with torch.no_grad():
#     probs = model(event_idx, value_idx, numeric_value, value_type_mask, t_values, src_mask, metadata_idx, inference=True)
#
#     print("Predicted probabilities:", probs)
