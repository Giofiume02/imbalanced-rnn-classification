import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out):
        # rnn_out: (batch, seq_len, hidden_dim)
        scores = self.attn(rnn_out).squeeze(-1)         # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)          # (batch, seq_len)
        context = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights

class RecurrentClassifier(nn.Module):
    """
    Improved RNN classifier:
    Conv1D → LSTM/GRU → Attention → Dense
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            rnn_type='GRU',
            bidirectional=False,
            dropout_rate=0.2
    ):
        super().__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # ---------- 1) CONV1D FEATURE EXTRACTOR ----------
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        rnn_input_dim = 64

        # ---------- 2) RNN BLOCK ----------
        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        rnn_module = rnn_map[rnn_type]

        dropout_val = dropout_rate if num_layers > 1 else 0

        self.rnn = rnn_module(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        rnn_out_dim = hidden_size * (2 if bidirectional else 1)

        # ---------- 3) ATTENTION ----------
        self.attention = Attention(rnn_out_dim)

        # ---------- 4) CLASSIFIER ----------
        self.fc1 = nn.Linear(rnn_out_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

        # ---------- INIT ----------
        self.reset_parameters()


    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """

        # --- CONV BLOCK ---
        x = x.transpose(1, 2)            # → (batch, channels=input_size, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)            # → (batch, seq_len, channels=64)

        # --- RNN BLOCK ---
        rnn_out, hidden = self.rnn(x)

        # If LSTM, take only hidden state h_n
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]

        # --- ATTENTION ---
        context, attn_weights = self.attention(rnn_out)

        # --- CLASSIFIER ---
        x = F.relu(self.fc1(context))
        x = self.bn(x)
        logits = self.fc2(x)

        return logits   # (batch, num_classes)

    def reset_parameters(self):
        # ---- CONV ----
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)

        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)

        # ---- RNN ----
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

                # Forget gate bias for LSTM
                if self.rnn_type == 'LSTM':
                    hidden = self.hidden_size
                    if "bias_ih" in name:
                        param.data[hidden:2*hidden] = 1.0

        # ---- ATTENTION ----
        nn.init.xavier_uniform_(self.attention.attn.weight)
        nn.init.constant_(self.attention.attn.bias, 0.0)

        # ---- CLASSIFIER ----
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()

        # Alpha must be a vector of weights for each classes
        # if None → no weighting
        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha

        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def smooth_labels(self, targets, num_classes):
        with torch.no_grad():
            smoothed = torch.full(
                (targets.size(0), num_classes),
                self.smoothing / (num_classes - 1),
                device=targets.device
            )
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return smoothed

    def forward(self, inputs, targets):
        """
        inputs: logits (batch_size, num_classes)
        targets: class indices (batch_size)
        """
        num_classes = inputs.size(1)

        # -- Label smoothing
        targets_smoothed = self.smooth_labels(targets, num_classes)

        # -- Log-softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        # -- Focal factor
        focal_factor = (1 - probs) ** self.gamma

        # -- Apply class weights alpha
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)  # shape (num_classes,)
            alpha = alpha.unsqueeze(0)            # shape (1, num_classes)
            weighted_log_probs = alpha * targets_smoothed * focal_factor * log_probs
        else:
            weighted_log_probs = targets_smoothed * focal_factor * log_probs

        loss = -weighted_log_probs.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def recurrent_summary(model, input_size):
    """
    Summary function compatible with:
      - Conv1d
      - Linear
      - LSTM / GRU / RNN
      - BatchNorm / LayerNorm
      - Custom Attention layer (context + attn_weights)
    """

    output_shapes = {}
    hooks = []

    # -----------------------------
    # 1. Define robust hook
    # -----------------------------
    def get_hook(name):
        def hook(module, input, output):

            # ---- CASE A: Attention(context, weights) ----
            if isinstance(output, tuple) and len(output) == 2:
                context, weights = output

                try:
                    ctx_shape = list(context.shape)
                except:
                    ctx_shape = "N/A"

                try:
                    w_shape = list(weights.shape)
                except:
                    w_shape = "N/A"

                output_shapes[name] = f"context: {ctx_shape}, attn_weights: {w_shape}"
                return

            # ---- CASE B: RNN outputs (rnn_out, hidden) ----
            if isinstance(output, tuple):
                shapes = []
                for o in output:
                    try:
                        shapes.append(list(o.shape))
                    except:
                        shapes.append("N/A")
                output_shapes[name] = str(shapes)
                return

            # ---- CASE C: Standard layer ----
            try:
                shape = list(output.shape)
            except:
                shape = "N/A"

            output_shapes[name] = f"{shape}"

        return hook

    # -----------------------------
    # 2. Register hooks
    # -----------------------------
    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv1d,
                               nn.Linear,
                               nn.GRU,
                               nn.LSTM,
                               nn.RNN,
                               nn.BatchNorm1d,
                               nn.LayerNorm)):
            hooks.append(module.register_forward_hook(get_hook(name)))

        if module.__class__.__name__ == "Attention":
            hooks.append(module.register_forward_hook(get_hook(name)))

    # -----------------------------
    # 3. Forward pass with dummy input
    # -----------------------------
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    # -----------------------------
    # 4. PRINT SUMMARY
    # -----------------------------
    print("-" * 100)
    print(f"{'Layer (type)':<40} {'Output Shape':<40} {'Params':<10}")
    print("=" * 100)

    total_params = 0

    for name, module in model.named_modules():

        if name in output_shapes:

            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params

            print(f"{name:<40} {output_shapes[name]:<40} {params:<10}")

    print("=" * 100)
    print(f"Total trainable params: {total_params:,}")
    print("-" * 100)
