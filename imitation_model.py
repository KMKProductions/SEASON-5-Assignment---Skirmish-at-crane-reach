"""The same small PyTorch policy learns from demonstrations and chooses actions in play.

Attention lets each token read other visible tokens. We write its projections explicitly
so you can follow the calculation from queries and keys to a weighted sum of values.
"""

from pathlib import Path

import torch
from encoder import ENTITY_FEATURES, MOVE_FEATURES, TARGET_FEATURES, VERSION
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = 1


class Attention(nn.Module):
    """Let each query row take a learned, weighted mixture of the context rows."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, queries, context, valid):
        def split(values):
            # Each head reads the same rows through its own smaller set of numbers.
            batch, count, _ = values.shape
            return values.reshape(batch, count, self.heads, self.head_width).transpose(1, 2)

        # Queries describe what a row looks for; keys describe what each context row offers.
        # Values carry the information we will mix after comparing queries with keys.
        q, k, v = split(self.query(queries)), split(self.key(context)), split(self.value(context))
        scores = (q @ k.transpose(-2, -1)) / self.head_width**0.5
        # Padding is only for batching. Give it zero attention weight, then make each
        # query's remaining weights sum to one with softmax.
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        weights = scores.softmax(dim=-1)
        # Take the weighted mixtures and join the heads back into one row per query.
        values = (weights @ v).transpose(1, 2).contiguous().flatten(2)
        return self.output(values)


class Block(nn.Module):
    """A transformer block: read other rows, then adjust each row with two learned layers.

    Normalization keeps the size of the numbers steady. Each addition keeps the previous
    representation and adds an update, which is called a residual connection.
    """

    def __init__(self, width, heads):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, eps=1e-5)
        self.attention = Attention(width, heads)
        self.norm_feedforward = nn.LayerNorm(width, eps=1e-5)
        self.expand = nn.Linear(width, 2 * width)
        self.contract = nn.Linear(2 * width, width)

    def forward(self, values, valid, context=None):
        normalized = self.norm_attention(values)
        # Entity rows read one another. Movement rows instead read the entity context.
        values = values + self.attention(normalized, normalized if context is None else context, valid)
        hidden = F.gelu(self.expand(self.norm_feedforward(values)), approximate="tanh")
        return values + self.contract(hidden)


class Policy(nn.Module):
    """Share entity context between a tile scorer and a target scorer."""

    def __init__(self, width=64, layers=2, heads=4):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (width, layers, heads)) or width % heads:
            raise ValueError("width, layers, and heads must be positive; width must be divisible by heads")
        self.config = dict(width=width, layers=layers, heads=heads)
        # If training offers no choices for a scorer, it keeps equal scores during play.
        self.head_available = dict(move=True, target=True)
        for head, features in (("move", MOVE_FEATURES), ("target", TARGET_FEATURES)):
            # Save training means and scales with the model; they stay fixed during play.
            self.register_buffer(f"{head}_mean", torch.zeros(len(features)))
            self.register_buffer(f"{head}_scale", torch.ones(len(features)))
        self.entity_projection = nn.Linear(len(ENTITY_FEATURES), width)
        self.encoder = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.entity_norm = nn.LayerNorm(width, eps=1e-5)
        self.move_projection = nn.Linear(len(MOVE_FEATURES), width)
        self.move_block = Block(width, heads)
        self.move_output = nn.Linear(width, 1)
        self.no_target = nn.Parameter(torch.zeros(width))
        self.target_hidden = nn.Linear(2 * width + len(TARGET_FEATURES), width)
        self.target_output = nn.Linear(width, 1)

    def encode(self, entities, entity_mask):
        """Turn entity features into context that both scorers can read."""
        if entities.shape[1] == 0 or not entity_mask[:, 0].all():
            raise ValueError("Every observation needs its acting-unit token at index 0")
        values = self.entity_projection(entities)
        for block in self.encoder:
            values = block(values, entity_mask)
        return self.entity_norm(values)

    def score_moves(self, context, entity_mask, options, candidate_mask):
        """Let each destination read the battlefield and produce one score."""
        if self.head_available["move"]:
            values = self.move_projection((options - self.move_mean) / self.move_scale)
            logits = self.move_output(self.move_block(values, entity_mask, context)).squeeze(-1)
        else:
            logits = options.new_zeros(options.shape[:2])
        # Padded options must never win or take probability away from real options.
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def score_targets(self, context, options, indices, candidate_mask):
        """Score each target using self, that enemy, and the chosen destination's features."""
        if ((indices < -1) | (indices >= context.shape[1]))[candidate_mask].any():
            raise ValueError("Target pointer is outside the entity tokens")
        if self.head_available["target"]:
            # gather looks up each named enemy. For -1, we first use a safe index,
            # then replace that row with the learned no-target representation.
            pointers = indices.clamp(min=0, max=context.shape[1] - 1)
            enemy = context.gather(1, pointers.unsqueeze(-1).expand(-1, -1, context.shape[-1]))
            enemy = torch.where((indices == -1).unsqueeze(-1), self.no_target, enemy)
            # The acting unit is always token 0. Pair it with every target option.
            own = context[:, :1].expand(-1, options.shape[1], -1)
            values = torch.cat((own, enemy, (options - self.target_mean) / self.target_scale), dim=-1)
            logits = self.target_output(F.gelu(self.target_hidden(values), approximate="tanh")).squeeze(-1)
        else:
            logits = options.new_zeros(options.shape[:2])
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def forward(
        self,
        entities,
        entity_mask,
        move_options,
        move_mask,
        target_options,
        target_mask,
        target_entity_indices,
    ):
        # Training supplies target features from the teacher's destination. During play,
        # choose_action calls the scorers in order and uses the learner's destination.
        context = self.encode(entities, entity_mask)
        return {
            "move": self.score_moves(context, entity_mask, move_options, move_mask),
            "target": self.score_targets(context, target_options, target_entity_indices, target_mask),
        }


def save_policy(model, path):
    """Save tensors and plain metadata, including the feature contract used to train them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "encoder_version": VERSION,
            "entity_features": list(ENTITY_FEATURES),
            "move_features": list(MOVE_FEATURES),
            "target_features": list(TARGET_FEATURES),
            "config": model.config,
            "head_available": model.head_available,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_policy(path):
    """Load for CPU play, checking that the saved model uses this encoder and architecture."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "model_version": MODEL_VERSION,
        "encoder_version": VERSION,
        "entity_features": list(ENTITY_FEATURES),
        "move_features": list(MOVE_FEATURES),
        "target_features": list(TARGET_FEATURES),
    }
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Model encoder or architecture differs; collect data and retrain")
    available = saved.get("head_available")
    if (
        not isinstance(available, dict)
        or set(available) != {"move", "target"}
        or any(type(value) is not bool for value in available.values())
    ):
        raise ValueError("Model has invalid head availability metadata")
    config = saved.get("config")
    if not isinstance(config, dict) or set(config) != {"width", "layers", "heads"}:
        raise ValueError("Model has incomplete architecture metadata; retrain")
    model = Policy(**config)
    model.load_state_dict(saved["state_dict"], strict=True)
    for head in available:
        if (
            not torch.isfinite(getattr(model, f"{head}_mean")).all()
            or not (
                torch.isfinite(getattr(model, f"{head}_scale")) & (getattr(model, f"{head}_scale") > 0)
            ).all()
        ):
            raise ValueError("Model has invalid feature normalization")
    model.head_available = available
    return model.eval()
