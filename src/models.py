import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


# Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


# There may be a bug in my implementation because it does not work well.
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))
        )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class GeMText(nn.Module):
    def __init__(self, dim=1, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(self, last_hidden_state, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        x = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling = self.mean_pooling(last_hidden_state, attention_mask)
        max_pooling = self.max_pooling(last_hidden_state, attention_mask)
        out = torch.cat([mean_pooling, max_pooling], dim=1)

        return out


class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm=256):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack(
            [all_hidden_states[layer_i][:, 0].squeeze() for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dr

        return out


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states=True)
            # robertaだと、max_position_embeddingsが514(512+2)になっているので、これを増やす。
            if cfg.model_name == "roberta-large" or cfg.model_name == "roberta-base" or "albert" in cfg.model_name:
                print(f"Roberta Model!, Change max_position_embeddings 514 -> {cfg.dataset.params.max_len}")
                self.config.update(
                    {
                        "position_embedding_type": None,
                        "max_position_embeddings": cfg.dataset.params.max_len,
                    }
                )

            self.config.hidden_dropout = 0.0
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_dropout = 0.0
            self.config.attention_probs_dropout_prob = 0.0

        else:
            self.config = torch.load(config_path)

        if pretrained:
            if cfg.model_name == "roberta-large" or cfg.model_name == "roberta-base" or "albert" in cfg.model_name:
                self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config, ignore_mismatched_sizes=True)
            else:
                self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel(self.config)

        # freeze layers
        if cfg.freeze_layers > 0:
            print(f"Freezing First {cfg.freeze_layers} Layers ...")
            for i, layer in enumerate(self.model.encoder.layer[: cfg.freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # reinit layers
        if cfg.reinit_layers > 0:
            print(f"Reinitializing Last {cfg.reinit_layers} Layers ...")
            for layer in self.model.encoder.layer[-cfg.reinit_layers :]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
            print("Done.!")

        # Pooling
        if self.cfg.pooling == "mean":
            self.pooling = MeanPooling()
        elif self.cfg.pooling == "max":
            self.pooling = MaxPooling()
        elif self.cfg.pooling == "min":
            self.pooling = MinPooling()
        elif self.cfg.pooling == "attention":
            self.pooling = AttentionPooling(self.config.hidden_size)
        elif self.cfg.pooling == "weightedlayer":
            self.pooling = WeightedLayerPooling(
                self.config.num_hidden_layers, layer_start=self.cfg.weightedlayer_start, layer_weights=None
            )
        elif self.cfg.pooling == "gem":
            self.pooling = GeMText()
        elif self.cfg.pooling == "meanmax":
            self.pooling = MeanMaxPooling()
        elif self.cfg.pooling == "lstm":
            self.pooling = LSTMPooling(self.config.num_hidden_layers, self.config.hidden_size)
        else:
            self.pooling = MeanPooling()

        # 今回は2個の連続値を予測するので、出力次元は2
        if self.cfg.pooling == "meanmax":
            self.fc = nn.Linear(self.config.hidden_size * 2, 2)
        else:
            self.fc = nn.Linear(self.config.hidden_size, 2)

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pooling(last_hidden_states, inputs["attention_mask"])

        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        outputs = self.fc(feature)

        return outputs
