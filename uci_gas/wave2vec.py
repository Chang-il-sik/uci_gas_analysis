
import torch.nn as nn
import torch.nn.functional as F

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class Wav2VecModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.prediction_steps = cfg.prediction_steps
        offset = cfg.offset

        if cfg.activation == "relu":
            activation = nn.ReLU()
        elif cfg.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + cfg.activation)

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            log_compression=cfg.log_compression,
            skip_connections=cfg.skip_connections_feat,
            residual_scale=cfg.residual_scale,
            non_affine_group_norm=cfg.non_affine_group_norm,
            activation=activation,
        )
        embed = feature_enc_layers[-1][0]

        self.vector_quantizer = None
        if cfg.vq_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=embed,
                num_vars=cfg.vq_vars,
                temp=cfg.vq_temp,
                groups=cfg.vq_groups,
                combine_groups=cfg.combine_groups,
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
                time_first=False,
                activation=activation,
                weight_proj_depth=cfg.vq_depth,
                weight_proj_factor=2,
            )
        elif cfg.vq_type == "kmeans":
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=embed,
                num_vars=cfg.vq_vars,
                groups=cfg.vq_groups,
                combine_groups=cfg.combine_groups,
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
                time_first=False,
                gamma=cfg.vq_gamma,
            )
        else:
            assert (
                cfg.vq_type == "none" or cfg.vq_type is None
            ), "Unknown quantizer type"

        if cfg.offset == "auto":
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)

        offset = int(offset)

        def make_aggregator():
            if cfg.aggregator == "cnn":
                agg_layers = eval(cfg.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                feature_aggregator = ConvAggegator(
                    conv_layers=agg_layers,
                    embed=embed,
                    dropout=cfg.dropout,
                    skip_connections=cfg.skip_connections_agg,
                    residual_scale=cfg.residual_scale,
                    non_affine_group_norm=cfg.non_affine_group_norm,
                    conv_bias=not cfg.no_conv_bias,
                    zero_pad=cfg.agg_zero_pad,
                    activation=activation,
                )
            elif cfg.aggregator == "gru":
                agg_dim = cfg.gru_dim
                feature_aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=embed,
                        hidden_size=agg_dim,
                        num_layers=1,
                        dropout=cfg.dropout,
                    ),
                    TransposeLast(deconstruct_idx=0),
                )
            else:
                raise Exception("unknown aggregator type " + cfg.aggregator)

            return feature_aggregator, agg_dim

        self.feature_aggregator, agg_dim = make_aggregator()

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=cfg.prediction_steps,
            n_negatives=cfg.num_negatives,
            cross_sample_negatives=cfg.cross_sample_negatives,
            sample_distance=cfg.sample_distance,
            dropout=cfg.dropout,
            offset=offset,
            balanced_classes=cfg.balanced_classes,
            infonce=cfg.infonce,
        )

        self.dropout_feats = nn.Dropout(p=cfg.dropout_features)
        self.dropout_agg = nn.Dropout(p=cfg.dropout_agg)

        if cfg.project_features == "none":
            self.project_features = None
        elif cfg.project_features == "same":
            self.project_features = self.feature_aggregator
        elif cfg.project_features == "new":
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)

        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """Maximum length supported by the model."""
        return sys.maxsize

    def get_logits(self, net_output):
        logits = net_output["cpc_logits"]
        return logits

    def get_targets(self, sample, net_output):
        t = net_output["cpc_targets"]
        if isinstance(t, tuple):
            t = t[0]
        return t.contiguous()

    def get_target_weights(self, targets, net_output):
        targets = net_output["cpc_targets"]
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]
        return None

    def get_extra_losses(self, net_output):
        loss = None
        if "prob_perplexity" in net_output:
            loss = net_output["num_vars"] - net_output["prob_perplexity"]
        elif "kmeans_loss" in net_output:
            loss = net_output["kmeans_loss"]

        return loss

def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers,
        dropout,
        log_compression,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x
    