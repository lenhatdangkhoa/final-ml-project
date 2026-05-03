from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.use_local_branch = getattr(config, "use_local_branch", True)
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

        if self.use_local_branch:
            kernel_size = int(getattr(config, "local_kernel_size", 5))
            if kernel_size < 1:
                kernel_size = 1
            padding = kernel_size // 2
            # Depthwise temporal convolution per channel to capture local dynamics.
            self.local_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.n_vars,
                    out_channels=self.n_vars,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=self.n_vars,
                    bias=False,
                ),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.local_proj = nn.Linear(config.seq_len, config.pred_len)
            self.local_alpha = nn.Parameter(
                torch.tensor(float(getattr(config, "local_alpha", 0.1)))
            )

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        local_output = None
        if self.use_local_branch:
            local_input = rearrange(input, "b l n -> b n l")
            local_feature = self.local_conv(local_input)
            local_output = self.local_proj(local_feature)
            local_output = rearrange(local_output, "b n d -> b d n")

        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        if local_output is not None:
            output = output + self.local_alpha * local_output
        output = self.cluster.revin(output, "denorm")
        return output, L_importance
