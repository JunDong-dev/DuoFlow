"""Causal temporal VAE with separate body and hand branches."""

import torch.nn as nn
import torch
from models.causal_cnn_streamer import CausalEncoder, CausalDecoder


class Causal_TAE3parts(nn.Module):
    def __init__(self,
                 inputdim=333,
                 hidden_size=1024,
                 down_t=1,
                 stride_t=3,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):

        super().__init__()

        self.decode_proj = nn.Linear(latent_dim, width)

        self.latent_dim = latent_dim


        self.encoder = CausalEncoder(inputdim, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, latent_dim=latent_dim, clip_range=clip_range)
        self.decoder = CausalDecoder(inputdim, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)


    def preprocess(self, x):
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder, mu, logvar = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])

        return x_encoder, mu, logvar

    def myencode(self, x):
        x_in = self.preprocess(x)
        x_encoder, mu, logvar = self.encoder(x_in)

        return x_encoder, mu, logvar


    def forward(self, x):
        x_in = self.preprocess(x)

        x_encoder, mu, logvar = self.encoder(x_in)
        x_encoder = self.decode_proj(x_encoder)

        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)
        return x_out, mu, logvar


    def forward_decoder(self, x):

        x_width = self.decode_proj(x)
        x_decoder = self.decoder(x_width)
        x_out = self.postprocess(x_decoder)
        return x_out


class Causal_HumanTAE3parts(nn.Module):


    def __init__(self,
                 inputdim=470,
                 hidden_size=1024,
                 down_t=1,
                 stride_t=3,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = []
                 ):

        super().__init__()
        self.inputdim = inputdim
        self.latent_dim = latent_dim


        if inputdim == 470:
            self.input_bodydim = 272 + 18
            self.input_handdim = 90
            self.input_expdim = 0
        elif inputdim == 433:
            self.input_bodydim = 153
            self.input_handdim = 90
            self.input_expdim = 100
        elif inputdim == 333:
            self.input_bodydim = 153
            self.input_handdim = 90
            self.input_expdim = 0
        else:
            raise ValueError(
                f"Causal_HumanTAE3parts only supports inputdim in "
                f"{{333, 433, 470}}, got {inputdim}"
            )


        self.body_latent_dim = latent_dim * 2
        self.hand_latent_dim = latent_dim
        self.exp_latent_dim = latent_dim if self.input_expdim > 0 else 0

        self.total_latent_dim = (
            self.body_latent_dim
            + self.hand_latent_dim * 2
            + self.exp_latent_dim
        )


        self.tae_body = Causal_TAE3parts(
            self.input_bodydim, hidden_size, down_t, stride_t, hidden_size,
            depth, dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=self.body_latent_dim, clip_range=clip_range,
        )
        self.tae_lhand = Causal_TAE3parts(
            self.input_handdim, hidden_size, down_t, stride_t, hidden_size,
            depth, dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=self.hand_latent_dim, clip_range=clip_range,
        )
        self.tae_rhand = Causal_TAE3parts(
            self.input_handdim, hidden_size, down_t, stride_t, hidden_size,
            depth, dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=self.hand_latent_dim, clip_range=clip_range,
        )
        if self.input_expdim > 0:
            self.tae_exp = Causal_TAE3parts(
                self.input_expdim, hidden_size, down_t, stride_t, hidden_size,
                depth, dilation_growth_rate, activation=activation, norm=norm,
                latent_dim=self.exp_latent_dim, clip_range=clip_range,
            )
        else:
            self.tae_exp = None


    def _split_input(self, x):


        b_end = self.input_bodydim
        l_end = b_end + self.input_handdim
        r_end = l_end + self.input_handdim
        x_b = x[:, :, :b_end]
        x_l = x[:, :, b_end:l_end]
        x_r = x[:, :, l_end:r_end]
        if self.tae_exp is not None:
            x_e = x[:, :, r_end:r_end + self.input_expdim]
            return x_b, x_l, x_r, x_e
        return x_b, x_l, x_r

    def _split_latent(self, x):


        b_end = self.body_latent_dim
        l_end = b_end + self.hand_latent_dim
        r_end = l_end + self.hand_latent_dim
        x_b = x[:, :, :b_end]
        x_l = x[:, :, b_end:l_end]
        x_r = x[:, :, l_end:r_end]
        if self.tae_exp is not None:
            x_e = x[:, :, r_end:r_end + self.exp_latent_dim]
            return x_b, x_l, x_r, x_e
        return x_b, x_l, x_r


    def encode(self, x):


        return self.myencode(x)

    def myencode(self, x):
        parts = self._split_input(x)
        if self.tae_exp is not None:
            x_b, x_l, x_r, x_e = parts
            h_e, mu_e, logvar_e = self.tae_exp.myencode(x_e)
        else:
            x_b, x_l, x_r = parts
            h_e = mu_e = logvar_e = None
        h_b, mu_b, logvar_b = self.tae_body.myencode(x_b)
        h_l, mu_l, logvar_l = self.tae_lhand.myencode(x_l)
        h_r, mu_r, logvar_r = self.tae_rhand.myencode(x_r)
        if self.tae_exp is not None:
            h = torch.cat([h_b, h_l, h_r, h_e], dim=-1)
            mu = torch.cat([mu_b, mu_l, mu_r, mu_e], dim=2)
            logvar = torch.cat([logvar_b, logvar_l, logvar_r, logvar_e], dim=2)
        else:
            h = torch.cat([h_b, h_l, h_r], dim=-1)
            mu = torch.cat([mu_b, mu_l, mu_r], dim=2)
            logvar = torch.cat([logvar_b, logvar_l, logvar_r], dim=2)
        return h, mu, logvar

    def forward(self, x):
        parts = self._split_input(x)
        if self.tae_exp is not None:
            x_b, x_l, x_r, x_e = parts
            x_out_e, mu_e, logvar_e = self.tae_exp(x_e)
        else:
            x_b, x_l, x_r = parts
            x_out_e = mu_e = logvar_e = None
        x_out_b, mu_b, logvar_b = self.tae_body(x_b)
        x_out_l, mu_l, logvar_l = self.tae_lhand(x_l)
        x_out_r, mu_r, logvar_r = self.tae_rhand(x_r)
        if self.tae_exp is not None:
            x_out = torch.cat([x_out_b, x_out_l, x_out_r, x_out_e], dim=2)
            mu = torch.cat([mu_b, mu_l, mu_r, mu_e], dim=2)
            logvar = torch.cat([logvar_b, logvar_l, logvar_r, logvar_e], dim=2)
        else:
            x_out = torch.cat([x_out_b, x_out_l, x_out_r], dim=2)
            mu = torch.cat([mu_b, mu_l, mu_r], dim=2)
            logvar = torch.cat([logvar_b, logvar_l, logvar_r], dim=2)
        return x_out, mu, logvar

    def forward_decoder(self, x):
        parts = self._split_latent(x)
        if self.tae_exp is not None:
            x_b, x_l, x_r, x_e = parts
            x_out_b = self.tae_body.forward_decoder(x_b)
            x_out_l = self.tae_lhand.forward_decoder(x_l)
            x_out_r = self.tae_rhand.forward_decoder(x_r)
            x_out_e = self.tae_exp.forward_decoder(x_e)
            x_out = torch.cat([x_out_b, x_out_l, x_out_r, x_out_e], dim=2)
        else:
            x_b, x_l, x_r = parts
            x_out_b = self.tae_body.forward_decoder(x_b)
            x_out_l = self.tae_lhand.forward_decoder(x_l)
            x_out_r = self.tae_rhand.forward_decoder(x_r)
            x_out = torch.cat([x_out_b, x_out_l, x_out_r], dim=2)
        return x_out
