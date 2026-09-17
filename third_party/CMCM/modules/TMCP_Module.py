'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.
'''

import pdb
import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill


class TMA(nn.Module):
    def __init__(
        self,
        dim,
        frame=8,
        instantiation="softmax",
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        super(TMA, self).__init__()
        self.dim = dim
        self.dim = dim
        self.frame = frame
        self.instantiation = instantiation
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_TMA(
            norm_module
        )

    def _construct_TMA(
        self, norm_module
    ):
        # Three convolution : conv_phi0 (x), conv_phi_1 (x-1), and conv_phi_2 (x-2).
        self.conv_phi_1 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi_2 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi0 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )

        # TODO: change the name to `norm`
        self.norm = norm_module(
            self.dim,
            eps=self.norm_eps,
        )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                batchnorm_weight = 0.
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_identity = x
        N, C, H, W = x.size()

        T = self.frame
        V = N // T
        x = x.view(V, T, C, H, W)
        x_1 = torch.cat((x[:,0,:,:,:].unsqueeze(1),x[:,:-1,:,:,:]), 1)
        x_2 = torch.cat((x_1[:, 0, :, :, :].unsqueeze(1), x_1[:, :-1, :, :, :]), 1)
        x_1 = x_1.view(N, C, H, W)
        x_2 = x_2.view(N, C, H, W)
        x = x.view(-1, C, H, W)

        phi_x_1 = self.conv_phi_1(x_1)
        phi_x_2 = self.conv_phi_2(x_2)
        phi_x0 = self.conv_phi0(x)

        phi_x_1 = phi_x_1.view(N, self.dim, -1)
        phi_x_2 = phi_x_2.view(N, self.dim, -1)
        phi_x0 = phi_x0.view(N, self.dim, -1)

        theta_x_1_2 = torch.einsum("nca,ncb->nab", (phi_x_1, phi_x_2))
        if self.instantiation == "softmax":
            theta_x_1_2 = theta_x_1_2 * (self.dim ** -0.5)
            theta_x_1_2 = nn.functional.softmax(theta_x_1_2, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_x_1_2.shape[2]
            theta_x_1_2 = theta_x_1_2 / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW).
        x_out = torch.einsum("ntg,ncg->nct", (theta_x_1_2, phi_x0))

        # (N, C, HxW) => (N, C, H, W).
        x_out = x_out.contiguous().view(N, self.dim, H, W)

        x_out = self.norm(x_out)
        return x_identity + x_out


class TCA(nn.Module):
    def __init__(self, dim, frame):
        super(TCA, self).__init__()
        self.dim = dim
        self.frame = frame

        print("-" * 20 + "TCA called" + '-' * 20)
        self.g1 = nn.Conv2d(
            self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
        )
        self.g2 = nn.Conv2d(
            self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, x_att):
        x = self.pool(x)
        [N, C, H, W] = x.size()
        V = N // self.frame
        x_t = x.view(V, self.frame, C, H, W)
        x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
        x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

        x_diff1 = x_t - x_t1
        x_diff1 = x_diff1.view(N, C, H, W)
        x_diff2 = (x_t - x_t2)*0.5
        x_diff2 = x_diff2.view(N, C, H, W)

        x_diff1 = self.TCA_diff(x_diff1)
        x_diff2 = self.TCA_diff(x_diff2)

        return  x_att * (x_diff1  + x_diff2)*0.5


    def TCA_diff(self, x):
        x = self.g1(x)
        x = self.relu(x)
        x = self.g2(x)
        x = self.sigmoid(x)
        return x


class TMCP_att_module(nn.Module):
    def __init__(
        self,
        dim,
        frame=8,
        ch_flag=True,
        sp_flag=True,
        conv_1d_flag=True,
    ):
        super(TMCP_att_module, self).__init__()
        self.dim = dim
        self.frame = frame
        self.ch_flag = ch_flag
        self.sp_flag = sp_flag
        self.conv_1d_flag = conv_1d_flag
        self.k = frame // 2 + 1 #temporal conv kernel
        self._construct_TMCP_att(
        )

    def _construct_TMCP_att(self
    ):
        self.relu = nn.ReLU(inplace=True)

        if self.ch_flag:
            self.TCA = TCA(self.dim, self.frame)

        if self.conv_1d_flag :
            print("-" * 20 + "1D conv Module called" + '-' * 20)
            if (torch.__version__).split('.')[1] >= '5': #version 1.5+
                k_padding = (self.k-1)//2
            else :
                k_padding  = self.k-1
            self.conv_1d = nn.Conv3d(
                self.dim, self.dim, kernel_size=(self.k,1,1),
                stride=1, padding=(k_padding,0,0),
                padding_mode='circular'
            )

        if self.sp_flag:
            print("-" * 20 + "TMA Module called" + '-' * 20)
            self.TMA = TMA(self.dim, self.frame)

        self.init_weights()


    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    #c2_msra_fill(m)
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d) \
                        or isinstance(m, nn.BatchNorm3d):
                    batchnorm_weight = 0.
                    if m.weight is not None:
                        m.weight.data.fill_(batchnorm_weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
            if self.conv_1d_flag:
                self.conv_1d.weight.data.fill_(0.)
            if self.sp_flag:
                self.TMA.init_weights()

    def forward(self, x):
        if self.sp_flag:
            x_att = self.TMA(x)
        else :
            x_att = x

        if self.ch_flag :
            x_att = self.TCA(x, x_att)

        if self.conv_1d_flag:
            [N, C, H, W] = x_att.size()
            V = N // self.frame
            x_att = x_att.reshape(V, self.frame, C, H, W).permute(0,2,1,3,4)
            x_att = self.conv_1d(x_att)
            x_att = x_att.permute(0,2,1,3,4,).reshape(N, C, H, W)
            return self.relu(x_att + x)
        else :
            return x_att  # following SE, w/o shortcut


class TMCP(nn.Module):
     def __init__(self, dim_in=2048, dim_out=128, num_segments=8, level='video',
                  sp_flag=True, ch_flag=True, conv_1d_flag=True):
         super(TMCP, self).__init__()
         print("-"*20 + "Temporal-attentive Covariance Pooling called" + '-'*20)
         self.level = level
         self.sp_flag = sp_flag
         self.ch_flag = ch_flag
         self.conv_1d_flag = conv_1d_flag
         dim_inner = 256
         self.num_segments = num_segments
         self.layer_reduce1 = nn.Conv2d(
             dim_in,
             dim_inner,
             kernel_size=1,
             stride=[1,1],
             padding=0,
             bias=False,
         )

         self.layer_reduce_bn1 = nn.BatchNorm2d(
             num_features=dim_inner,
         )

         self.layer_reduce2 = nn.Conv2d(
             dim_inner,
             dim_out,
             kernel_size=1,
             stride=[1,1],
             padding=0,
             bias=False,
         )

         self.layer_reduce_bn2 = nn.BatchNorm2d(
             num_features=dim_out,
         )

         self.relu_op = nn.ReLU(inplace=True)

         if self.sp_flag or self.ch_flag or self.conv_1d_flag:
             self.TMCP_att = TMCP_att_module(
                 dim_out, frame=num_segments,
                 sp_flag=sp_flag, ch_flag=ch_flag, conv_1d_flag=conv_1d_flag)
         else :
             # w/o att module, only matrix normalization
             from ops.Identity import Identity as att_module
             self.TMCP_att = att_module()

         self.parameter_init()

     def parameter_init(self):
         for n, m in self.named_modules():
             if m == self.TMCP_att :
                 return 0
             elif isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 print(n + '  layer  kaiming initialed ')
                 if hasattr(m, 'bias'):
                     if m.bias is not None:
                         nn.init.zeros_(m.bias)
             elif isinstance(m, nn.BatchNorm2d):
                     nn.init.ones_(m.weight)
                     nn.init.zeros_(m.bias)
                     print(n + '  layer  initialed ')


     def forward(self, input):

         x = self.layer_reduce1(input)
         x = self.layer_reduce_bn1(x)
         x = self.relu_op(x)

         x = self.layer_reduce2(x)
         x = self.layer_reduce_bn2(x)
         x = self.relu_op(x)

         # attention module
         x = self.TMCP_att(x)

         if self.level == 'video':
             x = x.view((-1, self.num_segments) + x.size()[1:])
             [bs, fr, dim, h, w] = x.size()
             x = x.permute(0,2,1,3,4)
             x = x.reshape(bs, dim, fr*h, w)

         #P_{TMCP}
         x = MPNCOV.CovpoolLayer(x)
         x = MPNCOV.SqrtmLayer(x, 3)
         x = MPNCOV.TriuvecLayer(x)
         x = x.unsqueeze(-1)

         return x
