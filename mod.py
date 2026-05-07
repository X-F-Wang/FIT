import torch
from torch import nn as nn
from torch.nn import functional as f

import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class RP(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RP, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        self.linear = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=1, stride=1,
                                 groups=G, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + (self.convmap(origin_weight) * self.linear(origin_weight)).view(*self.origin_kernel_shape)
        return f.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)

class FIM1(nn.Module):
    def __init__(self,dim):
        super(FIM1, self).__init__()

        self.Conv2d = nn.Conv2d(dim, 256, 3, 1, 1)
        self.out0 = nn.Conv2d(256,256,1,1,0)



    def forward(self, x):


        x0 = self.Conv2d(x)
        f = torch.fft.fftn(x, dim=(2, 3))

        fr = f.real
        fi = f.imag

        fr = self.Conv2d(fr)
        fi = self.Conv2d(fi)

        f = torch.complex(fr,fi)

        f0 =x0+f


        x = torch.abs(torch.fft.ifftn(f0, dim=(2, 3)))
        x= self.out0(x)
        return x

class FIM(nn.Module):
    def __init__(self,dim):
        super(FIM, self).__init__()

        self.Conv2d = nn.Conv2d(dim, 64, 3, 1, 1)
        self.out0 = nn.Conv2d(64,64,1,1,0)



    def forward(self, x):

        f = torch.fft.fftn(x, dim=(2, 3))

        fr = f.real
        fi = f.imag

        fr = self.Conv2d(fr)
        fi = self.Conv2d(fi)

        f = torch.complex(fr,fi)

        f0 =x+f


        x = torch.abs(torch.fft.ifftn(f0, dim=(2, 3)))
        x= self.out0(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()


    def forward(self, x):
        B,C,H,W = x.shape

        weight = nn.Parameter(torch.ones(H, W)).cuda()
        bias = nn.Parameter(torch.zeros(H, W)).cuda()
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * weight + bias

class FCSA(nn.Module):
    def __init__(self, dim):
        super(FCSA, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim*3, kernel_size=1)
        self.to_hidden_dw = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm = LayerNorm()

        self.patch_size = 1

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)
        output = output+x

        return output

class IISA(nn.Module):
    def __init__(self, midc, num = 4):
        super().__init__()

        self.headc = midc // num
        self.num = num
        self.midc = midc
        self.proj1 = nn.Conv2d(midc , midc * 2, kernel_size=1, padding=0, stride=1,
                               groups=self.headc)
        #print(f'MsSA_q: {self.num}')
        self.proj2 = nn.Conv2d(midc * 2, midc , 1)
        self.proj0 = nn.Conv2d(midc, midc, 1)
        self.bn = nn.BatchNorm2d(midc * 2)
        self.act = nn.GELU()

        for i in range(self.num):
            local_conv = nn.Conv2d(self.headc, self.headc, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1)
            setattr(self, f"local_conv_{i + 1}", local_conv)

        self.apply(self._init_weights) #常见的初始化方法

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feat, name='0'):
        # bias = x #[8, 256, 48, 48]
        x = torch.split(self.proj0(feat), self.headc, dim = 1)
        out = []
        for i in range(self.num):
            if (i % 2 == 0):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                x_i = x[i]
                out_ = local_conv(x_i)
            else:
                x_i = x[i]
                f = torch.fft.fftn(x_i, dim=(2, 3))
                fr = f.real
                fi = f.imag
                fr = local_conv(fr)
                fi = local_conv(fi)
                f = torch.complex(fr, fi)
                out_ = torch.abs(torch.fft.ifftn(f, dim=(2, 3)))

            out.append(out_)

        s_out = torch.cat([*out],dim = 1)
        out = self.proj2(self.act(self.bn(self.proj1(s_out))))

        return out