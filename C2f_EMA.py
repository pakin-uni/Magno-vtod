import torch
import torch.nn as nn

# Assuming the following classes are already defined elsewhere in module.py:
# Conv, Bottleneck, EMA

class C2f_EMA(nn.Module):
    def __init__(self, c1, c2, num_blocks=3, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        # Initial Conv to double the channels for splitting
        self.cv1 = Conv(c1, 2 * c_, 1, 1)

        # EMA block for the first branch
        self.ema = EMA(c_, c_)
        
        # A list of Bottleneck blocks for the second branch
        self.m = nn.ModuleList(
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(num_blocks)
        )

        # Final Conv to combine all features
        self.cv2 = Conv((2 + num_blocks) * c_, c2, 1, 1)

    def forward(self, x):
        # Initial projection and split into two halves
        x = self.cv1(x)
        y = list(x.chunk(2, 1))  # Split into y[0] and y[1], both with c_ channels

        # Process the first half with EMA
        y[0] = self.ema(y[0])
        
        # Sequentially pass the second half through the bottleneck blocks
        for m in self.m:
            y.append(m(y[-1]))

        # Concatenate all parts and apply the final convolution
        # The list 'y' now contains: [ema_output, initial_split_2, bneck1_out, bneck2_out, ...]
        out = self.cv2(torch.cat(y, 1))

        return out
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        assert channels % groups == 0, "Channels must be divisible by groups"
        self.groups = groups
        self.gc = channels // groups  # channels per group

        # Main convolution branch
        self.conv_main = nn.Conv2d(self.gc, self.gc, kernel_size=3, padding=1, bias=False)

        # Channel Attention path
        self.conv_1x1_channel = nn.Conv2d(self.gc * 2, self.gc, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.reweight_channel = nn.Conv2d(self.gc, self.gc, kernel_size=1, bias=False)

        # Cross-Spatial Learning (CSL) path
        self.group_norm = nn.GroupNorm(num_groups=self.gc, num_channels=self.gc)
        self.matmul = lambda a, b: a @ b

        # Final combination layers
        self.sigmoid_final = nn.Sigmoid()
        self.reweight_final = nn.Conv2d(self.gc, self.gc, kernel_size=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        G, gc = self.groups, self.gc
        
        # Step 1: Grouping
        x_grouped = x.view(B, G, gc, H, W).reshape(B * G, gc, H, W)
        
        # Step 2: Main Conv(3x3)
        x_conv_3x3 = self.conv_main(x_grouped)

        # Step 3: Channel Attention Path
        y_pool = F.adaptive_avg_pool2d(x_grouped, (H, 1))
        x_pool = F.adaptive_avg_pool2d(x_grouped, (1, W))
        
        y_pool = y_pool.view(B * G, gc, H)
        x_pool = x_pool.view(B * G, gc, W)
        
        concatenated_pools = torch.cat([y_pool, x_pool], dim=2)
        channel_attention = self.sigmoid_channel(self.conv_1x1_channel(concatenated_pools.unsqueeze(-1)))
        
        # Step 4: Cross-Spatial Learning (CSL) Path
        # Input to the bottom CSL branch is the original input re-weighted by the channel attention map.
        reweighted_input = self.reweight_channel(channel_attention) * x_grouped
        
        # Branch 1 (top) from Conv(3x3)
        pooled_b1 = F.adaptive_avg_pool2d(x_conv_3x3, (1, 1)).view(B * G, gc)
        softmax_b1 = F.softmax(pooled_b1, dim=1).view(B * G, 1, gc)
        matmul_b1 = self.matmul(softmax_b1, pooled_b1.view(B * G, gc, 1))

        # Branch 2 (bottom) from GroupNorm
        normed_b2 = self.group_norm(reweighted_input)
        pooled_b2 = F.adaptive_avg_pool2d(normed_b2, (1, 1)).view(B * G, gc)
        softmax_b2 = F.softmax(pooled_b2, dim=1).view(B * G, 1, gc)
        matmul_b2 = self.matmul(softmax_b2, pooled_b2.view(B * G, gc, 1))

        cross_spatial_attention = matmul_b1 + matmul_b2
        
        # Step 5: Final Combination
        # The diagram shows the CSL path is added to a reweighted grouped input, followed by a sigmoid.
        csl_attn_map = cross_spatial_attention.view(B * G, gc, 1, 1)
        
        combined_attn = csl_attn_map + x_grouped
        
        # Corrected order: Sigmoid before final re-weight
        final_attn_sig = self.sigmoid_final(combined_attn)
        final_attn = self.reweight_final(final_attn_sig)
        
        # Corrected: Apply the final attention map to the x_grouped input
        out = x_grouped * final_attn
        
        return out.view(B, C, H, W)