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