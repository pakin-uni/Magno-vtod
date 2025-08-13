class C2f_EMA(nn.Module):
    def __init__(self, c1, c2, num_blocks=1, shortcut=False, g=1, e=0.2):
        super().__init__()
        c_ = int(c2 * e)

        # Initial projection and EMA
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.ema = EMA(2 * c_)

        # First bottleneck after EMA
        self.m1 = Bottleneck(2 * c_, c_, shortcut, g, e=1.0)

        # Split path after EMA
        self.cv2 = Conv(2 * c_, c_, 1, 1)

        # Single bottleneck after split
        self.m2 = Bottleneck(c_, c_, shortcut, g, e=1.0)

        # Final output projection
        self.cv3 = Conv(4 * c_, c2, 1, 1)  # 2*c_ (cv1 output) + m1 + m2 = 4*c_

    def forward(self, x):
        y = self.cv1(x)
        y = self.ema(y)

        y1 = self.m1(y)
        y2 = self.cv2(y)
        y2a = self.m2(y2)

        out = self.cv3(torch.cat((y, y1, y2a), dim=1))
        return out

class C2f_EMA(nn.Module):
    def __init__(self, c1, c2, num_blocks=3, shortcut=False, g=1, e=0.5, dropout_p=0.0):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.ema = EMA(2 * c_)

        # Optional dropout for regularization after EMA
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        # Create bottleneck blocks dynamically based on num_blocks
        self.m_blocks = nn.ModuleList()
        if num_blocks >= 1:
            self.m_blocks.append(Bottleneck(2 * c_, c_, shortcut, g, e=1.0))
        if num_blocks >= 2:
            self.m_blocks.append(Bottleneck(c_, c_, shortcut, g, e=1.0))
        if num_blocks >= 3:
            self.m_blocks.append(Bottleneck(c_, c_, shortcut, g, e=1.0))

        self.cv2 = Conv(2 * c_, c_, 1, 1)
        self.cv3 = Conv((2 + num_blocks) * c_, c2, 1, 1)  # adjust output channels accordingly

    def forward(self, x):
        y = self.cv1(x)
        y = self.ema(y)
        y = self.dropout(y)

        outs = []
        if len(self.m_blocks) >= 1:
            y1 = self.m_blocks[0](y)
            outs.append(y1)
        if len(self.m_blocks) >= 2:
            y2a = self.m_blocks[1](self.cv2(y))
            outs.append(y2a)
        if len(self.m_blocks) >= 3:
            y2b = self.m_blocks[2](outs[-1])
            outs.append(y2b)

        # Concatenate input, plus all bottleneck outputs
        out = self.cv3(torch.cat([y] + outs, dim=1))
        return out

