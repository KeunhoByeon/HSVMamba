import torch

WITH_TRITON = False


# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        elif scans == 3:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = torch.rot90(x, 1, dims=(2, 3)).flatten(2, 3)
            y[:, 2, :, :] = torch.rot90(x, 2, dims=(2, 3)).flatten(2, 3)
            y[:, 3, :, :] = torch.rot90(x, 3, dims=(2, 3)).flatten(2, 3)
        elif scans == 4:
            # TODO: global_cross
            raise NotImplementedError
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        elif scans == 3:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = torch.rot90(x, 1, dims=(1, 2)).flatten(1, 2)
            y[:, :, 2, :] = torch.rot90(x, 2, dims=(1, 2)).flatten(1, 2)
            y[:, :, 3, :] = torch.rot90(x, 3, dims=(1, 2)).flatten(1, 2)
        elif scans == 4:
            # TODO: global_cross
            raise NotImplementedError

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
        elif scans == 3:
            oy = y[:, 0, :, :].contiguous().view(B, D, -1)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3)
            y = oy
        elif scans == 4:
            # TODO: global_cross
            raise NotImplementedError
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
        elif scans == 3:
            oy = y[:, :, 0, :].contiguous().view(B, -1, D)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2)
            y = oy
        elif scans == 4:
            # TODO: global_cross
            raise NotImplementedError

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            raise NotImplementedError

        B, C, H, W = x.shape
        if not in_channel_first:
            B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)
        y = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        if one_by_one:
            raise NotImplementedError

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)
        y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            raise NotImplementedError

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        if one_by_one:
            raise NotImplementedError

        x = x.view(B, C, H, W) if in_channel_first else x.view(B, H, W, C)
        x = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)

        return x, None, None, None, None


def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, **kwargs):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;

    if x.is_cuda:
        with torch.cuda.device(x.device):
            return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, **kwargs):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;

    if y.is_cuda:
        with torch.cuda.device(y.device):
            return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
