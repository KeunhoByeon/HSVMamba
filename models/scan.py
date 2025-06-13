import torch

WITH_TRITON = False


# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0, window_size=7):
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
            no_flip = True if B <= 3 else False

            assert H % window_size == 0 and W % window_size == 0
            P = H // window_size
            N = P * P

            patches = torch.stack([
                x[:, :, ph::P, pw::P]  # (B, C, h', w')
                for ph in range(P) for pw in range(P)
            ], dim=1)  # (B, N, C, h', w')

            dir_ids = torch.arange(N, device=x.device) % 8  # (N,)
            patches = patches.view(B * N, C, window_size, window_size)  # (B*N, C, h, w)

            # 3. Apply directional transforms
            y = torch.empty_like(patches)
            for d in range(8):
                idx = (dir_ids == d).repeat(B)  # (B*N,)
                if idx.any():
                    p_sel = patches[idx]
                    rot = d % 4
                    if rot > 0:
                        p_sel = torch.rot90(p_sel, k=rot, dims=(2, 3))
                    if d % 8 >= 4 and not no_flip:
                        p_sel = torch.flip(p_sel, dims=[-2])
                    y[idx] = p_sel
            y = y.view(B, N, C, window_size, window_size)
            y = y.flatten(3, 4)

            # dir_ids = torch.arange(B, device=x.device) % 8
            # y = torch.empty_like(x)
            # for d in range(8):
            #     idx = dir_ids == d
            #     if idx.any():
            #         x_sel = x[idx]
            #         rot = d % 4
            #         if rot > 0:
            #             x_sel = torch.rot90(x_sel, k=rot, dims=(2, 3))
            #         if d >= 4:
            #             x_sel = torch.flip(x_sel, dims=[-2])  # flip height
            #         y[idx] = x_sel
            # y = y.flatten(2, 3)  # (B, C, H*W)
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
            no_flip = True if B <= 3 else False

            assert H % window_size == 0 and W % window_size == 0
            P = H // window_size
            N = P * P

            patches = torch.stack([
                x[:, ph::P, pw::P, :]  # (B, h', w', C)
                for ph in range(P) for pw in range(P)
            ], dim=1)  # (B, N, h', w', C)

            dir_ids = torch.arange(N, device=x.device) % 8  # (N,)
            patches = patches.view(B * N, window_size, window_size, C)  # (B*N, h, w, C)

            # 3. Apply directional transforms
            y = torch.empty_like(patches)
            for d in range(8):
                idx = (dir_ids == d).repeat(B)  # (B*N,)
                if idx.any():
                    p_sel = patches[idx]
                    rot = d % 4
                    if rot > 0:
                        p_sel = torch.rot90(p_sel, k=rot, dims=(1, 2))
                    if d % 8 >= 4 and not no_flip:
                        p_sel = torch.flip(p_sel, dims=[-3])
                    y[idx] = p_sel

            y = y.view(B, N, window_size, window_size, C)
            y = y.flatten(2, 3)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0, window_size=7):
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
            no_flip = True if B <= 3 else False

            N = int(B * K)
            y = y.view(N, D, H, W)

            # Direction ID for each patch index
            dir_ids = torch.arange(N, device=y.device) % 8  # (N,)

            # Inverse transform for each patch
            for d in range(8):
                idx = (dir_ids == d)
                if idx.any():
                    patch = y[idx]
                    rot = -(d % 4) % 4  # rot90 inverse
                    if d % 8 >= 4 and not no_flip:
                        patch = torch.flip(patch, dims=[-2])
                    if rot > 0:
                        patch = torch.rot90(patch, k=rot, dims=(2, 3))
                    y[idx] = patch

            y = y.view(B, K, D, H, W)

            # Reconstruct original grid (B, C, H, W)
            P = int(K ** 0.5)

            i = 0
            x_rec = torch.zeros((B, D, H * P, W * P), device=y.device, dtype=y.dtype)
            for ph in range(P):
                for pw in range(P):
                    x_rec[:, :, ph::P, pw::P] = y[:, i]
                    i += 1

            y = x_rec.view(B, D, -1)
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
            no_flip = True if B <= 3 else False

            N = int(B * K)
            y = y.view(N, H, W, D)

            # Direction ID for each patch index
            dir_ids = torch.arange(N, device=y.device) % 8  # (N,)

            # Inverse transform for each patch
            for d in range(8):
                idx = (dir_ids == d)
                if idx.any():
                    patch = y[idx]
                    rot = -(d % 4) % 4  # rot90 inverse
                    if d % 8 >= 4 and not no_flip:
                        patch = torch.flip(patch, dims=[-2])
                    if rot > 0:
                        patch = torch.rot90(patch, k=rot, dims=(2, 3))
                    y[idx] = patch

            y = y.view(B, K, H, W, D)

            # Reconstruct original grid (B, C, H, W)
            P = int(K ** 0.5)

            i = 0
            x_rec = torch.zeros((B, H * P, W * P, D), device=y.device, dtype=y.dtype)
            for ph in range(P):
                for pw in range(P):
                    x_rec[:, ph::P, pw::P, :] = y[:, i]
                    i += 1

            y = x_rec.view(B, -1, D)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, window_size=7):
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
        y = cross_scan_fwd(x, in_channel_first, out_channel_first, scans, window_size=window_size)

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
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, window_size=7):
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
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans, window_size)

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
        x = x.view(B, -1, C, H, W) if out_channel_first else x.view(B, H, W, -1, C)

        return x, None, None, None, None


def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, window_size=7, **kwargs):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;

    if x.is_cuda:
        with torch.cuda.device(x.device):
            return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans, window_size)
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


if __name__ == '__main__':
    import torch
    from math import isqrt

    B, C, H, W = 4, 3, 56, 56
    scans = 4
    in_channel_first = True
    out_channel_first = True
    window_size = 7

    # Dummy input
    x = torch.randn(B, C, H, W)
    x.requires_grad_(True)

    # Forward scan
    print("Input shape:", x.shape)
    y = cross_scan_fwd(x, in_channel_first, out_channel_first, scans, window_size=window_size)
    print("Scan output shape:", y.shape)

    # Reshape for merge
    if scans == 4:
        B, N, C, L = y.shape
        patch_hw = isqrt(L)
        assert patch_hw * patch_hw == L, "Flattened patch shape is not square!"
        y = y.view(B, N, C, patch_hw, patch_hw)
    print("Reshape for merge:", y.shape)

    # Merge
    x_merged = cross_merge_fwd(y, in_channel_first, out_channel_first, scans, )
    print("Merged shape:", x_merged.shape)

    x_rec = x_merged.view(B, C, H, W)

    print("Reconstructed shape:", x_rec.shape)

    # Check reconstruction error
    error = (x - x_rec).abs().max().item()
    print(f"Max reconstruction error: {error:.2e}")

    # Check backward
    x_rec.sum().backward()
    print("Backward passed:", x.grad is not None and torch.all(torch.isfinite(x.grad)))
    print("Gradient max abs:", x.grad.abs().max().item())
