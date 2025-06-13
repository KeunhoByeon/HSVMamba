import torch

from models.hsvmamba import HSVSSM
# from VMamba.classification.models.vmamba import vmamba_tiny_s1l8
import time

def hsvmamba_tiny_s1l8(channel_first=True, **kwargs):
    # return vmamba_tiny_s1l8()
    model = HSVSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )
    return model


def test_hsvss_block():
    model = hsvmamba_tiny_s1l8()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    B, C, H, W = 5, 3, 224, 224
    x = torch.randn(B, C, H, W, requires_grad=True)
    y = torch.randint(0, 999, (B,))

    start = time.time()

    with torch.no_grad():
        for step in range(100):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                model = model.cuda()

                torch.cuda.reset_peak_memory_stats()  # 이거 추가하면 step별 최대값 리셋 가능
                torch.cuda.synchronize()  # 정확한 측정을 위해 동기화
                before_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB 단위

            y_hat = model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            print(f"Shape: {x.shape} -> {y_hat.shape}, Loss {loss}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 연산 끝까지 기다림
                after_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

                print(f"[Step {step}] Allocated Memory: {after_mem:.2f} MB, "
                      f"Peak Memory: {peak_mem:.2f} MB")

            print(f"Shape: {x.shape} -> {y_hat.shape}, Loss {loss}")

        end = time.time()
        print(end - start)


def local_window_method():
    P = 2
    x = torch.arange(64)
    x = x.reshape(1, 1, 8, 8)
    print(x.numpy().shape)
    print(x.numpy())
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // P, P, W // P, P)  # (B, C, H//P, P, W//P, P)
    print(x.numpy().shape)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H//P, W//P, C, P, P)
    print(x.numpy().shape)
    x = x.view(-1, C, P, P)
    print(x.numpy().shape)
    print(x.numpy()[:, 0, :, :])
    y = x.view(B, H // P, W // P, C, P, P)  # (B, H//P, W//P, C, P, P)
    print(y.numpy().shape)
    y = y.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, H, W)
    print(y.numpy().shape)
    y = y.view(B, C, H, W)
    print(y.numpy().shape)
    print(y.numpy())


def global_window_method():
    patch_size = 2
    x = torch.arange(64)
    x = x.reshape(1, 1, 8, 8)
    print(x.numpy().shape)

    B, C, H, W = x.shape
    P = H // patch_size

    # 1. Crop if needed (안전)
    H_pad = H - (H % P)
    W_pad = W - (W % P)
    x = x[:, :, :H_pad, :W_pad]
    print(x.numpy().shape)

    # 2. Dilated sampling → strided indexing
    # Shape: (B, C, H//P, P, W//P, P)
    x = x.reshape(B, C, H_pad // P, P, W_pad // P, P)
    print(x.numpy().shape)
    # Shape: (B, C, P, P, H//P, W//P)
    print(x.numpy().shape)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    print(x.numpy().shape)
    # Shape: (B * P * P, C, H//P, W//P)
    x = x.view(B * (P * P), C, H_pad // P, W_pad // P)
    print(x.numpy().shape)

    # 3. SSM 처리
    print(x.numpy())
    y = x  # (B * P * P, C, H//P, W//P)

    # 4. 다시 합치기
    y = y.view(B, P, P, C, H_pad // P, W_pad // P)
    print(y.numpy().shape)
    y = y.permute(0, 3, 4, 1, 5, 2).contiguous()  # (B, C, H//P, P, W//P, P)
    print(y.numpy().shape)
    y = y.view(B, C, H_pad, W_pad)
    print(y.numpy().shape)
    print(y.numpy())

    return y


if __name__ == "__main__":
    test_hsvss_block()
