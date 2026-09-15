import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconHead(nn.Module):
    """
    Head B: back-projection reconstruction from the shared encoder's
    multi-scale 2D features into a synthesized 3D density volume.

    Paradigm (see TB_CXR_Diagnostic_3D_Proposal.md §6): encode 2D -> lift
    into a 3D voxel grid -> decode with 3D convolutions. The "lift" step
    reshapes the deepest 2D feature map's channels into an initial (C, D,
    H, W) volume — a channel-to-depth unprojection, not full geometry-aware
    ray sampling. This is a right-sized reimplementation of the DuoLift-CNN
    back-projection *paradigm*, not a vendored copy of the released
    DuoLift code/weights (see code/readme.md's Phase 3 status for why).

    Takes the *full* multi-scale feature list (matching SharedEncoder's
    contract and DiagnosticHead's sibling interface) but only consumes
    features[-1] for now — shallower scales are accepted for future
    skip-connection fusion (finer spatial detail than the 10x10 bottleneck
    alone can provide) and currently ignored. That's an intentional,
    documented gap, not an oversight.

    Output is a raw (unbounded) density volume — never consumed by Head A;
    see recon/firewall_test.py for the code-level isolation guarantee.
    """

    def __init__(self, in_channels: int, volume_size: int = 128,
                lift_depth: int = 8, base_channels: int = 32):
        super().__init__()
        self.volume_size = volume_size
        self.lift_depth = lift_depth
        self.lift_channels = base_channels * 2

        # 2D -> (lift_channels * lift_depth) channels, so a reshape below
        # turns "channels" into an initial depth axis.
        self.to_lift = nn.Conv2d(in_channels, self.lift_channels * lift_depth, kernel_size=1)

        def up_block(c_in: int, c_out: int) -> nn.Module:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            up_block(self.lift_channels, base_channels * 2),
            up_block(base_channels * 2, base_channels),
            up_block(base_channels, base_channels // 2),
            up_block(base_channels // 2, base_channels // 4),
        )
        self.to_density = nn.Conv3d(base_channels // 4, 1, kernel_size=1)
        # Real CT density is ~50-63% near-zero air/background (measured on
        # held-out LIDC-IDRI — see code/readme.md's Phase 3 section). Default
        # init leaves this bias near 0, so softplus(0)=ln(2)~=0.693 is every
        # voxel's starting prediction; reaching the sparse floor (<0.02)
        # needs pre-activation below ~-3.9, and a checkpoint trained 2000
        # steps only moved this bias to -0.095 — nowhere near that, which is
        # the confirmed mechanism behind predictions that are never near-zero
        # anywhere. Starting the bias there directly, instead of asking
        # gradient descent to find it from ~0, should make the sparse
        # background the default rather than something to slowly discover.
        nn.init.constant_(self.to_density.bias, -4.0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        features: the full multi-scale list from SharedEncoder.
        Returns a (B, 1, volume_size, volume_size, volume_size) density volume.
        """
        x = features[-1]  # (B, C, H, W) — deepest, most semantic
        b, _, h, w = x.shape

        x = self.to_lift(x)                                   # (B, lift_channels*lift_depth, H, W)
        x = x.view(b, self.lift_channels, self.lift_depth, h, w)  # (B, C, D, H, W)
        x = self.decoder(x)
        x = self.to_density(x)

        if x.shape[-1] != self.volume_size:
            x = F.interpolate(x, size=(self.volume_size,) * 3, mode="trilinear", align_corners=False)

        # CT density (post transform_hu_to_density) is a physically non-negative
        # attenuation quantity — an unconstrained raw output leaves the DRR
        # renderer's input near-zero/negative at initialization, which produces
        # a degenerate (near-zero-variance) projection and exactly-zero
        # gradients through the re-projection loss (verified empirically: a
        # linear output stalled shape-induction training completely, softplus
        # doesn't). Smooth and non-negative, unlike a plain ReLU's dead zone.
        return F.softplus(x)
