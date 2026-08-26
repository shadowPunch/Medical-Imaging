"""
Automated check of the Head A / Head B isolation guarantee
(TB_CXR_Diagnostic_3D_Proposal.md §11a): Head B's reconstruction output must
never enter any path that affects Head A's probability or threshold. This
is meant to hold structurally, not just by convention — this test exists so
a future edit that accidentally couples the two paths gets caught here
first, not in a values-look-slightly-off debugging session months later.

Two independent checks:
  1. Value isolation — Head A's diagnostic logit is bit-for-bit identical
     whether the model was built with or without a recon_head. Guards
     against any shared state (buffers, in-place ops) between the two heads.
  2. Gradient isolation — after backpropagating through the diagnosis path
     alone, every recon_head parameter's .grad is exactly None. Guards
     against forward() secretly touching recon_head even in a way that
     doesn't change values (e.g. a no-op call still built into the graph).

Run directly: python -m recon.firewall_test
"""
import torch

from models.tb_model import build_model


def check_value_isolation() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 320, 320)

    torch.manual_seed(42)
    model_plain = build_model("efficientnet_b0", pretrained=False, with_recon=False)
    torch.manual_seed(42)
    model_with_recon = build_model("efficientnet_b0", pretrained=False, with_recon=True)

    model_plain.eval()
    model_with_recon.eval()
    with torch.no_grad():
        out_plain = model_plain(x)
        out_with_recon = model_with_recon(x)

    assert torch.equal(out_plain, out_with_recon), (
        "FIREWALL BREACH: diagnostic output differs depending on whether a "
        "recon_head is attached. Head A's forward() must be fully "
        "independent of self.recon_head."
    )
    print("[ok] value isolation — diagnostic logit identical with/without recon_head")


def check_gradient_isolation() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 320, 320)
    labels = torch.randint(0, 2, (2, 1)).float()

    model = build_model("efficientnet_b0", pretrained=False, with_recon=True)
    model.train()

    logits = model(x)  # diagnosis path only — forward_recon() never called
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    leaked = [name for name, p in model.recon_head.named_parameters() if p.grad is not None]
    assert not leaked, (
        f"FIREWALL BREACH: gradients reached recon_head parameters {leaked} "
        "from a diagnosis-only backward pass. Head B must never be part of "
        "Head A's computation graph."
    )
    print("[ok] gradient isolation — no gradient reached recon_head from a diagnosis-only backward pass")


def check_recon_path_independently_works() -> None:
    """Sanity check the other direction — forward_recon() itself still works
    and does NOT touch self.head, so the isolation is symmetric."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 320, 320)
    model = build_model("efficientnet_b0", pretrained=False, with_recon=True, volume_size=32)
    model.train()

    volume = model.forward_recon(x)
    loss = volume.pow(2).mean()
    loss.backward()

    leaked = [name for name, p in model.head.named_parameters() if p.grad is not None]
    assert not leaked, (
        f"FIREWALL BREACH: gradients reached Head A parameters {leaked} from "
        "a reconstruction-only backward pass."
    )
    print(f"[ok] reverse isolation — forward_recon() (volume shape {tuple(volume.shape)}) "
          "doesn't touch Head A either")


def main() -> None:
    check_value_isolation()
    check_gradient_isolation()
    check_recon_path_independently_works()
    print("\nAll firewall checks passed.")


if __name__ == "__main__":
    main()
