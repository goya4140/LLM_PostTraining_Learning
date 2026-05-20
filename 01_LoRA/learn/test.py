"""
Unit tests for LoRA from-scratch implementation.
All unit tests run without downloading any model.

Usage:
    python test.py              # unit tests only (~15 cases)
    python test.py -v           # verbose output
    python test.py --full       # include model-loading integration test (~1GB download)
"""
import argparse
import math
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from lora_layer import LoRALinear
from model import inject_lora, freeze_non_lora, get_lora_state_dict


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────

def make_lora_layer(in_f=64, out_f=32, r=4, alpha=8.0, dropout=0.0, bias=True):
    layer = LoRALinear(in_f, out_f, r=r, alpha=alpha, dropout=dropout, bias=bias)
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    return layer


class TinyModel(nn.Module):
    """Minimal model with q_proj, v_proj (targets) and mlp, norm (non-targets)."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.mlp    = nn.Linear(64, 128)   # NOT a target
        self.norm   = nn.LayerNorm(64)     # NOT nn.Linear


# ───────────────────────────────────────────────────────────
# Suite 1: LoRALinear
# ───────────────────────────────────────────────────────────

class TestLoRALinear(unittest.TestCase):

    def test_init_zero_delta(self):
        """ΔW = B·A = 0 at init because lora_B is initialized to zeros."""
        layer = make_lora_layer()
        delta = layer.lora_B @ layer.lora_A
        self.assertTrue(
            torch.allclose(delta, torch.zeros_like(delta)),
            "Expected B·A == 0 at init (B=0 initialization)"
        )

    def test_output_shape_3d(self):
        """Forward with [batch, seq, in_features] produces correct shape."""
        layer = make_lora_layer(in_f=64, out_f=32)
        x = torch.randn(3, 10, 64)
        self.assertEqual(layer(x).shape, (3, 10, 32))

    def test_output_shape_2d(self):
        """Forward with [batch, in_features] produces correct shape."""
        layer = make_lora_layer(in_f=64, out_f=32)
        x = torch.randn(5, 64)
        self.assertEqual(layer(x).shape, (5, 32))

    def test_scaling_factor(self):
        """Output includes lora contribution scaled by alpha/r."""
        r, alpha = 4, 8.0
        layer = make_lora_layer(in_f=16, out_f=8, r=r, alpha=alpha)
        with torch.no_grad():
            layer.lora_B.fill_(1.0)   # non-zero B so delta != 0
        x = torch.randn(2, 16)
        out = layer(x)
        expected = (F.linear(x, layer.weight, layer.bias)
                    + F.linear(F.linear(x, layer.lora_A), layer.lora_B) * (alpha / r))
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_merge_equivalence(self):
        """Merged weight produces identical output to LoRA forward pass."""
        layer = make_lora_layer(in_f=32, out_f=16, r=4)
        with torch.no_grad():
            nn.init.kaiming_uniform_(layer.lora_B)  # non-zero B
        x = torch.randn(5, 32)
        layer.eval()
        with torch.no_grad():
            out_lora   = layer(x)
            out_merged = layer.merge_weights()(x)
        self.assertTrue(torch.allclose(out_lora, out_merged, atol=1e-5),
                        "Merged layer must produce same output as LoRA forward")

    def test_gradient_routing(self):
        """Only lora_A and lora_B accumulate gradients; base weight does not."""
        layer = make_lora_layer()
        layer(torch.randn(2, 64)).sum().backward()
        self.assertIsNone(layer.weight.grad,
                          "Base weight must have no gradient")
        self.assertIsNotNone(layer.lora_A.grad, "lora_A must have gradient")
        self.assertIsNotNone(layer.lora_B.grad, "lora_B must have gradient")

    def test_param_compression(self):
        """LoRA params are < 5% of base params for typical LLM dimensions."""
        in_f, out_f, r = 768, 768, 8
        layer = LoRALinear(in_f, out_f, r=r)
        lora_params = layer.lora_A.numel() + layer.lora_B.numel()
        ratio = lora_params / layer.weight.numel()
        self.assertLess(ratio, 0.05,
                        f"Expected < 5% LoRA params, got {ratio:.3%}")

    def test_no_bias_mode(self):
        """LoRALinear(bias=False) has no bias and runs correctly."""
        layer = LoRALinear(32, 16, r=4, bias=False)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer(torch.randn(3, 32)).shape, (3, 16))

    def test_dropout_train_vs_eval(self):
        """Dropout applied in train mode introduces variance; eval mode is deterministic."""
        layer = LoRALinear(128, 64, r=8, dropout=0.5)
        with torch.no_grad():
            nn.init.kaiming_uniform_(layer.lora_B)  # non-zero B
        x = torch.randn(50, 128)

        layer.train()
        train_outs = torch.stack([layer(x) for _ in range(20)])
        train_var = train_outs.var(dim=0).mean().item()

        layer.eval()
        eval_outs = torch.stack([layer(x) for _ in range(20)])
        eval_var = eval_outs.var(dim=0).mean().item()

        self.assertGreater(train_var, eval_var,
                           "Train mode should be more variable than eval mode due to dropout")


# ───────────────────────────────────────────────────────────
# Suite 2: inject_lora
# ───────────────────────────────────────────────────────────

class TestInjectLoRA(unittest.TestCase):

    def _inject(self, model, r=4, alpha=8.0):
        inject_lora(model, r=r, alpha=alpha, dropout=0.0,
                    target_modules=["q_proj", "v_proj"])

    def test_target_layers_replaced(self):
        """q_proj and v_proj become LoRALinear after injection."""
        m = TinyModel()
        self._inject(m)
        self.assertIsInstance(m.q_proj, LoRALinear)
        self.assertIsInstance(m.v_proj, LoRALinear)

    def test_non_target_preserved(self):
        """mlp (not in target_modules) stays as nn.Linear."""
        m = TinyModel()
        self._inject(m)
        self.assertIsInstance(m.mlp, nn.Linear)

    def test_non_linear_preserved(self):
        """LayerNorm is never replaced even if its name is in target_modules."""
        m = TinyModel()
        inject_lora(m, r=4, alpha=8.0, dropout=0.0,
                    target_modules=["q_proj", "v_proj", "norm"])
        self.assertIsInstance(m.norm, nn.LayerNorm)

    def test_weights_preserved(self):
        """LoRA injection preserves original weight values."""
        m = TinyModel()
        orig_w = m.q_proj.weight.data.clone()
        self._inject(m)
        self.assertTrue(torch.allclose(m.q_proj.weight.data, orig_w),
                        "Weight values must not change during injection")

    def test_forward_runs(self):
        """Model forward pass works after LoRA injection."""
        m = TinyModel()
        self._inject(m)
        out = m.q_proj(torch.randn(2, 8, 64))
        self.assertEqual(out.shape, (2, 8, 64))

    def test_lora_dimensions(self):
        """Injected LoRA layers have correct A and B shapes."""
        m = TinyModel()
        r = 8
        inject_lora(m, r=r, alpha=16.0, dropout=0.0,
                    target_modules=["q_proj"])
        self.assertEqual(m.q_proj.lora_A.shape, (r, 64))   # (r, in_features)
        self.assertEqual(m.q_proj.lora_B.shape, (64, r))   # (out_features, r)


# ───────────────────────────────────────────────────────────
# Suite 3: freeze_non_lora
# ───────────────────────────────────────────────────────────

class TestFreezeNonLoRA(unittest.TestCase):

    def _setup(self):
        m = TinyModel()
        inject_lora(m, r=4, alpha=8.0, dropout=0.0,
                    target_modules=["q_proj", "v_proj"])
        freeze_non_lora(m)
        return m

    def test_only_lora_trainable(self):
        """After freeze_non_lora, only lora_A and lora_B require grad."""
        m = self._setup()
        for name, param in m.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertTrue(param.requires_grad, f"{name} should require grad")
            else:
                self.assertFalse(param.requires_grad, f"{name} should NOT require grad")

    def test_base_weight_unchanged_after_step(self):
        """Base weight does not change after an optimizer step."""
        m = self._setup()
        w_before = m.q_proj.weight.data.clone()
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
        m.q_proj(torch.randn(2, 64)).sum().backward()
        opt.step()
        self.assertTrue(torch.allclose(m.q_proj.weight.data, w_before),
                        "Base weight must not change (LoRA only updates A and B)")

    def test_lora_params_change_after_step(self):
        """lora_A DOES change after an optimizer step."""
        m = self._setup()
        a_before = m.q_proj.lora_A.data.clone()
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
        m.q_proj(torch.randn(2, 64)).sum().backward()
        opt.step()
        self.assertFalse(torch.allclose(m.q_proj.lora_A.data, a_before),
                         "lora_A should change after optimizer step")


# ───────────────────────────────────────────────────────────
# Suite 4: get_lora_state_dict
# ───────────────────────────────────────────────────────────

class TestGetLoRAStateDict(unittest.TestCase):

    def _setup(self):
        m = TinyModel()
        inject_lora(m, r=4, alpha=8.0, dropout=0.0,
                    target_modules=["q_proj", "v_proj"])
        return m

    def test_only_lora_keys(self):
        """State dict contains only lora_A and lora_B keys."""
        sd = get_lora_state_dict(self._setup())
        for k in sd:
            self.assertTrue("lora_A" in k or "lora_B" in k,
                            f"Unexpected key: {k}")

    def test_all_lora_params_present(self):
        """State dict has lora_A and lora_B for both q_proj and v_proj."""
        sd = get_lora_state_dict(self._setup())
        self.assertEqual(len([k for k in sd if "lora_A" in k]), 2)
        self.assertEqual(len([k for k in sd if "lora_B" in k]), 2)

    def test_tensors_on_cpu(self):
        """All tensors are on CPU regardless of model device."""
        sd = get_lora_state_dict(self._setup())
        for k, v in sd.items():
            self.assertEqual(v.device.type, "cpu", f"{k} must be on CPU")

    def test_tensors_are_clones(self):
        """Mutating the model does not affect the saved state dict."""
        m = self._setup()
        sd = get_lora_state_dict(m)
        # Corrupt lora_A in-place
        for name, p in m.named_parameters():
            if "lora_A" in name:
                p.data.fill_(999.0)
                break
        # Saved sd should be unaffected
        for k, v in sd.items():
            if "lora_A" in k:
                self.assertFalse(torch.all(v == 999.0),
                                 "State dict must be a clone, not a view")
                break


# ───────────────────────────────────────────────────────────
# Optional: model-loading integration tests (--full)
# ───────────────────────────────────────────────────────────

class TestModelIntegration(unittest.TestCase):
    """Requires downloading Qwen2.5-0.5B (~1GB). Run with --full."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from config import Config
            cfg = Config()
            print(f"\nLoading {cfg.model_name} for integration tests...")
            cls.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, trust_remote_code=True)
            cls.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, torch_dtype=torch.float16,
                device_map="cpu", trust_remote_code=True)
            cls.cfg = cfg
            cls._ok = True
        except Exception as e:
            print(f"\nSkipping integration tests: {e}")
            cls._ok = False

    def _skip(self):
        if not self._ok:
            self.skipTest("Model not available")

    def test_trainable_ratio(self):
        """LoRA injection leaves < 2% trainable parameters."""
        self._skip()
        import copy
        from utils import count_parameters
        m = copy.deepcopy(self.model)
        inject_lora(m, r=self.cfg.lora_r, alpha=self.cfg.lora_alpha,
                    dropout=0.0, target_modules=self.cfg.target_modules)
        freeze_non_lora(m)
        total, trainable = count_parameters(m)
        ratio = trainable / total
        print(f"\nTrainable ratio: {ratio:.4%}")
        self.assertLess(ratio, 0.02)
        self.assertGreater(ratio, 1e-4)

    def test_forward_finite(self):
        """LoRA-injected model produces finite logits."""
        self._skip()
        import copy
        m = copy.deepcopy(self.model)
        inject_lora(m, r=self.cfg.lora_r, alpha=self.cfg.lora_alpha,
                    dropout=0.0, target_modules=self.cfg.target_modules)
        freeze_non_lora(m)
        m.eval()
        inputs = self.tokenizer("Hello!", return_tensors="pt")
        with torch.no_grad():
            logits = m(**inputs).logits
        self.assertFalse(torch.isnan(logits).any(), "Logits must be finite")


# ───────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA unit tests")
    parser.add_argument("--full", action="store_true",
                        help="Include integration tests (requires ~1GB model download)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestLoRALinear, TestInjectLoRA, TestFreezeNonLoRA, TestGetLoRAStateDict]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    if args.full:
        suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))

    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
