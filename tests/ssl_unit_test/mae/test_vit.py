import pytest
from functools import partial

import torch
from torch import nn
from nvidia_tao_pytorch.ssl.mae.model.vit import (
    VisionTransformer, vit_base_patch16, vit_large_patch16, vit_huge_patch14,
    vit_group
)

# Test cases for VisionTransformer class
class TestVisionTransformer:
    def test_init(self):
        model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768)
        assert hasattr(model, 'fc_norm')
        assert not hasattr(model, 'norm')

    @pytest.mark.parametrize("global_pool", ["", "token"])
    def test_forward_features(self, global_pool):
        batch_size = 2
        image_size = (224, 224)

        model = VisionTransformer(global_pool=global_pool, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=768)
        x = torch.randn(batch_size, 3, image_size[0], image_size[1])
        outcome = model.forward_features(x)

        if global_pool:
            assert outcome.shape == (batch_size, 1, model.embed_dim)
        else:
            assert outcome.shape == (batch_size, model.embed_dim)

# Test cases for vit_base_patch16 function
class TestVitBasePatch16:
    def test_model(self):
        model = vit_base_patch16()
        assert isinstance(model, VisionTransformer)

    @pytest.mark.parametrize("global_pool", ['', 'token'])
    def test_forward_features(self, global_pool):
        batch_size = 2
        image_size = (224, 224)
        patch_size = 16

        model = vit_base_patch16(global_pool=global_pool)
        x = torch.randn(batch_size, 3, image_size[0], image_size[1])
        outcome = model.forward_features(x)

        if global_pool:
            assert outcome.shape == (batch_size, 1, model.embed_dim)
        else:
            assert outcome.shape == (batch_size, model.embed_dim)

# Test cases for vit_large_patch16 function
class TestVitLargePatch16:
    def test_model(self):
        model = vit_large_patch16()
        assert isinstance(model, VisionTransformer)

    @pytest.mark.parametrize("global_pool", ['', 'token'])
    def test_forward_features(self, global_pool):
        batch_size = 2
        image_size = (224, 224)
        patch_size = 16

        model = vit_large_patch16(global_pool=global_pool)
        x = torch.randn(batch_size, 3, image_size[0], image_size[1])
        outcome = model.forward_features(x)

        if global_pool:
            assert outcome.shape == (batch_size, 1, model.embed_dim)
        else:
            assert outcome.shape == (batch_size, model.embed_dim)

# Test cases for vit_huge_patch14 function
class TestVitHugePatch14:
    def test_model(self):
        model = vit_huge_patch14()
        assert isinstance(model, VisionTransformer)

    @pytest.mark.parametrize("global_pool", ['', 'token'])
    def test_forward_features(self, global_pool):
        batch_size = 2
        image_size = (224, 224)
        patch_size = 16

        model = vit_huge_patch14(global_pool=global_pool)
        x = torch.randn(batch_size, 3, image_size[0], image_size[1])
        outcome = model.forward_features(x)

        if global_pool:
            assert outcome.shape == (batch_size, 1, model.embed_dim)
        else:
            assert outcome.shape == (batch_size, model.embed_dim)

# Test cases for vit_group list
class TestVitGroup:
    @pytest.mark.parametrize("model_func", vit_group)
    def test_model(self, model_func):
        model = model_func()
        assert isinstance(model, VisionTransformer)
