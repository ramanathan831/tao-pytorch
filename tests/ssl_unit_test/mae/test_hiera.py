import pytest
import torch
from nvidia_tao_pytorch.ssl.mae.model.hiera import (
    Hiera, hiera_tiny_224, hiera_small_224,
    hiera_base_224, hiera_large_224, hiera_huge_224
)

@pytest.fixture(scope="module", params=[hiera_tiny_224, hiera_small_224, hiera_base_224, hiera_large_224, hiera_huge_224])
def model(request):
    return request.param()

@pytest.fixture(scope="module")
def input_data():
    return torch.randn(1, 3, 224, 224)

def test_forward_pass(model, input_data):
    output = model(input_data)
    assert output is not None, "Forward pass failed"

def test_output_shape(model, input_data):
    output = model(input_data)
    assert output.shape == torch.Size([1, 1000]), "Output shape mismatch"

def test_forward_with_mask(model, input_data):
    hh, ww = model.mask_spatial_shape
    mask = torch.randint(low=0, high=2, size=(input_data.shape[0], hh * ww)).bool()
    output = model(input_data, mask)
    assert output is not None, "Forward pass with mask failed"

def test_forward_with_return_intermediates(model, input_data):
    output, intermediates = model(input_data, return_intermediates=True)
    assert output is not None, "Forward pass with return_intermediates failed"
    assert len(intermediates) > 0, "No intermediate outputs returned"