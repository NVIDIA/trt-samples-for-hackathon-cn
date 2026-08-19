# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export a small FCN-shaped segmentation model, and the test image the C++ side reads.

The upstream `quickstart/SemanticSegmentation` tutorial runs FCN-ResNet101 on a downloaded
photograph. That is a 200 MB download plus torchvision weights, and this cookbook does not
download at run time, so this script *builds* an equivalent-shaped model instead: an
encoder/decoder that consumes `[1, 3, H, W]` and emits `[1, N_CLASS, H, W]` logits, which is
the only thing the C++ runtime cares about.

The image is written as binary PPM (P6), which needs no image library on either side.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(31193)
torch.manual_seed(31193)

N_CLASS = 4
N_HEIGHT = 224
N_WIDTH = 224
OPSET = 17

output_path = Path(__file__).parent
onnx_file = output_path / "model-segmentation.onnx"
image_file = output_path / "data-input.ppm"

# The four colours the classes correspond to, in the same 0-255 space as the image
REFERENCE_COLOUR = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]], dtype=np.float32)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class Segmenter(nn.Module):
    """Down twice, upsample back, per-pixel logits out -- the shape of an FCN.

    An FCN with *random* weights argmaxes every pixel to the same class, which makes for a
    demo that looks broken. So the classifier head is not random: nearest-reference-colour
    classification is exactly a 1x1 convolution, because

        argmin_k |x - c_k|^2  ==  argmax_k (2 c_k . x - |c_k|^2)

    which is a linear layer with weight `2 c_k` and bias `-|c_k|^2`. The strided
    encoder/decoder branch is kept (it is what puts `Convolution` and `Resize` layers in the
    engine) but scaled down so it perturbs the colour decision rather than replacing it.
    """

    def __init__(self, n_class: int) -> None:
        super().__init__()
        self.down1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.middle = nn.Conv2d(32, 32, 3, padding=1)
        self.context = nn.Conv2d(32, n_class, 1)
        self.colour = nn.Conv2d(3, n_class, 1)

        # Reference colours expressed in the normalised space the network actually sees
        normalised = (REFERENCE_COLOUR[:n_class] / 255.0 - IMAGE_MEAN) / IMAGE_STD
        with torch.no_grad():
            self.colour.weight.copy_(torch.from_numpy(2.0 * normalised).view(n_class, 3, 1, 1))
            self.colour.bias.copy_(torch.from_numpy(-(normalised ** 2).sum(axis=1)))

    def forward(self, x):
        y = F.relu(self.down1(x))
        y = F.relu(self.down2(y))
        y = F.relu(self.middle(y))
        y = self.context(y)
        y = F.interpolate(y, scale_factor=4, mode="bilinear", align_corners=False)
        return self.colour(x) + 0.1 * y

def write_ppm(path: Path, image: np.ndarray) -> None:
    """Write `[H, W, 3]` uint8 as binary PPM (P6)."""
    height, width, _ = image.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return

def make_image() -> np.ndarray:
    """Four quadrants with different colour statistics, so the argmax map is readable."""
    image = np.zeros((N_HEIGHT, N_WIDTH, 3), dtype=np.uint8)
    half_h, half_w = N_HEIGHT // 2, N_WIDTH // 2
    image[:half_h, :half_w] = [200, 40, 40]
    image[:half_h, half_w:] = [40, 200, 40]
    image[half_h:, :half_w] = [40, 40, 200]
    image[half_h:, half_w:] = [200, 200, 40]
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def main() -> None:
    model = Segmenter(N_CLASS).eval()
    dummy = torch.zeros(1, 3, N_HEIGHT, N_WIDTH)
    torch.onnx.export(
        model,
        (dummy, ),
        onnx_file,
        dynamo=False,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Wrote {onnx_file.name}: input [1, 3, {N_HEIGHT}, {N_WIDTH}] -> output [1, {N_CLASS}, {N_HEIGHT}, {N_WIDTH}]")

    image = make_image()
    write_ppm(image_file, image)
    print(f"Wrote {image_file.name}: {image.shape[1]}x{image.shape[0]} P6 PPM")

    # The reference the C++ side is checked against
    with torch.no_grad():
        tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        logits = model((tensor - mean) / std)
        class_map = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    np.save(output_path / "data-reference.npy", class_map)
    print(f"Reference class map: {np.bincount(class_map.reshape(-1), minlength=N_CLASS).tolist()} pixels per class")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
