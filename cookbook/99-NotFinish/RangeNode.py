#   copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os

import torch


class GaussianUpsampling(torch.nn.Module):
    """Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301
    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """Upsample hidden states according to durations.
        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor): Mask tensor (B, T_feats).
            d_masks (Tensor): Mask tensor (B, T_text).
        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).
        """
        B = ds.size(0)

        if h_masks is None:
            #T_feats = ds.sum().item()
            T_feats = ds.sum()
            # by wili
            # if we use "ds.sum().item()" rather than "ds.sum()", TensorRT is able to build engine successfully,
            # but the engine is incorrect, because that will solid the size of T_feats as a buildtime constant, not data-dependent
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).float()
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        logging.warning(f"p_attn shape is {p_attn.shape}")
        logging.warning(f"hs shape is {hs.shape}")
        hs_out = torch.matmul(p_attn, hs)
        logging.warning(f"hs shape is {hs.shape}")
        return hs_out


def export_model():

    model = GaussianUpsampling()

    hs = torch.randn(1, 128, 256, dtype=torch.float)
    ds = torch.randint(1, 120, (1, 128))

    logging.warning(f"ds is  {ds}")

    dummy_inputs = (hs, ds)

    torch_out = model(hs, ds)

    torch.onnx.export(model, dummy_inputs, "GaussUpSample.onnx", verbose=True,
        opset_version=12,
        input_names=["hs", "ds"],
        output_names=["hs_out"],
        dynamic_axes={"hs":{1:'txt_len'}, "ds":{1:"txt_len"}, "hs_out":{1:"len"}})

if __name__ == '__main__':
    export_model()
    os.system("trtexec --onnx=GaussUpSample.onnx --minShapes=hs:1x1x256,ds:1x1 --optShapes=hs:1x16x256,ds:1x16 --maxShapes=hs:1x64x256,ds:1x64 --verbose")

# By wili
# Reference model: https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/jets/length_regulator.py
# Error information
# [E] Error[10]: Could not find any implementation for node {ForeignNode[(Unnamed Layer* 56) [Constant].../MatMul]}.
# [E] Error[10]: [optimizer.cpp::computeCosts::3873] Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[(Unnamed Layer* 56) [Constant].../MatMul]}.)
# [E] Engine could not be created from network
# [E] Building engine failed
# [E] Failed to create engine from model or file.
# [E] Engine set up failed
# &&&& FAILED TensorRT.trtexec [TensorRT v8600] # trtexec --onnx=GaussUpSample.onnx --minShapes=hs:1x1x256,ds:1x1 --optShapes=hs:1x16x256,ds:1x16 --maxShapes=hs:1x64x256,ds:1x64 --verbose
