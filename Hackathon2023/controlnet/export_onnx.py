#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os
import torch
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity
import surgeon_graph
import tensorrt as trt

disable_verbosity()

save_dir = "onnx"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(
    load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth',
                    location='cuda'))
model = model.cuda()


class CLIPNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = model.cond_stage_model.tokenizer
        self.transformer = model.cond_stage_model.transformer

    def forward(self, x):
        x = self.transformer(x, None, None, None, False, None)
        x = x.last_hidden_state
        return x


clip = CLIPNet()
x_ = torch.ones((2, 77), dtype=torch.int32).to("cuda")
torch.onnx.export(clip,
                  x_,
                  os.path.join(save_dir, "clip.onnx"),
                  input_names=["token"],
                  output_names=["c_in"],
                  opset_version=18)
surgeon_graph.clip_rm_inf_and_change_inout_type(
    os.path.join(save_dir, "clip.onnx"), os.path.join(save_dir,
                                                      "clip_opt.onnx"))
os.system("trtexec --onnx=" + save_dir +
          "/clip_opt.onnx --fp16 --saveEngine=clip_fp16.trt")


class GuidedHintNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = model.control_model.input_hint_block

    def forward(self, x):
        x = x.float() / 255.0
        x = torch.cat([x, x, x], dim=1)  # 1x3x256x384
        x = self.net(x, None, None)  # 未用到  context
        return x


class ControlNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = model.control_model
        self.diffusion_model = model.model.diffusion_model
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def forward(self, context, control_outputs, encoder_outputs, middle_h,
                emb_outputs):
        y = self.diffusion_model(None, None, context, control_outputs,
                                 encoder_outputs, middle_h, emb_outputs, False)
        return y


x_noisy = torch.randn((1, 4, 32, 48), dtype=torch.float32).to("cuda")
hint = torch.randint(0, 126, (1, 1, 256, 384), dtype=torch.uint8).to("cuda")
guided_hint = torch.randn((1, 320, 32, 48), dtype=torch.float32).to("cuda")
t = torch.ones([1], dtype=torch.int32).to("cuda")
context = torch.randn((2, 77, 768), dtype=torch.float32).to("cuda")
idx = torch.ones([1], dtype=torch.int32).to("cuda")

controlnet = ControlNet()
control_out = controlnet.control_model(x_noisy, guided_hint, t, context)
control_output_names = []
for i in range(13):
    control_output_names.append("out_" + str(i))
torch.onnx.export(controlnet.control_model, (x_noisy, guided_hint, t, context),
                  os.path.join(save_dir, "sd_control.onnx"),
                  input_names=['x_in', "h_in", "t_in", "c_in"],
                  output_names=control_output_names,
                  opset_version=18)
surgeon_graph.add_plugins_and_change_inout_type(
    os.path.join(save_dir, "sd_control.onnx"),
    os.path.join(save_dir, "sd_control_opt.onnx"), True)
os.system(
    "trtexec --onnx=" + save_dir +
    "/sd_control_opt.onnx --saveEngine=sd_control_fp16.trt --fp16  --staticPlugins=./plugin/build/libplugin.so"
)

hs = controlnet.diffusion_model(x_noisy, t, context)
encoder_output_names = []
for i in range(12):
    encoder_output_names.append("h_" + str(i))
torch.onnx.export(controlnet.diffusion_model, (x_noisy, t, context),
                  os.path.join(save_dir, "sd_encoder.onnx"),
                  input_names=['x_in', "t_in", "c_in"],
                  output_names=["emb"] + encoder_output_names + ["middle_h"],
                  opset_version=18)
surgeon_graph.add_plugins_and_change_inout_type(
    os.path.join(save_dir, "sd_encoder.onnx"),
    os.path.join(save_dir, "sd_encoder_opt.onnx"), True)
os.system(
    "trtexec --onnx=" + save_dir +
    "/sd_encoder_opt.onnx --fp16 --saveEngine=sd_encoder_fp16.trt  --staticPlugins=./plugin/build/libplugin.so"
)

h = hs.pop(13)
emb = hs.pop(0)
decoder_input_names = [
    "c_in"
] + control_output_names + encoder_output_names + ["middle_h", "emb"]
torch.onnx.export(controlnet, (context, control_out, hs, h, emb),
                  os.path.join(save_dir, "sd_decoder.onnx"),
                  input_names=decoder_input_names,
                  output_names=["latent"],
                  opset_version=18)
surgeon_graph.add_plugins_and_change_inout_type(
    os.path.join(save_dir, "sd_decoder.onnx"),
    os.path.join(save_dir, "sd_decoder_opt.onnx"), True)
os.system(
    "trtexec --onnx=" + save_dir +
    "/sd_decoder_opt.onnx --fp16 --saveEngine=sd_decoder_fp16.trt  --staticPlugins=./plugin/build/libplugin.so"
)

controlnet0 = GuidedHintNet()
controlnet0.eval()
torch.onnx.export(controlnet0,
                  hint,
                  os.path.join(save_dir, "hint.onnx"),
                  input_names=["canny"],
                  output_names=["h_in"],
                  opset_version=18)
surgeon_graph.only_change_inout_type(os.path.join(save_dir, "hint.onnx"),
                                     os.path.join(save_dir, "hint_opt.onnx"))
os.system("trtexec --onnx=" + save_dir +
          "/hint_opt.onnx --fp16 --saveEngine=hint_fp16.trt")


class Decoder(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_quant_conv = model.first_stage_model.post_quant_conv
        self.decoder = model.first_stage_model.decoder

    def forward(self, x):
        x = 1. / 0.18215 * x
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 3, 1)
        x = x * 127.5 + 127.5
        x = x.clip(0, 255).type(torch.uint8)
        return x


decoder = Decoder()
decoder.eval()
z = torch.randn((1, 4, 32, 48), dtype=torch.float32).to("cuda")
torch.onnx.export(decoder,
                  z,
                  os.path.join(save_dir, "decoder.onnx"),
                  input_names=["latent"],
                  output_names=["img"],
                  opset_version=18)
surgeon_graph.add_plugins_and_change_inout_type(
    os.path.join(save_dir, "decoder.onnx"),
    os.path.join(save_dir, "decoder_opt.onnx"))
os.system(
    "trtexec --onnx=" + save_dir +
    "/decoder_opt.onnx --fp16 --saveEngine=decoder_fp16.trt --staticPlugins=./plugin/build/libplugin.so"
)
