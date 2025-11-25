import torch
import random
import tensorrt as trt
from cuda.bindings import runtime as cudart

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from transformers import CLIPTokenizer
import nvtx
import infer_cudagraph
import ctypes
import DDIM_scheduler

plugin_libs = ["plugin/build/libplugin.so"]
for plugin in plugin_libs:
    ctypes.cdll.LoadLibrary(plugin)


class hackathon():

    def load_all_engines(self, trt_engine_file_list):
        engine_table = {}
        for engine_file_name in trt_engine_file_list:
            with open(engine_file_name, "rb") as f:
                engine_str = f.read()
            engine = trt.Runtime(
                self.trt_logger).deserialize_cuda_engine(engine_str)
            context = engine.create_execution_context()
            engine_name = engine_file_name.split('.')[0]
            engine_table[engine_name] = (engine, context)
        return engine_table

    def control_outputs(self):
        inter_data_shapes = []

        batch_size = 2
        h = 32
        w = 48
        inter_data_shapes.append((1, 320, h, w))
        for i in range(2):
            inter_data_shapes.append((batch_size, 320, h, w))

        inter_data_shapes.append((batch_size, 320, h // 2, w // 2))

        for i in range(2):
            inter_data_shapes.append((batch_size, 640, h // 2, w // 2))

        inter_data_shapes.append((batch_size, 640, h // 4, w // 4))

        for i in range(2):
            inter_data_shapes.append((batch_size, 1280, h // 4, w // 4))

        for i in range(4):
            inter_data_shapes.append((batch_size, 1280, h // 8, w // 8))

        control_outputs = []
        for i in range(13):
            temp = torch.randn(inter_data_shapes[i],
                               dtype=torch.float16).cuda().contiguous()
            control_outputs.append(temp)
        return control_outputs

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.scheduler = DDIM_scheduler.DDIM()
        self.timesteps = self.scheduler.ddim_timesteps
        # timesteps = np.array([1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701, 751, 801, 851, 901, 951])
        self.time_range = self.timesteps
        self.total_steps = self.timesteps.shape[0]
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        batch_size = 1

        engine_name_list = [
            "sd_control_fp16.trt", "sd_encoder_fp16.trt",
            "sd_decoder_fp16.trt", "hint_fp16.trt", "clip_fp16.trt",
            "decoder_fp16.trt"
        ]
        engine_table = self.load_all_engines(engine_name_list)

        #---------- load hint engine -----------------------------
        self.hint_context = engine_table["hint_fp16"][1]
        self.hint_out = torch.randn((batch_size, 320, 32, 48),
                                    dtype=torch.float16).cuda().contiguous()

        #---------- load clip engine ------------------------------
        self.clip_context = engine_table["clip_fp16"][1]
        self.clip_out = torch.randn((batch_size * 2, 77, 768),
                                    dtype=torch.float16).cuda().contiguous()

        #---------- load sd_control engine -----------------------------
        self.sd_control_context = engine_table["sd_control_fp16"][1]
        self.img = torch.randn((batch_size, 4, 32, 48),
                               device="cuda").to(torch.float16).contiguous()
        self.step = torch.ones((batch_size, ),
                               dtype=torch.int32).cuda().contiguous()
        self.control_outputs_list = self.control_outputs()

        #---------- load sd_encoder engine ------------------------------
        self.sd_encoder_context = engine_table["sd_encoder_fp16"][1]
        self.encoder_outputs = self.control_outputs()
        self.encoder_outputs.pop(12)
        self.middle_h = torch.randn((2, 1280, 4, 6),
                                    dtype=torch.float16).cuda().contiguous()
        self.emb_outputs = torch.randn(
            (1, 1280), dtype=torch.float16).cuda().contiguous()

        #--------- load sd_decoder engine -------------------------------
        self.sd_decoder_context = engine_table["sd_decoder_fp16"][1]
        self.index = torch.ones((batch_size, ),
                                dtype=torch.int32).cuda().contiguous()
        self.latent = torch.zeros(2, 4, 32, 48,
                                  dtype=torch.float16).cuda().contiguous()

        #---------- load vae engine -------------------------------
        self.vae_context = engine_table["decoder_fp16"][1]
        self.vae_out = torch.zeros((batch_size, 256, 384, 3),
                                   dtype=torch.uint8).cuda().contiguous()

        _, self.stream_0 = cudart.cudaStreamCreate()
        _, self.stream_1 = cudart.cudaStreamCreate()
        _, self.event = cudart.cudaEventCreateWithFlags(
            cudart.cudaEventDisableTiming)
        self.sd_control_cudagraph = infer_cudagraph.cudagraph_engine(
            engine_table["sd_control_fp16"][0],
            engine_table["sd_control_fp16"][1],
            [self.img, self.hint_out, self.step, self.clip_out] +
            self.control_outputs_list, self.stream_0)

        self.sd_encoder_cudagraph = infer_cudagraph.cudagraph_engine(
            engine_table["sd_encoder_fp16"][0],
            engine_table["sd_encoder_fp16"][1],
            [self.img, self.step, self.clip_out] + [self.emb_outputs] +
            self.encoder_outputs + [self.middle_h], self.stream_1)

        self.sd_decoder_cudagraph = infer_cudagraph.cudagraph_engine(
            engine_table["sd_decoder_fp16"][0],
            engine_table["sd_decoder_fp16"][1], [self.clip_out] +
            self.control_outputs_list + self.encoder_outputs +
            [self.middle_h, self.emb_outputs, self.latent], self.stream_0)

        self.vae_cudagraph = infer_cudagraph.cudagraph_engine(
            engine_table["decoder_fp16"][0], self.vae_context,
            [self.img, self.vae_out], self.stream_0)

        print("finished")

    @torch.no_grad()
    def ddim_sampling(self, batch_size):

        # print(f"Running DDIM Sampling with {total_steps} timesteps")
        img = torch.randn((batch_size, 4, 32, 48),
                          device="cuda").to(torch.float16)
        self.img.copy_(img)
        for i, step in enumerate(self.time_range):
            index = self.total_steps - i - 1
            self.index[:] = index
            self.step[:] = step
            rng = nvtx.start_range(message="controlnet")

            self.sd_encoder_cudagraph.infer()
            cudart.cudaEventRecord(self.event, self.stream_1)
            self.sd_control_cudagraph.infer()
            # cudart.cudaStreamSynchronize(self.sd_control_cudagraph.stream)
            # cudart.cudaStreamSynchronize(self.sd_encoder_cudagraph.stream)
            cudart.cudaStreamWaitEvent(self.stream_0, self.event,
                                       cudart.cudaEventWaitDefault)
            self.sd_decoder_cudagraph.infer()
            cudart.cudaStreamSynchronize(self.stream_0)
            nvtx.end_range(rng)
            x = DDIM_scheduler.step(self.img, self.latent, index,
                                    self.scheduler.ddim_alphas,
                                    self.scheduler.ddim_alphas_prev,
                                    self.scheduler.ddim_sqrt_one_minus_alphas)
            self.img.copy_(x)
        return self.img

    @torch.no_grad()
    def inference_hint(self, detected_map):
        buffer_device = []
        buffer_device.append(detected_map.contiguous().data_ptr())
        buffer_device.append(self.hint_out.contiguous().data_ptr())

        rng = nvtx.start_range(message="hint")
        self.hint_context.execute_v2(buffer_device)
        nvtx.end_range(rng)

    @torch.no_grad()
    def inference_clip(self, prompts):
        tokens = self.tokenizer(prompts,
                                truncation=True,
                                max_length=77,
                                return_length=True,
                                return_overflowing_tokens=False,
                                padding="max_length",
                                return_tensors="pt")
        input_ids = tokens["input_ids"].cuda().type(torch.int32)
        buffer_device = []
        buffer_device.append(input_ids.contiguous().data_ptr())
        buffer_device.append(self.clip_out.contiguous().data_ptr())

        rng = nvtx.start_range(message="clip")
        self.clip_context.execute_v2(buffer_device)
        nvtx.end_range(rng)

    @torch.no_grad()
    def inference_vae(self):
        rng = nvtx.start_range(message="vae")
        self.vae_cudagraph.infer()
        nvtx.end_range(rng)

    @nvtx.annotate()
    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples,
                image_resolution, ddim_steps, guess_mode, strength, scale,
                seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = torch.from_numpy(detected_map).cuda()
            self.inference_hint(detected_map)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            self.inference_clip([prompt + ', ' + a_prompt, n_prompt] *
                                num_samples + [n_prompt] * num_samples)

            #self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # this will not change in hackathon
            self.ddim_sampling(num_samples)

            self.inference_vae()
            results = [self.vae_out.cpu().numpy()[0]]
        return results
