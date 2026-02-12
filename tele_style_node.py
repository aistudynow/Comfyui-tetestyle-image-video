import torch
import os
import sys
import hashlib
import gc
import time
from collections import OrderedDict
import folder_paths
import numpy as np
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler
)

from .telestylevideo_transformer import WanTransformer3DModel
from .telestylevideo_pipeline import WanPipeline


def _tensor_to_pil(image_tensor):
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0).cpu()
    arr = (image_tensor.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(image):
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


_IMAGE_PIPE_CACHE = {}
_IMAGE_RESULT_CACHE = OrderedDict()
_IMAGE_RESULT_CACHE_MAX = 12
_ATTENTION_WARNED_MODES = set()
_DIFFSYNTH_NOISE_PATTERNS = (
    "No qwen_image_blockwise_controlnet models available. This is not an error.",
    "No siglip2_image_encoder models available. This is not an error.",
    "No dinov3_image_encoder models available. This is not an error.",
    "No qwen_image_image2lora_style models available. This is not an error.",
    "No qwen_image_image2lora_coarse models available. This is not an error.",
    "No qwen_image_image2lora_fine models available. This is not an error.",
    "Fused LoRA layers cannot be cleared by `pipe.clear_lora()`.",
    "Fused LoRA layers cannot be cleared by pipe.clear_lora().",
)


class _FilteredStream:
    def __init__(self, target, patterns):
        self._target = target
        self._patterns = patterns
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(p in line for p in self._patterns):
                self._target.write(line + "\n")
        return len(data)

    def flush(self):
        if self._buffer:
            line = self._buffer
            self._buffer = ""
            if not any(p in line for p in self._patterns):
                self._target.write(line)
        self._target.flush()


class _SuppressKnownDiffSynthLogs:
    def __enter__(self):
        self._out = _FilteredStream(sys.stdout, _DIFFSYNTH_NOISE_PATTERNS)
        self._err = _FilteredStream(sys.stderr, _DIFFSYNTH_NOISE_PATTERNS)
        self._ctx_out = redirect_stdout(self._out)
        self._ctx_err = redirect_stderr(self._err)
        self._ctx_out.__enter__()
        self._ctx_err.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ctx_err.__exit__(exc_type, exc_val, exc_tb)
        self._ctx_out.__exit__(exc_type, exc_val, exc_tb)
        self._out.flush()
        self._err.flush()
        return False


def _get_sdp_context(attention_mode):
    mode = _normalize_attention_mode(attention_mode)
    if mode == "sdpa" or not torch.cuda.is_available():
        return nullcontext()

    try:
        if mode == "flash_attn":
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        if mode == "sage_attn":
            # SageAttention is configured at ComfyUI startup (`--use-sage-attention`).
            if mode not in _ATTENTION_WARNED_MODES:
                print("TeleStyle Info: `sage_attn` uses ComfyUI global backend. Start ComfyUI with --use-sage-attention.")
                _ATTENTION_WARNED_MODES.add(mode)
            return nullcontext()
    except Exception:
        pass

    return nullcontext()


def _normalize_attention_mode(mode):
    mode = str(mode or "sdpa").strip().lower()
    if mode in ("sdpa", "default", "auto", "", "mem_efficient", "math"):
        return "sdpa"
    if mode in ("flash_attn", "flash", "flash_attention", "flash-attn"):
        return "flash_attn"
    if mode in ("sage_attn", "sage", "sageattention", "segattention"):
        return "sage_attn"
    return "sdpa"


def _is_invalid_black_tensor(t):
    if t is None:
        return True
    if torch.isnan(t).any() or torch.isinf(t).any():
        return True
    max_v = float(t.max().item())
    min_v = float(t.min().item())
    if max_v <= 1e-4:
        return True
    if (max_v - min_v) <= 1e-5 and max_v < 0.02:
        return True
    return False


def _benchmark_qwen_attention(pipe, amp_dtype):
    if not torch.cuda.is_available():
        return "sdpa", {}

    prompt = "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1."
    images = [
        Image.new("RGB", (256, 256), (127, 127, 127)),
        Image.new("RGB", (256, 256), (180, 120, 90)),
    ]

    timings = {}
    modes = ("flash_attn", "sdpa")
    for mode in modes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            attn_context = _get_sdp_context(mode)
            with torch.inference_mode(), attn_context, torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = pipe(
                    prompt,
                    edit_image=images,
                    seed=123,
                    num_inference_steps=2,
                    height=256,
                    width=256,
                    edit_image_auto_resize=False,
                    cfg_scale=1.0,
                )
            torch.cuda.synchronize()
            if isinstance(out, list):
                out = out[0]
            tensor = _pil_to_tensor(out)
            if _is_invalid_black_tensor(tensor):
                print(f"TeleStyle Warning: Attention benchmark mode `{mode}` produced invalid output; skipping.")
                continue
            timings[mode] = time.perf_counter() - start
        except Exception as e:
            print(f"TeleStyle Warning: Attention benchmark mode `{mode}` failed: {e}")

    if not timings:
        return "sdpa", timings
    best_mode = min(timings, key=timings.get)
    return best_mode, timings


def _pil_hash(image):
    # Deterministic cache key for resized PIL images.
    h = hashlib.sha1()
    h.update(f"{image.mode}:{image.size[0]}x{image.size[1]}".encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


def _clear_image_runtime_cache():
    # Keep only one heavy Qwen pipeline in memory to avoid OOM when options change.
    _IMAGE_PIPE_CACHE.clear()
    _IMAGE_RESULT_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _comfy_vram_cleanup(cleanup_mode):
    mode = str(cleanup_mode or "none").strip().lower()
    if mode in ("none", "off", ""):
        return

    try:
        import comfy.model_management as mm
    except Exception:
        mm = None

    try:
        if mode == "unload_all_models" and mm is not None and hasattr(mm, "unload_all_models"):
            mm.unload_all_models()

        if mm is not None and hasattr(mm, "soft_empty_cache"):
            try:
                mm.soft_empty_cache(force=True)
            except TypeError:
                mm.soft_empty_cache()
    except Exception as e:
        print(f"TeleStyle Warning: Comfy VRAM cleanup failed: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class TeleStyleLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"dtype": (["fp16", "bf16"], {"default": "bf16"})}}

    RETURN_TYPES = ("TELE_STYLE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_all"
    CATEGORY = "TeleStyle"

    def load_all(self, dtype):
        device = torch.device("cuda")
        repo_id = "Danzelus/TeleStyle_comfy"
        base_path = os.path.join(folder_paths.models_dir, "telestyle_models")
        
        if dtype == "bf16":
            target_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16
            vae_dtype = torch.float16
        
        files = [
            "weights/dit.ckpt", "weights/prompt_embeds.pth",
            "Wan2.1-T2V-1.3B-Diffusers/transformer_config.json",
            "Wan2.1-T2V-1.3B-Diffusers/vae/config.json",
            "Wan2.1-T2V-1.3B-Diffusers/vae/diffusion_pytorch_model.safetensors",
            "Wan2.1-T2V-1.3B-Diffusers/scheduler/scheduler_config.json"
        ]
        
        for f in files:
            dest = os.path.join(base_path, f)
            if not os.path.exists(dest):
                try:
                    hf_hub_download(repo_id=repo_id, filename=f, local_dir=base_path, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"TeleStyle Warning: Could not download {f}: {e}")

        wan_path = os.path.join(base_path, "Wan2.1-T2V-1.3B-Diffusers")
        
        vae = AutoencoderKLWan.from_pretrained(os.path.join(wan_path, "vae"), torch_dtype=vae_dtype).to(device)
        
        config = OmegaConf.to_container(OmegaConf.load(os.path.join(wan_path, "transformer_config.json")))
        transformer = WanTransformer3DModel(**config)
        
        ckpt_path = os.path.join(base_path, "weights/dit.ckpt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if "transformer_state_dict" in sd:
                sd = sd["transformer_state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            transformer.load_state_dict(sd)
        
        transformer.to(device, dtype=target_dtype)

        embeds_path = os.path.join(base_path, "weights/prompt_embeds.pth")
        if os.path.exists(embeds_path):
            loaded_embeds = torch.load(embeds_path, map_location="cpu")
            if isinstance(loaded_embeds, dict):
                p_embeds = loaded_embeds.get("prompt_embeds", loaded_embeds)
                n_embeds = loaded_embeds.get("negative_prompt_embeds", torch.zeros_like(p_embeds))
            else:
                p_embeds = loaded_embeds
                n_embeds = torch.zeros_like(p_embeds)
        else:
            p_embeds = torch.zeros(1, 1, 4096, dtype=vae_dtype)
            n_embeds = torch.zeros(1, 1, 4096, dtype=vae_dtype)

        scheduler_config_path = os.path.join(wan_path, "scheduler")
        scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(scheduler_config_path)
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config, shift=3.0)
        except TypeError:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_config_path)

        return ({
            "vae": vae, 
            "transformer": transformer, 
            "scheduler": scheduler, 
            "p_embeds": p_embeds, 
            "n_embeds": n_embeds, 
            "dtype": vae_dtype,
            "device": device,
            "target_dtype": target_dtype,
            "vae_dtype": vae_dtype,
        },)

class TeleStyleVideoInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TELE_STYLE_MODEL",),
                "video_frames": ("IMAGE",),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 42}),
                # Backward-compatibility for old workflows that stored an extra "fixed" widget.
                "time_schedule": (["fixed"], {"default": "fixed"}),
                "scheduler": (["FlowMatchEuler", "UniPC", "DPM++"], {"default": "DPM++"}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "acceleration": (["sdpa", "flash_attn", "sage_attn"], {"default": "sdpa"}),
            },
            "optional": {
                "style_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TeleStyle"

    def process(self, model, video_frames, steps, cfg, seed, time_schedule, scheduler, fast_mode, enable_tiling, acceleration, style_image=None):
        device = torch.device("cuda")
        m = model
        dtype = m["dtype"]
        
        if enable_tiling or video_frames.shape[0] > 16:
            m["vae"].enable_tiling()
            m["vae"].enable_slicing()
        else:
            m["vae"].disable_tiling()
            m["vae"].disable_slicing()

        base_config = m["scheduler"].config
        scheduler_map = {
            "FlowMatchEuler": FlowMatchEulerDiscreteScheduler,
            "UniPC": UniPCMultistepScheduler,
            "DPM++": DPMSolverMultistepScheduler,
        }
        Cls = scheduler_map.get(scheduler, FlowMatchEulerDiscreteScheduler)

        if scheduler == "UniPC":
            pipe_scheduler = Cls.from_config(
                base_config,
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type="flow_prediction",
                flow_shift=3.0,
                use_flow_sigmas=True,
            )
        elif scheduler == "DPM++":
            pipe_scheduler = Cls.from_config(
                base_config,
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="scaled_linear",
                solver_order=2,
                prediction_type="flow_prediction",
                algorithm_type="dpmsolver++",
                use_flow_sigmas=True,
                flow_shift=3.0,
            )
        else:
            try:
                pipe_scheduler = Cls.from_config(base_config, shift=3.0)
            except TypeError:
                pipe_scheduler = Cls.from_config(base_config)

        pipe = WanPipeline(transformer=m["transformer"], vae=m["vae"], scheduler=pipe_scheduler)
        
        if fast_mode:
            pipe.to(device)
        else:
            pipe.enable_sequential_cpu_offload()

        p_embeds = m["p_embeds"].to(device, dtype=dtype)
        n_embeds = m["n_embeds"].to(device, dtype=dtype)

        F_in = video_frames.shape[0]
        target_frames = ((F_in - 1) // 4 + 1) * 4 + 1
        if ((target_frames - 1) // 4 + 1) == 2:
            target_frames += 4
        needed = target_frames - F_in
        if needed > 0:
            last_frame = video_frames[-1:].repeat(needed, 1, 1, 1)
            video_frames = torch.cat([video_frames, last_frame], dim=0)

        H, W = video_frames.shape[1], video_frames.shape[2]
        H_new = (H // 16) * 16
        W_new = (W // 16) * 16
        if H_new != H or W_new != W:
            video_frames = video_frames[:, :H_new, :W_new, :]
            if style_image is not None:
                style_image = style_image[:, :H_new, :W_new, :]

        src = video_frames.permute(3, 0, 1, 2).unsqueeze(0).to(device, dtype=dtype)
        src = (src - 0.5) * 2.0

        if style_image is None:
            style_image = torch.rand((1, H_new, W_new, 3), dtype=video_frames.dtype, device=video_frames.device)

        ref = style_image.permute(3, 0, 1, 2).unsqueeze(0)
        ref = ref[:, :, :1, :, :].to(device, dtype=dtype)
        ref = (ref - 0.5) * 2.0

        acceleration = _normalize_attention_mode(acceleration)
        attn_context = _get_sdp_context(acceleration)
        if acceleration == "sage_attn":
            if "sage_attn_video" not in _ATTENTION_WARNED_MODES:
                print("TeleStyle Info: `sage_attn` uses ComfyUI global backend. Start ComfyUI with --use-sage-attention.")
                _ATTENTION_WARNED_MODES.add("sage_attn_video")
        with torch.no_grad():
            s_lat = m["vae"].encode(src).latent_dist.mode()
            f_lat = m["vae"].encode(ref).latent_dist.mode()

            vae_config = m["vae"].config
            latents_mean = torch.tensor(vae_config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype=dtype)
            latents_std = torch.tensor(vae_config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype=dtype)

            s_lat = (s_lat - latents_mean) / latents_std
            f_lat = (f_lat - latents_mean) / latents_std

        with torch.no_grad(), attn_context:
            output = pipe(
                source_latents=s_lat,
                first_latents=f_lat,
                neg_first_latents=torch.zeros_like(f_lat),
                prompt_embeds=p_embeds,
                negative_prompt_embeds=n_embeds,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.Generator(device=device).manual_seed(seed),
                output_type="latent"
            )
            latents = output.frames

        with torch.no_grad():
            latents = latents.to(m["vae"].dtype)
            latents = latents * latents_std + latents_mean

            video = m["vae"].decode(latents, return_dict=False)[0]
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.permute(0, 2, 3, 4, 1).float()
            out = video

        if not fast_mode:
            torch.cuda.empty_cache()

        if out.ndim == 5:
            out = out.squeeze(0)
        if out.shape[0] == 3:
            out = out.permute(1, 2, 3, 0)
        elif out.shape[1] == 3:
            out = out.permute(0, 2, 3, 1)

        if out.shape[0] > F_in:
            out = out[:F_in]

        return (torch.clamp(out, 0.0, 1.0).cpu().float(),)


class TeleStyleImageModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dtype": (["bf16", "fp16"], {"default": "fp16"}),
                "attention_mode": (["sdpa", "flash_attn", "sage_attn"], {"default": "sdpa"}),
                "auto_benchmark_attention": ("BOOLEAN", {"default": False}),
                "cache_policy": (["reuse_cached", "force_reload"], {"default": "reuse_cached"}),
                "vram_cleanup_before_load": (["none", "soft_empty_cache", "unload_all_models"], {"default": "none"}),
                "compile_dit": ("BOOLEAN", {"default": False}),
                "enable_tf32": ("BOOLEAN", {"default": True}),
                "enable_vram_management": ("BOOLEAN", {"default": False}),
                "vram_limit_gb": ("INT", {"default": 0, "min": 0, "max": 96, "step": 1}),
                "clear_result_cache": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TELE_STYLE_IMAGE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "TeleStyle"

    def load_model(
        self,
        dtype,
        attention_mode,
        auto_benchmark_attention,
        cache_policy,
        vram_cleanup_before_load,
        compile_dit,
        enable_tf32,
        enable_vram_management,
        vram_limit_gb,
        clear_result_cache,
    ):
        try:
            from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        except Exception as e:
            raise RuntimeError(
                "TeleStyle official image nodes require DiffSynth. Install with: "
                "pip install git+https://github.com/modelscope/DiffSynth-Studio.git@11315d7 "
                "transformers==4.57.3 accelerate==1.2.1"
            ) from e

        attention_mode = _normalize_attention_mode(attention_mode)
        source_tag = "huggingface:Qwen/Qwen-Image-Edit-2509"

        cache_key = (
            f"qwen_image_{dtype}_{attention_mode}_ab{int(bool(auto_benchmark_attention))}_"
            f"{int(compile_dit)}_{int(enable_tf32)}_"
            f"{source_tag}_{int(enable_vram_management)}_{int(vram_limit_gb)}"
        )
        if bool(clear_result_cache):
            _IMAGE_RESULT_CACHE.clear()

        if cache_policy == "force_reload":
            print("TeleStyle Info: Forcing reload and clearing runtime cache.")
            _clear_image_runtime_cache()

        cleanup_mode = str(vram_cleanup_before_load or "none")
        if cleanup_mode != "none":
            print(f"TeleStyle Info: Running VRAM cleanup mode: {cleanup_mode}.")
            _clear_image_runtime_cache()
            _comfy_vram_cleanup(cleanup_mode)

        if cache_policy == "reuse_cached" and cache_key in _IMAGE_PIPE_CACHE:
            print("TeleStyle Info: Reusing cached Qwen image pipeline.")
            cached_entry = _IMAGE_PIPE_CACHE[cache_key]
            if isinstance(cached_entry, dict):
                cached_pipe = cached_entry.get("pipe")
                cached_attention_mode = _normalize_attention_mode(cached_entry.get("attention_mode", attention_mode))
            else:
                cached_pipe = cached_entry
                cached_attention_mode = attention_mode
            return ({
                "pipe": cached_pipe,
                "dtype": dtype,
                "attention_mode": cached_attention_mode,
                "auto_benchmark_attention": bool(auto_benchmark_attention),
                "compile_dit": bool(compile_dit),
                "enable_tf32": bool(enable_tf32),
                "qwen_source": "huggingface",
                "enable_vram_management": bool(enable_vram_management),
                "vram_limit_gb": int(vram_limit_gb),
            },)
        if cache_policy == "reuse_cached" and _IMAGE_PIPE_CACHE:
            print("TeleStyle Info: Releasing previous Qwen image pipeline before loading a new config.")
            _clear_image_runtime_cache()

        device = "cuda"
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        base_path = os.path.join(folder_paths.models_dir, "telestyle_models")

        if torch.cuda.is_available() and enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        lora_files = [
            "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
            "weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        ]
        local_lora_paths = []
        for filename in lora_files:
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                hf_hub_download(
                    repo_id="Tele-AI/TeleStyle",
                    filename=filename,
                    local_dir=base_path,
                    local_dir_use_symlinks=False,
                )
            local_lora_paths.append(full_path)

        try:
            with _SuppressKnownDiffSynthLogs():
                model_cfg_extra = {"offload_device": "cpu"} if enable_vram_management else {}
                model_configs = []
                for pattern in (
                    "transformer/diffusion_pytorch_model*.safetensors",
                    "text_encoder/model*.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                ):
                    cfg_kwargs = dict(
                        model_id="Qwen/Qwen-Image-Edit-2509",
                        download_source="huggingface",
                        origin_file_pattern=pattern,
                    )
                    if model_cfg_extra:
                        try:
                            model_configs.append(ModelConfig(**cfg_kwargs, **model_cfg_extra))
                            continue
                        except TypeError:
                            pass
                    model_configs.append(ModelConfig(**cfg_kwargs))

                processor_config = ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2509",
                    download_source="huggingface",
                    origin_file_pattern="processor/",
                )

                from_pretrained_kwargs = dict(
                    torch_dtype=torch_dtype,
                    device=device,
                    model_configs=model_configs,
                    tokenizer_config=None,
                    processor_config=processor_config,
                )
                if enable_vram_management and int(vram_limit_gb) > 0:
                    from_pretrained_kwargs["vram_limit"] = int(vram_limit_gb)

                try:
                    pipe = QwenImagePipeline.from_pretrained(**from_pretrained_kwargs)
                except TypeError as e:
                    if "vram_limit" in str(e):
                        from_pretrained_kwargs.pop("vram_limit", None)
                        pipe = QwenImagePipeline.from_pretrained(**from_pretrained_kwargs)
                    else:
                        raise

                pipe.load_lora(pipe.dit, local_lora_paths[0])
                pipe.load_lora(pipe.dit, local_lora_paths[1])

                if enable_vram_management and hasattr(pipe, "enable_vram_management"):
                    try:
                        pipe.enable_vram_management()
                        print("TeleStyle Info: DiffSynth VRAM management enabled.")
                    except Exception as e:
                        print(f"TeleStyle Warning: enable_vram_management failed: {e}")

                if bool(auto_benchmark_attention):
                    if attention_mode == "sage_attn":
                        print("TeleStyle Info: Attention benchmark skipped for `sage_attn` (uses global ComfyUI backend).")
                    elif enable_vram_management:
                        print("TeleStyle Info: Attention benchmark skipped because VRAM management is enabled.")
                    else:
                        benchmark_mode, benchmark_timings = _benchmark_qwen_attention(pipe, torch_dtype)
                        if benchmark_timings:
                            timing_ms = ", ".join(
                                f"{k}={v * 1000:.1f}ms" for k, v in sorted(benchmark_timings.items(), key=lambda x: x[1])
                            )
                            print(f"TeleStyle Info: Attention benchmark timings: {timing_ms}")
                        if benchmark_mode != attention_mode:
                            print(f"TeleStyle Info: Auto benchmark selected `{benchmark_mode}` over `{attention_mode}`.")
                        attention_mode = benchmark_mode

                if compile_dit and hasattr(torch, "compile"):
                    try:
                        pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead", fullgraph=False)
                        print("TeleStyle Info: torch.compile enabled for Qwen DiT.")
                    except Exception as e:
                        print(f"TeleStyle Warning: torch.compile failed, continuing without compile: {e}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _clear_image_runtime_cache()
                raise RuntimeError(
                    "TeleStyle Image Model Loader ran out of VRAM while loading Qwen. "
                    "Use one loader only, keep dtype=fp16, turn compile_dit off, and restart ComfyUI to clear old allocations."
                ) from e
            raise

        _IMAGE_PIPE_CACHE[cache_key] = {
            "pipe": pipe,
            "attention_mode": attention_mode,
        }
        return ({
            "pipe": pipe,
            "dtype": dtype,
            "attention_mode": attention_mode,
            "auto_benchmark_attention": bool(auto_benchmark_attention),
            "compile_dit": bool(compile_dit),
            "enable_tf32": bool(enable_tf32),
            "qwen_source": "huggingface",
            "enable_vram_management": bool(enable_vram_management),
            "vram_limit_gb": int(vram_limit_gb),
        },)


class TeleStyleOfficialImageInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TELE_STYLE_IMAGE_MODEL",),
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.", "multiline": True}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 123}),
                "min_edge": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TeleStyle"

    def process(
        self,
        model,
        content_image,
        style_image,
        prompt,
        steps,
        seed,
        min_edge,
    ):
        pipe = model["pipe"]
        content = _tensor_to_pil(content_image)
        style = _tensor_to_pil(style_image)

        steps = int(steps)
        min_edge = int(min_edge) - int(min_edge) % 16

        w, h = content.size
        if w > h:
            ratio = w / h
            out_h = min_edge
            out_w = int(out_h * ratio)
            out_w = out_w - out_w % 16
        else:
            ratio = h / w
            out_w = min_edge
            out_h = int(out_w * ratio)
            out_h = out_h - out_h % 16

        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS

        images = [
            content.resize((out_w, out_h), resample),
            style.resize((min_edge, min_edge), resample),
        ]

        attention_mode = _normalize_attention_mode(model.get("attention_mode", "sdpa"))

        cache_key = (
            model.get("dtype", "bf16"),
            attention_mode,
            int(steps),
            int(seed),
            int(min_edge),
            int(out_w),
            int(out_h),
            str(prompt),
            _pil_hash(images[0]),
            _pil_hash(images[1]),
        )
        if cache_key in _IMAGE_RESULT_CACHE:
            cached = _IMAGE_RESULT_CACHE.pop(cache_key)
            if not _is_invalid_black_tensor(cached):
                _IMAGE_RESULT_CACHE[cache_key] = cached
                return (cached.clone(),)
            print("TeleStyle Warning: Ignoring cached invalid/black image result and regenerating.")

        amp_dtype = torch.float16
        if model.get("dtype", "bf16") == "bf16":
            amp_dtype = torch.bfloat16

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if torch.cuda.is_available()
            else nullcontext()
        )
        def _run_once(current_mode):
            attn_context = _get_sdp_context(current_mode)
            with torch.inference_mode(), attn_context, autocast_context:
                out = pipe(
                    prompt,
                    edit_image=images,
                    seed=int(seed),
                    num_inference_steps=steps,
                    height=out_h,
                    width=out_w,
                    edit_image_auto_resize=False,
                    cfg_scale=1.0,
                )
            if isinstance(out, list):
                out = out[0]
            return _pil_to_tensor(out)

        try:
            output_tensor = _run_once(attention_mode)
        except RuntimeError as e:
            err = str(e)
            allocator_assert = (
                "CUDAMallocAsyncAllocator" in err
                or "free_upper_bound + pytorch_used_bytes" in err
                or "INTERNAL ASSERT FAILED" in err
            )
            if not allocator_assert:
                raise

            print(
                "TeleStyle Warning: CUDA allocator internal assert detected. "
                "Running recovery: soft VRAM cleanup, TF32 off for retry, attention=sdpa."
            )
            _comfy_vram_cleanup("soft_empty_cache")
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass

            retry_mode = "sdpa"
            try:
                output_tensor = _run_once(retry_mode)
            except RuntimeError as retry_e:
                raise RuntimeError(
                    "TeleStyle failed after CUDA allocator recovery retry. "
                    "Try: attention_mode=sdpa, cache_policy=force_reload (once), "
                    "vram_cleanup_before_load=unload_all_models (once), or start ComfyUI with --disable-cuda-malloc."
                ) from retry_e

        if _is_invalid_black_tensor(output_tensor) and attention_mode != "sdpa":
            print(
                f"TeleStyle Warning: {attention_mode} produced invalid/black output. "
                "Retrying once with sdpa."
            )
            output_tensor = _run_once("sdpa")

        if not _is_invalid_black_tensor(output_tensor):
            _IMAGE_RESULT_CACHE[cache_key] = output_tensor
        while len(_IMAGE_RESULT_CACHE) > _IMAGE_RESULT_CACHE_MAX:
            _IMAGE_RESULT_CACHE.popitem(last=False)

        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "TeleStyleLoader": TeleStyleLoader,
    "TeleStyleVideoInference": TeleStyleVideoInference,
    "TeleStyleImageModelLoader": TeleStyleImageModelLoader,
    "TeleStyleOfficialImageInference": TeleStyleOfficialImageInference,
}
