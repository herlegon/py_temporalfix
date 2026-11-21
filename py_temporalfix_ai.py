from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
import logging
import math
from pathlib import Path
import re
import sys
import signal
from warnings import warn

from hytils import blue, green, lightcyan, lightgreen, purple, red, yellow
try:
    import winsound
except ImportError:
    pass
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import denc
from denc import (
    MediaStream,
    VideoStreamFormat,
    dlogger,
    DecoderThread,
    EncoderThread,
    ProgressThread,
    Device,
    add_io_arguments,
    add_debug_arguments,
    add_sar_arguments,
    add_resize_arguments,
    add_inference_arguments,
    generate_output_filepath,
    BoundedFloat,
)
from nnarchs import (
    nnlib,
    NnModel,
    Hdtype,
    HdtypeToTorch,
    NnFrameworkType,
    PyTorchModel,
    generate_out_model_fp,
    ShapeStrategy,
)
from hsys import (
    is_feature_supported,
    VramMonitor,
)
from hinstall import download_models

from inference.cuda_temporalfix_thread import CudaTemporalInferenceThread
from inference.trt_temporalfix_thread import TrtTemporalInferenceThread
from inference.run_threads import run_threads



def main():
    dlogger.setLevel(logging.INFO)

    parser = ArgumentParser()
    add_io_arguments(parser)
    parser.add_argument(
        "-s",
        "--strength",
        type=BoundedFloat(0, 3),
        metavar="[0..3]",
        default=2,
        required=False,
        help="""Suppression strength of temporal inconsistencies in the range `0.0-3.0`. Higher means more aggressive.
Higher resolution tends to need higher strength. Too high may oversmooth small movements. Default: 2.
\n"""
    )
    parser.add_argument(
        "-mdir",
        "--model_dir",
        type=str,
        default="",
        required=False,
        help="""Model directory
\n"""
    )
    add_inference_arguments(parser)
    add_debug_arguments(parser)
    args = parser.parse_args()
    debug: bool = args.debug
    if debug:
        dlogger.setLevel(logging.DEBUG)

    if not is_feature_supported(feature='cuda'):
        raise SystemError(red(f"Missing CUDA GPU"))

    # Input file
    #-------------------------------------------------------------------------------------
    in_fp: Path = Path(args.input).resolve()
    in_media: MediaStream = denc.open(filepath=in_fp)
    dlogger.info(f"{lightcyan(f"Input file:")}  {in_fp}")
    dlogger.debug(in_media.video)

    # Output file
    #-------------------------------------------------------------------------------------
    suffix = f"_strength_{args.strength}"
    output_fp = generate_output_filepath(args, suffix=suffix)

    # Output media
    #-------------------------------------------------------------------------------------
    out_media = in_media.to_output(filepath=output_fp)
    dlogger.debug(out_media.video)

    # Models
    #-------------------------------------------------------------------------------------
    if args.model_dir:
        model_dir: Path = Path(args.model_dir).resolve()
    else:
        # Specific for me
        model_dir = Path(__file__).resolve().parents[1] / "ml_models" / "temporalfix"
        if not model_dir.is_dir():
            model_dir: Path = Path(__file__).resolve().parent / "external" / "models"
        model_dir: Path = Path(__file__).resolve().parent / "external" / "models"

    temporal_models: dict[int, str] = {
        0: "temporalfix_s0_v1.pth",
        1: "temporalfix_s1_v1.1.pth",
        2: "temporalfix_s2_v1.pth",
        3: "temporalfix_s3_v1.pth",
    }

    # Download model if missing, no retry in case of github issue
    missing = [m for m in temporal_models.values() if not (model_dir / m).is_file()]
    if missing:
        results = download_models(
            names=missing,
            dest_dir=model_dir,
            base_url="https://github.com/pifroggi/vs_temporalfix/raw/main/vs_temporalfix/models",
            force=False
        )
        for name, ok in results.items():
            if not ok:
                dlogger.error(red(f"Missing model: {name}"))
                sys.exit()

    strength = round(float(args.strength), 2)
    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    weighting = strength - strength_lower


    # Inference device
    device: str = "cuda:0"

    # Open and convert model
    torch_model_path: Path = model_dir / temporal_models[strength_lower]
    torch_model: PyTorchModel = nnlib.open(torch_model_path)
    tensorrt_model_fp: Path = None
    is_inference_trt: bool = args.trt
    dlogger.info(lightcyan(f"Framework: ") + f"{'TensorRT' if is_inference_trt else 'CUDA'}")
    if is_inference_trt:
        # Try to load trt engine if already exists
        shape_strategy: ShapeStrategy = ShapeStrategy(
            type='static',
            opt_size=list(reversed(in_media.video.shape[:2]))
        )
        trt_dtype = 'fp16' if args.fp16 else 'fp32'
        opset: int = 20
        tensorrt_model_fp: Path = generate_out_model_fp(
            model=torch_model,
            to=NnFrameworkType.TENSORRT,
            opset=opset,
            dtype=trt_dtype,
            device=device,
            out_dir=model_dir,
            shape_strategy=shape_strategy,
            suffix=f"{args.strength}",
        )

    # Generate the torch model used for the inference
    if (
        (is_inference_trt and not tensorrt_model_fp.exists())
        or not is_inference_trt
    ):
        if strength_lower != strength_upper:
            lower_model: PyTorchModel = torch_model
            upper_model: PyTorchModel = nnlib.open(model_path=model_dir / temporal_models[strength_upper])

            # lower_model.state_dict
            # upper_model.state_dict
            consolidated_state_dict = OrderedDict(
                (
                    key, torch.lerp(value, upper_model.state_dict[key], weighting)
                    if torch.is_floating_point(value) else value
                )
                for key, value in lower_model.state_dict.items()
            )

            # Let's patch the model state_dict
            torch_model.state_dict = consolidated_state_dict
            # No need to keep the uppdermodel in memory
            del upper_model

    if is_inference_trt:
        dlogger.info(lightcyan("TensortRT engine filepath: ") + f"{tensorrt_model_fp}")
        if not tensorrt_model_fp.exists():
            dlogger.info(lightcyan(f"\nConvert model to a tensorRT engine:"))
            dlogger.info(blue("    Device: ") + f"{device}")
            dlogger.info(blue("    Resolution: ") + f"{shape_strategy.opt_size}")
            dlogger.info(blue("    Datatype: ") + f"{trt_dtype}")
            model = nnlib.convert_to_tensorrt(
                model=torch_model,
                shape_strategy=shape_strategy,
                dtype=trt_dtype,
                device=device,
                opset=opset,
                out_dir=model_dir,
                suffix=f"{args.strength}",
            )
        else:
            model = nnlib.open(model_path=tensorrt_model_fp)
    else:
        model = torch_model

    dlogger.info(lightcyan(f"\nModel:"))
    dlogger.debug(model)
    dlogger.info(blue(f"  framework: ") + model.framework.type.value)
    dlogger.info(blue(f"  arch: ") + f"{model.arch_name}")
    dlogger.info(blue(f"  scale: ") + f"{model.scale}")

    # use cuda device 0 by default
    in_media.video.device = Device(name=device)
    out_media.video.device = Device(name=device)

    if is_inference_trt:
        infer_dtype: Hdtype = trt_dtype
    else:
        infer_dtype: Hdtype = 'fp32'
        if args.fp32:
            infer_dtype = 'fp32'
        elif args.fp16:
            infer_dtype = 'fp16'
        elif args.bf16:
            infer_dtype = 'bf16'

    dlogger.info(blue(f"  device: ") + f"{device}")
    dlogger.info(blue(f"  datatype: ") + f"{infer_dtype}")

    # Resource monitor: device
    device_arg_list = device.split(':')
    cuda_device_no = int(device_arg_list[-1] if len(device_arg_list) > 1 else 0)
    vram_monitor = VramMonitor(device=cuda_device_no)
    vram_monitor.start()

    # Chain
    #-------------------------------------------------------------------------------------
    in_sf: VideoStreamFormat = in_media.video.stream_format()
    dlogger.debug(green(f"Decoder stream format: ") + f"{in_sf}")
    decoder_thread = DecoderThread(media_stream=in_media, verbose=debug)

    out_media: MediaStream = in_media.to_output(filepath=output_fp)
    out_sf = deepcopy(in_sf)
    out_media.video.input_stream_format = out_sf
    dlogger.debug(purple(f"Encoder input stream format: ") + f"{out_sf}")
    dlogger.debug(f"input->FFmpeg pipe: {out_media.video.pipe_format}")
    encoder_thread = EncoderThread(media_stream=out_media, verbose=debug)


    infer_thread: TrtTemporalInferenceThread |  CudaTemporalInferenceThread
    if is_inference_trt:
        infer_thread = TrtTemporalInferenceThread(name=model.arch.name, debug=debug)
    else:
        infer_thread = CudaTemporalInferenceThread(name=model.arch.name, debug=debug)

    infer_thread.initialize(
            model=model,
            device=device,
            dtype=infer_dtype,
            prescale=None
        )


    decoder_thread.set_consumer(infer_thread)
    infer_thread.set_producer(decoder_thread)
    infer_thread.set_consumer(encoder_thread)
    encoder_thread.set_producer(decoder_thread)

    # Progress bar
    #-------------------------------------------------------------------------
    total_frames: int = in_media.video.frame_count
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    encoder_thread.set_progress_thread(progress_thread)


    # Main loop
    #-------------------------------------------------------------------------
    stats = {}

    total_frames: int = out_media.video.frame_count
    elapsed = run_threads(
        dec_thread=decoder_thread,
        enc_thread=encoder_thread,
        infer_threads=[infer_thread,],
        progress_thread=progress_thread,
        total_frames=total_frames,
        vram_stats=stats,
        vram_monitor=vram_monitor,
        debug=debug
    )

    print(f"elapsed: {elapsed:.02f}s ({total_frames/elapsed:.02f}fps)")
    print(f"VRAM mean: {stats['mean_mb']:.0f} MB | peak: {stats['peak_mb']:.0f} MB")

    if not args.cpu:
        torch.cuda.empty_cache()

    if sys.platform == "win32":
        winsound.Beep(frequency=440, duration=200)

    dlogger.info(f"Done")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


