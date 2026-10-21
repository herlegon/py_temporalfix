from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
import os
from pathlib import Path
from pprint import pformat, pprint
import signal
import subprocess
import sys
from typing import Literal

from arg_parse import arg_parse
from hytils import lightcyan, lightgreen, red, yellow

__file_dir__ = Path(__file__).resolve().parent
sys.path.insert(0, str(__file_dir__))
import denc
from denc import (
    MediaStream,
    dlogger,
    DecoderThread,
    EncoderThread,
    ProgressThread,
    Device,
    VideoStream,
    PixFmt,
    PIXEL_FORMATS,
    PipeFormat,
    generate_encoder_command,
)
import torch
from vs.vsscript import extract_info_from_vs_script




def main():
    dlogger.setLevel(logging.INFO)

    # Verify the installation
    vspipe_exe: Path = __file_dir__ / "external" / "vspython" / "VSPipe.exe"
    if sys.platform == "linux":
        vspipe_exe = "vspipe"

    if not vspipe_exe.exists(follow_symlinks=True):
        sys.exit(red(f"""
Error: missing tools: vspipe_exe.
Please install these dependencies (refer to the documentation).
        """))

    # Parse arguments
    args: Namespace = arg_parse()
    debug: bool = args.debug
    if debug:
        dlogger.setLevel(logging.DEBUG)

    dlogger.debug(f"Python executable: {sys.executable}")

    # Input file
    in_fp: Path = Path(args.input).resolve()
    in_media: MediaStream = denc.open(in_fp)
    dlogger.info(f"{lightcyan(f"Input file:")}  {in_fp}")
    dlogger.debug(in_media.video)

    # Arguments
    t_radius: float = args.t_radius
    strength: float = args.strength

    # Output file
    #   priority: `output` -> `out_dir` -> none
    output: str = args.output
    out_dir: str = args.out_dir
    append_params: bool = args.append_params
    if output:
        output_fp: Path = Path(output).resolve()
    elif out_dir:
        output_fp: Path = Path(out_dir).resolve() / in_fp.name
    else:
        output_fp: Path = in_fp.parent / f"{in_fp.stem}_radius_{t_radius}_strength_{strength}{in_fp.suffix}"
        append_params = False

    if append_params:
        output_fp: Path = output_fp.parent / f"{output_fp.stem}_radius_{t_radius}_strength_{strength}{output_fp.suffix}"

    if output_fp == in_fp:
        dlogger.error(red(f"Error: Output file must not be the input file"))
        sys.exit()
    dlogger.info(f"{lightcyan(f"Output file:")} {output_fp}")

    out_media = in_media.to_output(filepath=output_fp)
    out_media.video.metadata.update({
        'py_temporalfix': f"tr={t_radius}, strength={strength}"
    })

    # Output of the script pixel format:
    vs_out_pix_fmt: PixFmt = PixFmt.YUV420P
    if PIXEL_FORMATS[out_media.video.pix_fmt.value]['pipe_bpc'] > 8:
        vs_out_pix_fmt = PixFmt.YUV444P16
    dlogger.debug(lightcyan(f"vs_out_pix_fmt: ") + vs_out_pix_fmt.value)
    out_media.video.pipe_format = PipeFormat(
        dtype=None,
        pix_fmt=vs_out_pix_fmt.value,
        shape=out_media.video.pipe_format.shape
    )

    dlogger.debug(out_media.video)
    pprint(out_media.video.pipe_format)

    # VSpipe command
    vs_command: list[str] = [
        str(vspipe_exe),
        str(__file_dir__ / "vs" / "vstf.vpy"),
        "--arg", f"input_fp=\"{str(in_fp)}\"",
        "--arg", f"tr={t_radius}",
        "--arg", f"strength={strength}",
        "--arg", f"pix_fmt={vs_out_pix_fmt.value}",
        "-",
    ]
    dlogger.debug(lightcyan("VS command:"))
    dlogger.debug(' '.join(vs_command))

    # Clean environment for vspython
    # vs_path: list[str] = []
    forbidden_names: tuple[str] = (
        'python',
        'conda',
        'vapoursynth',
        'ffmpeg',
    )

    # Create path used by vs subprocess
    vs_path: list[str] = []
    sep: str = ";"
    if sys.platform == "win32":
        for dir in ("Scripts", "vs-scripts", "vs-plugins", ""):
            vs_path.insert(0, str(__file_dir__ / "external" / "vspython" / dir))

    elif sys.platform == "linux":
        for dir in (
            "/usr/lib/x86_64-linux-gnu/vapoursynth",
            "/usr/lib/bin"
        ):
            vs_path.insert(0, dir)
        sep = ':'

    vs_path.insert(0, str(__file_dir__))
    # Clean environnment for vs
    vs_env = os.environ.copy()
    if sys.platform == 'win32':
        del vs_env['PATH']
        for k, v in vs_env.copy().items():
            k_lower, v_lower = k.lower(), v.lower()
            for n in forbidden_names:
                if n in k_lower:
                    try:
                        del vs_env[k]
                        dlogger.debug(f"removing: {k}: {v}")
                    except:
                        pass

                if n in v_lower:
                    try:
                        del vs_env[k]
                        dlogger.debug(f"removing: {k}: {v}")
                    except:
                        pass
        vs_env['PATH'] = sep.join(vs_path)

    # dlogger.debug(f"Environment:\n{pformat(vs_env)}")
    # Environnment
    # if debug:
    #     extract_info_from_vs_script(vs_command=vs_command, vs_env=vs_env)

    encoder_command: list[str] = generate_encoder_command(media=out_media)
    parent_dir: str = out_media.video.filepath.parent
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Encoder process
    encoder_subprocess: subprocess.Popen | None = None
    try:
        encoder_subprocess = subprocess.Popen(
            encoder_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        dlogger.critical(red(f"[E] Unexpected error: {type(e)}"))
        sys.exit()
    if encoder_subprocess is None:
        dlogger.critical(red(f"[E] Encoder process is not started"))
        sys.exit()

    # Vs process
    vs_subprocess: subprocess.Popen | None = None
    try:
        vs_subprocess = subprocess.Popen(
            vs_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=vs_env,
        )
    except Exception as e:
        dlogger.critical(red(f"[E] Unexpected error: {type(e)}"))
        sys.exit()

    dlogger.debug(lightcyan("Encoder command:"))
    dlogger.debug(lightgreen(' '.join(encoder_command)))


    frame_count: int = in_media.video.frame_count
    nbytes: int = out_media.video.pipe_format.nbytes
    frame: bytes = None
    line: str = ''
    os.set_blocking(encoder_subprocess.stdout.fileno(), False)
    os.set_blocking(vs_subprocess.stderr.fileno(), False)
    dlogger.info(f"Processing:")
    try:
        for _ in range(frame_count):
            # print(f"reading frame no. {i}", end="\r")
            frame: bytes = vs_subprocess.stdout.read(nbytes)
            if frame is None:
                dlogger.critical(red("Frame is none"))
            encoder_subprocess.stdin.write(frame)

            line = encoder_subprocess.stdout.readline().decode('utf-8')
            if line:
                print(line.strip(), end='\r', file=sys.stderr)

            line = vs_subprocess.stderr.readline().decode('utf-8')
            if line:
                print(line.strip(), end='\r', file=sys.stderr)
        print()
    except:
        pass

    stdout_b: bytes | None = None
    stderr_b: bytes | None = None
    try:
        # Arbitrary timeout value
        stdout_b, stderr_b = encoder_subprocess.communicate(timeout=10)
    except:
        encoder_subprocess.kill()
        return

    if stdout_b is not None:
        stdout = stdout_b.decode('utf-8)')
        if stdout:
            dlogger.debug(f"FFmpeg stdout:\n{stdout}")

    if stderr_b is not None:
        stderr = stderr_b.decode('utf-8)')
        if stderr:
            dlogger.debug(f"FFmpeg stderr:\n{stderr}")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


