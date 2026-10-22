from argparse import Namespace
import logging
import os
from pathlib import Path
from pprint import pprint
import signal
import subprocess
import sys
import time


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
from arg_parse import arg_parse
from hytils import lightcyan, lightgreen, red, yellow

from vs.vsscript import extract_info_from_vs_script




def main():
    dlogger.setLevel(logging.INFO)

    # Verify the installation
    vspipe_exe: Path = __file_dir__ / "external" / "vspython" / "VSPipe.exe"
    if sys.platform == "linux":
        vs_env_dir: Path = Path(os.environ.get("VS_ENV_DIR", Path.home() / "vs-env")).expanduser()
        vspipe_exe = vs_env_dir / "bin" / "vspipe"

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
        "--arg", f"out_pix_fmt={vs_out_pix_fmt.value}",
        "-",
    ]
    dlogger.debug(lightcyan("VS command:"))
    dlogger.debug(' '.join(vs_command))

    # Clean environment for vspython
    # vs_path: list[str] = []
    forbidden_vars: tuple[str] = (
        'python',
        'conda',
        'vapoursynth',
        'ffmpeg',
    )

    protected_vars: tuple[str] = []
    if sys.platform == 'linux':
        protected_vars = (
            "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "SHELL", "XDG_RUNTIME_DIR", "DISPLAY",
        )

    # Clean environnment for vs
    vs_env = os.environ.copy()
    if sys.platform == "win32":
        del vs_env["PATH"]

    for k, v in vs_env.copy().items():
        if k in protected_vars:
            continue

        k_lower, v_lower = k.lower(), v.lower()
        for n in forbidden_vars:
            if n in k_lower:
                try:
                    del vs_env[k]
                    # dlogger.debug(f"removing: {k}: {v}")
                except:
                    pass

            if n in v_lower:
                try:
                    del vs_env[k]
                    # dlogger.debug(f"removing: {k}: {v}")
                except:
                    pass

    # Create path used by vs subprocess
    vs_path: list[str] = []
    sep: str = ";"
    if sys.platform == "win32":
        for dir in ("Scripts", "vs-scripts", "vs-plugins", ""):
            vs_path.insert(0, str(__file_dir__ / "external" / "vspython" / dir))

    elif sys.platform == "linux":
        venv_bin = vs_env_dir / "bin"
        venv_lib = vs_env_dir / "lib"

        # PATH : venv d'abord, puis le strict minimum système
        for d in ("/usr/local/bin", "/usr/bin", "/bin"):
            vs_path.append(d)
        vs_path.insert(0, str(venv_bin))

        # Le Python embarqué dans vspipe ne lit pas VIRTUAL_ENV :
        # il faut lui donner explicitement les site-packages du venv.
        site_pkgs = [str(p) for p in sorted(venv_lib.glob("python*/site-packages"))]
        if site_pkgs:
            vs_env["PYTHONPATH"] = sep.join(site_pkgs)
        vs_env["VIRTUAL_ENV"] = str(vs_env_dir)

        # Bibliothèques partagées (libvapoursynth-script.so, plugins…)
        ld_paths = [str(venv_lib)] + site_pkgs
        vs_env["LD_LIBRARY_PATH"] = sep.join(ld_paths)

    vs_path.insert(0, str(__file_dir__))
    vs_env['PATH'] = sep.join(vs_path)

    # dlogger.debug(f"Environment:\n{pformat(vs_env)}")

    # if debug:
    #     extract_info_from_vs_script(vs_command=vs_command, vs_env=vs_env)

    encoder_command: list[str] = generate_encoder_command(media=out_media)
    parent_dir: str = out_media.video.filepath.parent
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Vs process
    vs_subprocess: subprocess.Popen = None
    try:
        vs_subprocess = subprocess.Popen(
            vs_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            env=vs_env,
        )
    except Exception as e:
        dlogger.critical(red(f"[E] Unexpected error: {type(e)}"))
        sys.exit()

    # Encoder process
    encoder_subprocess: subprocess.Popen = None
    try:
        # VSPipe -> FFmpeg
        encoder_subprocess = subprocess.Popen(
            encoder_command,
            stdin=vs_subprocess.stdout,
            stdout=subprocess.PIPE,
            stderr= None,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        dlogger.critical(red(f"[E] Unexpected error: {type(e)}"))
        sys.exit()
    if encoder_subprocess is None:
        dlogger.critical(red(f"[E] Encoder process is not started"))
        sys.exit()


    dlogger.debug(lightcyan("Encoder command:"))
    dlogger.debug(lightgreen(' '.join(encoder_command)))

    frame_count: int = in_media.video.frame_count
    nbytes: int = out_media.video.pipe_format.nbytes
    frame: bytes = None
    line: str = ''


    vs_subprocess.stdout.close()
    frame_count = in_media.video.frame_count
    t0 = last_t = time.perf_counter()
    last_n = 0
    try:
        for line in encoder_subprocess.stdout:        # se termine à l'EOF
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdigit():
                done = int(value)
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    inst_fps = (done - last_n) / (now - last_t)
                    last_n, last_t = done, now
                    eta = (frame_count - done) / inst_fps if inst_fps > 0 else 0.0
                    print(f"{done}/{frame_count}  {inst_fps:6.2f} fps  ETA {eta:5.0f}s",
                        end="\r", file=sys.stdout)
        print()
        rc_enc = encoder_subprocess.wait()
        rc_vs = vs_subprocess.wait()
    except BaseException:
        for p in (vs_subprocess, encoder_subprocess):
            if p.poll() is None:
                p.kill()
        raise

    if rc_enc != 0 or rc_vs != 0:
        dlogger.critical(red(f"Échec : ffmpeg={rc_enc}, VSPipe={rc_vs}"))



    # os.set_blocking(encoder_subprocess.stdout.fileno(), False)
    # os.set_blocking(vs_subprocess.stderr.fileno(), False)
    # dlogger.debug(lightgreen(f"nbytes={nbytes}"))
    # dlogger.info(f"Processing:")
    # try:
    #     for i in range(frame_count):
    #         print(f"reading frame no. {i}", end="\r")
    #         frame: bytes = vs_subprocess.stdout.read(nbytes)
    #         if frame is None:
    #             dlogger.critical(red("Frame is none"))
    #         encoder_subprocess.stdin.write(frame)

    #         line = encoder_subprocess.stdout.readline().decode('utf-8')
    #         if line:
    #             print(line.strip(), end='\r', file=sys.stderr)

    #         line = vs_subprocess.stderr.readline().decode('utf-8')
    #         if line:
    #             print(line.strip(), end='\r', file=sys.stderr)
    #     print()
    # except:
    #     pass

    # dlogger.info(f"Wait for the end...")
    # stdout_b: bytes | None = None
    # stderr_b: bytes | None = None
    # try:
    #     # Arbitrary timeout value
    #     stdout_b, stderr_b = encoder_subprocess.communicate(timeout=10)
    # except:
    #     encoder_subprocess.kill()
    #     return

    # if stdout_b is not None:
    #     stdout = stdout_b.decode('utf-8)')
    #     if stdout:
    #         dlogger.debug(f"FFmpeg stdout:\n{stdout}")

    # if stderr_b is not None:
    #     stderr = stderr_b.decode('utf-8)')
    #     if stderr:
    #         dlogger.debug(f"FFmpeg stderr:\n{stderr}")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


