from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
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
    ProgressThread,
    PixFmt,
    PIXEL_FORMATS,
    PipeFormat,
    generate_encoder_command,
    generate_output_filepath,
    BoundedInteger,
    add_io_arguments,
    add_seek_arguments,
    add_debug_arguments,
)
from hytils import lightcyan, lightgreen, red

from vs.vsscript import extract_info_from_vs_script




def main():
    dlogger.setLevel(logging.INFO)

    # Verify the installation
    vspipe_exe: Path = __file_dir__ / "external" / "vspython" / "lib" / "site-packages" / "vapoursynth" / "VSPipe.exe"

    if sys.platform == "linux":
        vs_env_dir: Path = Path(os.environ.get("VS_ENV_DIR", Path.home() / "vs-env")).expanduser()
        vspipe_exe = vs_env_dir / "bin" / "vspipe"

    if not vspipe_exe.exists(follow_symlinks=True):
        sys.exit(red(f"""
Error: missing tools: vspipe_exe.
Please install these dependencies (refer to the documentation).
        """))

    # Parse arguments
    parser = ArgumentParser(
        description="Python wrapper for vs_temporalfix script",
        formatter_class=RawTextHelpFormatter
    )
    add_io_arguments(parser)

    parser.add_argument(
        "-tr",
        "--t_radius",
        type=BoundedInteger(1, 10),
        metavar="[1..10]",
        default=6,
        required=False,
        help="""Temporal radius sets the number of frames to average over. Higher means more stable.
If you get blending/ghosting on small movements or blocky artifacts, reduce this.
\n"""
    )

    parser.add_argument(
        "-s",
        "--strength",
        type=BoundedInteger(1, 800),
        metavar="[1..800]",
        default=400,
        required=False,
        help="""Suppression strength of temporal inconsistencies.
Higher means more aggressive.
\n"""
    )

    add_seek_arguments(parser)
    add_debug_arguments(parser)
    args: Namespace = parser.parse_args()

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
    output_fp = generate_output_filepath(
        args=args,
        suffix=f"_radius_{t_radius}_strength_{strength}"
    )

    # Output media
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

    encoder_command: list[str] = generate_encoder_command(media=out_media, progress=True)
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
            stderr=subprocess.PIPE,
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

    # Run
    vs_subprocess.stdout.close()

    frame_count = in_media.video.frame_count
    progress = ProgressThread(total=frame_count)
    progress.start()
    try:
        encoder_stdout: list[str] = encoder_subprocess.stdout
        last: int = 0
        # key=value from -progress pipe:1
        for line in encoder_stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdigit():
                n = int(value)
                progress.put(n - last)
                last = n

        progress.put(0, force=True)
        progress.put(None)
        encoder_return_code = encoder_subprocess.wait()

        vs_return_code = vs_subprocess.wait()
        vs_errors: str = vs_subprocess.stderr.read().decode("utf-8", "replace")
        if vs_return_code != 0:
            dlogger.critical(red(f"VSPipe failed: ({vs_return_code}):\n{vs_errors}"))

    except BaseException:
        progress.stop(force=True)
        for p in (vs_subprocess, encoder_subprocess):
            if p.poll() is None:
                p.kill()
        raise

    finally:
        progress.join(timeout=10)

    if encoder_return_code != 0 or vs_return_code != 0:
        dlogger.critical(red(f"Error: ffmpeg={encoder_return_code}, VSPipe={vs_return_code}"))




if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


