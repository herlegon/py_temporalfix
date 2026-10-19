from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from pathlib import Path
import signal
import sys

from arg_parse import arg_parse
from hytils import lightcyan, red

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denc
from denc import (
    MediaStream,
    dlogger,
    DecoderThread,
    EncoderThread,
    ProgressThread,
    Device,
)




def main():
    dlogger.setLevel(logging.INFO)

    # Verify the installation
    root_dir: Path = Path(__file__).resolve()
    vspipe_exe: Path = root_dir / "external" / "vspython" / "VSPipe.exe"
    if sys.platform == "linux":
        vspipe_exe = "vspipe"

    if vspipe_exe.exists(follow_symlinks=True):
        sys.exit(red(f"""
Error: missing tools: vspipe_exe.
Please install these dependencies (refer to the documentation).
        """))

    # Parse arguments
    args: Namespace = arg_parse()
    debug: bool = args.debug
    if debug:
        dlogger.setLevel(logging.DEBUG)

    # Input file
    in_fp: Path = Path(args.input).resolve()
    in_media: MediaStream = denc.open(in_fp)
    dlogger.info(f"{lightcyan(f"Input file:")}  {in_fp}")
    dlogger.debug(in_media.video)

    # Arguments
    radius: float = args.radius
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
        output_fp: Path = in_fp.parent / f"{in_fp.stem}_radius_{radius}_strength_{strength}{in_fp.suffix}"
        append_params = False

    if append_params:
        output_fp: Path = output_fp.parent / f"{output_fp.stem}_radius_{radius}_strength_{strength}{output_fp.suffix}"

    if output_fp == in_fp:
        dlogger.error(red(f"Error: Output file must not be the input file"))
        sys.exit()
    dlogger.info(f"{lightcyan(f"Output file:")} {output_fp}")

    out_media = in_media.to_output(filepath=output_fp)
    dlogger.debug(out_media.video)


    dlogger.debug(f"Python executable dir: {sys.executable}")

    # VSpipe command
    vs_command: list[str] = [
        vspipe_exe,
        str(root_dir / "vs" / "vstf.vpy"),
        "--arg", f"input_fp=\"{in_fp}\"",
        "--arg", f"tr={radius}",
        "--arg", f"strength={strength}",
        "--arg", f"pix_fmt={out_media.video.pix_fmt}",
        "-",
    ]
    dlogger.debug(lightcyan("VS command:"))
    dlogger.debug(' '.join(vs_command))



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


