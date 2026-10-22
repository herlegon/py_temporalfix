from pathlib import Path
import sys
from typing import Literal
__file_dir__ = Path(__file__).resolve().parent
root = __file_dir__.parent
for subd in ("Scripts", "vs-scripts", "vs-plugins", ""):
    sys.path.insert(0, str(__file_dir__.parent / "external" / "vspython" / subd))
sys.path.insert(0, str(__file_dir__))
sys.path.insert(0, str(root))
from multiprocessing import cpu_count
if sys.platform == "win32":
    from vs_temporalfix import vs_temporalfix

import vapoursynth as vs
core = vs.core
core.num_threads = int(cpu_count() - 2)
core.max_cache_size = 20000

input_fp: str
clip = core.bs.VideoSource(source=input_fp.replace("\"", ""), cachemode = 0)
# clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_BT709)
# clip = clip.std.SetFrameProps(_Primaries=vs.MATRIX_BT709)
# clip = clip.std.SetFrameProps(_ChromaLocation=vs.MATRIX_BT709)
# clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_FULL)

strength: int
tr: int
if clip.format != vs.YUV444P16:
    clip = core.resize.Lanczos(
        clip, format=vs.YUV444P16, matrix_in_s="709"
    )

if sys.platform == "win32":
    clip = vs_temporalfix(
        clip,
        strength=int(strength),
        tr=int(tr),
        debug=False
    )

out_pix_fmt: Literal['yuv420p', 'yuv444p16le']
if out_pix_fmt == 'yuv444p16le' and clip.format != vs.YUV444P16:
    clip = core.resize.Bicubic(clip, format=vs.YUV444P16, matrix_in_s="709")

elif out_pix_fmt == 'yuv420p' and clip.format != vs.YUV420P8:
    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_in_s="709")

clip.set_output()
