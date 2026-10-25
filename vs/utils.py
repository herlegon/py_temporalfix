
# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

import re
import math
import vapoursynth as vs

core = vs.core


def temporal_median(clip, radius=1, planes=None):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.TemporalMedian(clip, radius=radius, planes=planes)
    else:
        return core.tmedian.TemporalMedian(clip, radius=radius, planes=planes)


def repair(clip, repairclip, mode=[1]):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Repair(clip, repairclip, mode=mode)
    else:
        return core.rgvs.Repair(clip, repairclip, mode=mode)


def median(clip, radius=1, planes=None):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Median(clip, radius=radius, planes=planes)
    elif radius == 1:
        return core.std.Median(clip, planes=planes)
    else:
        return core.ctmf.CTMF(clip, radius=radius, planes=planes)


def basic_expr(clips, expr, format=None):
    # backend for basic exprs supported by std.Expr
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr, format=format)
    else:
        return core.std.Expr(clips, expr, format=format)


def advanced_expr(clips, expr, format=None):
    # backend for advanced exprs not possible with std.Expr
    return core.akarin.Expr(clips, expr, format=format)


def box_blur(clip, planes=None, hradius=1, hpasses=1, vradius=1, vpasses=1):
    # optional plugin for slight speed boost
    if hasattr(core, "vszip"):
        return core.vszip.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
    else:
        return core.std.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)


def min_blur(clip, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée https://avisynth.nl/index.php/MinBlur

    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    RG4  = median(clip, radius=2, planes=planes)
    expr = "x y - x z - * 0 < x dup y - abs x z - abs < y z ? ?"
    return basic_expr([clip, RG11, RG4], [expr if i in planes else "" for i in range(clip.format.num_planes)])


def average_color_fix(clip, ref, radius=4, passes=4):
    # simplified from https://github.com/pifroggi/vs_colorfix
    blurred_reference = box_blur(ref, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    blurred_clip = box_blur(clip, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    diff_clip = core.std.MakeDiff(blurred_reference, blurred_clip)
    return core.std.MergeDiff(clip, diff_clip)


def average_color_fix_fast(clip, ref, downscale_factor=8):
    # faster but faint blocky artifacts
    width  = int(clip.width  / downscale_factor) >> clip.format.subsampling_w << clip.format.subsampling_w
    height = int(clip.height / downscale_factor) >> clip.format.subsampling_h << clip.format.subsampling_h
    downscaled_reference = core.resize.Bilinear(ref,  width=width, height=height)
    downscaled_clip = core.resize.Bilinear(clip, width=width, height=height)
    diff_clip = core.std.MakeDiff(downscaled_reference, downscaled_clip)
    diff_clip = core.resize.Bilinear(diff_clip, width=clip.width, height=clip.height)
    return core.std.MergeDiff(clip, diff_clip)


def frequency_merge(low, high, radius=40, passes=3):
    # merges low freqs of one clip with high freqs of another clip
    low_remaining  = box_blur(low,  hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    high_removed   = box_blur(high, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    high_remaining = core.std.MakeDiff(high, high_removed)
    return core.std.MergeDiff(low_remaining, high_remaining)


def tweak_darks(src, strength=2.5, amp=0.2):
    # simplified DitherLumaRebuild function that works on full range
    # DitherLumaRebuild function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # originally created by cretindesalpes https://forum.doom9.org/showthread.php?p=1548318
    bd = src.format.bits_per_sample
    scale = 1 << (bd - 8)

    x = "x" if bd == 8 else f"x {scale} /"
    t = f"{x} 255 / 0 max 1 min"
    k = (strength - 1) * amp
    e = f"{k} {1 + amp} {(1 + amp) * amp} {t} {amp} + / - * {t} {1 - k} * + {1 << bd} *"
    expr = [e] + [""] * (src.format.num_planes - 1)
    return basic_expr([src], expr)


def contrasharp(clip, src, rep=24, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée at the VERY GRAINY thread https://forum.doom9.org/showthread.php?p=1076491

    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd   = clip.format.bits_per_sample
    mid  = 1 << (bd - 1)
    num  = clip.format.num_planes

    s    = min_blur(clip, planes)  # damp down remaining spots of the denoised clip
    RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    ssD  = core.std.MakeDiff(s, RG11, planes)  # the difference of a simple kernel blur
    allD = core.std.MakeDiff(src, clip, planes)  # the difference achieved by the denoising
    ssDD = repair(ssD, allD, [rep if i in planes else 0 for i in range(num)])  # limit the difference to the max of what the denoising removed locally
    expr = "x {} - abs y {} - abs < x y ?".format(mid, mid)  # abs(diff) after limiting may not be bigger than before
    ssDD = basic_expr([ssDD, ssD], [expr if i in planes else "" for i in range(num)])
    return core.std.MergeDiff(clip, ssDD, planes)  # apply the limited difference (sharpening is just inverse blurring)



def lowfreq_denoise(low, high, motionmask, thsad=200, tr=6):
    # temporally denoise low frequencies only

    bs  = 8
    pel = 1

    # downscale clips
    downscale_factor = 8
    width  = int(low.width  / downscale_factor) >> low.format.subsampling_w << low.format.subsampling_w
    height = int(low.height / downscale_factor) >> low.format.subsampling_h << low.format.subsampling_h
    low_down   = core.resize.Bicubic(low,      width=width, height=height)
    motionmask = core.resize.Point(motionmask, width=width, height=height)
    motionmask = core.std.Maximum(motionmask)                  # expand mask
    prefilter  = tweak_darks(low_down, strength=2.5, amp=0.2)  # brighten darks

    # create superclips
    pref_sup = core.mvu.Super(prefilter, blksize=bs, overlap=bs // 2, pel=pel, sharp=1, rfilter=2)
    low_sup  = core.mvu.Super(low_down,  blksize=bs, overlap=bs // 2, pel=pel, sharp=0, onelevel=True)

    # analyse and degrain
    low_vecs = core.mvu.AnalyseMany(pref_sup, radius=tr, search=2, searchparam=1, mvlambda=0, lsad=400, plevel=0, pnew=0, pzero=0, globalmv=False)
    low_degr = core.mvu.Degrain(low_down, low_sup, low_vecs, thsad=[thsad, thsad], planes=[0])

    # merge
    low_degr = core.std.MaskedMerge(low_degr, low_down, motionmask)               # reduce blending/ghosting
    low_degr = core.resize.Bicubic(low_degr, width=low.width, height=low.height)  # resize back to original res
    return frequency_merge(low_degr, high, 10, 3)                                 # merge low freqs with original high freqs

