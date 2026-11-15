from __future__ import annotations
from enum import IntEnum
from queue import Queue
import time

import torch
from torch import Tensor
from hytils import blue, blue, red

from denc import (
    BaseThread,
    NnFrame,
    wait_frame_ready,
    mark_frame_ready,
    dlogger,
)
from nnarchs import (
    nnlib,
    Hdtype,
    PyTorchModel,
    PyTorchSession,
)
from gpu_algos import (
    gpu_resize_to_,
    gpu_resize_,
)


class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTYING = 0x03
    EMPTY = 0x04



class TemporalFrameCacheMirroring:

    def __init__(self, window_size: int = 7) -> None:
        self.window_size: int = window_size
        self.radius: int = window_size // 2
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []        # each item: (1,3,H,W), chronological order
        self.feats: list[Tensor] = []          # each item: (1,4,H,W), parallel to self.tensors
        self.tail: list[Tensor] = []           # end-of-clip mirror padding (images)
        self.feat_tail: list[Tensor] = []      # end-of-clip mirror padding (features)
        self.pad_i: int = 0
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.feats.clear()
        self.tail.clear()
        self.feat_tail.clear()
        self.pad_i = 0
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING))


    def is_empty(self) -> bool:
        return bool(self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT))


    def _enter_emptying(self) -> None:
        # tensors holds 7 items, the last frame of the clip is at [-1]
        # mirror padding: f(N-2), f(N-3), f(N-4)
        # Features are shared with the images: a mirrored frame reuses the same
        # feature tensor (the encoder is per-frame, so no re-encoding is needed).
        self.tail = [self.tensors[-(k + 1)] for k in range(1, self.radius + 1)]
        self.feat_tail = [self.feats[-(k + 1)] for k in range(1, self.radius + 1)]
        self.pad_i = 0
        self.state = _CACHE_STATE.EMPTYING


    def append(self, frame: NnFrame | None, feat: Tensor | None = None) -> None:
        """
        frame: incoming frame, frame.tensor is (1,3,H,W)
        feat:  encoder features of frame.tensor, (1,4,H,W), computed once by the
               caller (nn_module.encode_frame). Required when frame is not None.
        """
        if frame is None:
            return
        if feat is None:
            raise ValueError("cache: append() needs the frame features")

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor]
            self.feats = [feat]
            self.frames = [frame]
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self.tensors.append(frame.tensor)
            self.feats.append(feat)
            self.frames.append(frame)
            # dlogger.debug(f"cache frame no: " + ", ".join(str(f.f_no) for f in self.frames))

            if len(self.frames) == self.radius + 1:
                # [f0 f1 f2 f3] -> [f3 f2 f1 f0 f1 f2 f3]
                # Same mirroring for images and features
                self.tensors = self.tensors[self.radius:0:-1] + self.tensors
                self.feats = self.feats[self.radius:0:-1] + self.feats
                if frame.last:          # N == radius + 1
                    self._enter_emptying()
                else:
                    self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")
            self.tensors.append(frame.tensor)
            self.feats.append(feat)
            self.frames.append(frame)
            if frame.last:
                self._enter_emptying()

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def _fill_tail_padding(self) -> None:
        # Complete the window with end-of-clip mirror padding (images + features)
        while len(self.tensors) < self.window_size:
            self.tensors.append(self.tail[self.pad_i])
            self.feats.append(self.feat_tail[self.pad_i])
            self.pad_i += 1


    def get_window(self) -> tuple[Tensor, Tensor] | None:
        """
        Returns (x, feats) ready for the network, or None if not ready:
          x:     (1,7,3,H,W)
          feats: (1,7,4,H,W)
        Order is [0, -3, -2, -1, +1, +2, +3] (center first).
        """
        if self.state == _CACHE_STATE.PROCESSING:
            # dlogger.debug(f"generated a window with frame no: " + ", ".join(str(f.f_no) for f in self.frames))
            pass
        elif self.state == _CACHE_STATE.EMPTYING:
            self._fill_tail_padding()
        else:
            return None

        c = self.radius  # center index in chronological order (3 for a 7 window)

        # Chronological [-3,-2,-1,0,+1,+2,+3] -> center first [0,-3,-2,-1,+1,+2,+3]
        t = self.tensors
        f = self.feats
        x = torch.cat([t[c], *t[:c], *t[c + 1:]], dim=0).unsqueeze(0)
        feats = torch.cat([f[c], *f[:c], *f[c + 1:]], dim=0).unsqueeze(0)
        return x, feats


    def current_frame(self) -> NnFrame | None:
        if self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING):
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            # Drop the oldest image/feature references so the GPU memory can be reused
            self.tensors.pop(0)
            self.feats.pop(0)
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None




class CudaTemporalInferenceThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=debug)


    def initialize(
        self,
        model: PyTorchModel | None,
        device: str = "cuda:0",
        dtype: Hdtype = 'fp32',
        prescale: tuple[int, int, int] | float | None = None,
    ) -> None:
        self.model = model
        self.cache = TemporalFrameCacheMirroring(window_size=model.arch.infer_type.inputs)
        self.cuda_session: PyTorchSession = nnlib.session(model=model)
        self.cuda_session.initialize(
            device=device,
            dtype=dtype,
            warmup=False,
        )
        self.infer_stream = torch.cuda.Stream(device)

        if isinstance(prescale, list | tuple):
            self.prescale = (prescale[0], prescale[1])
        else:
            self.prescale = prescale


    def run(self):
        self.perform_cuda_infer()
        while not self.in_queue.empty():
            self.in_queue.get_nowait()


    @torch.inference_mode()
    def perform_cuda_infer(self) -> None:
        verbose: bool = self.verbose
        if verbose:
            dlogger.debug(blue(f"[V][I][CUDA] Cuda TemporalInferenceThread"))
        in_queue: Queue = self.in_queue

        cache: TemporalFrameCacheMirroring = self.cache
        cuda_stream = self.infer_stream
        session_dtype: torch.dtype = self.cuda_session.dtype
        nn_module = self.cuda_session.module

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    dlogger.debug(blue("[V][I][CUDA] waiting"))

                frame = None
                if not cache.emptying():
                    # receive frame from consumer
                    input = in_queue.get(block=True)
                    if input is None or self._stop_event.is_set():
                        break
                    frame: NnFrame = input

                    wait_frame_ready(frame=frame, stream=cuda_stream)
                    if verbose:
                        verbose_prefix: str = f"[V][I][CUDA][{frame.f_no}]"

                    # Resize before inference
                    if self.prescale is not None:
                        if isinstance(self.prescale, list | tuple):
                            gpu_resize_to_(
                                frame=frame,
                                out_size=self.prescale,
                                interpolation_method="bicubic"
                            )
                        else:
                            gpu_resize_(
                                frame=frame,
                                scale_factor=self.prescale,
                                interpolation_method="bicubic"
                            )

                    # dlogger.debug(purple(f"received frame no.{frame.f_no}"), "last" if frame.last else "")
                    # Append frame to cache (resized)
                    in_tensor = frame.tensor
                    in_tensor = in_tensor.to(dtype=session_dtype)
                    frame.tensor = in_tensor.contiguous()
                    if verbose:
                        dlogger.debug(f"cache {frame.f_no}: {in_tensor.data_ptr()}")

                    # Encode the incoming frame exactly once (1 frame instead of 7 per step).
                    # The features stay on the GPU, on the same stream.
                    in_feat = nn_module.encode_frame(frame.tensor)

                    # CHANGED: the cache now stores the features next to the image
                    cache.append(frame=frame, feat=in_feat)


                # elif verbose:
                #     dlogger.debug(blue(f"[V][I][CUDA] emptying"))

                # Do not process if cache is not ready
                if not cache.is_ready():
                    if verbose:
                        dlogger.debug(blue(f"[V][I][CUDA] ask for frame"))
                    self.producer.set_produce_flag()
                    continue

                # Cache is ready, perform inference
                window = cache.get_window()
                if window is None:
                    raise ValueError("Not enough frames in window, why?")
                in_tensors, in_feats = window   # (1,7,3,H,W), (1,7,4,H,W)
                if verbose:
                    dlogger.debug(f"   window: {in_tensors.data_ptr()}, feats: {in_feats.data_ptr()}")


                # inference
                #-----------------------------------------------------------------------------
                out_tensor = nn_module(in_tensors, feats=in_feats)

                out_tensor = torch.clamp_(out_tensor, 0., 1.)
                out_frame = cache.current_frame()
                out_frame.tensor = out_tensor

                mark_frame_ready(frame=out_frame, stream=cuda_stream)
                if verbose:
                    dlogger.debug(blue(f"{verbose_prefix} send to encoder: {out_tensor.shape}, {out_tensor.dtype}, {out_tensor.data_ptr()}"))
                self.consumer.put_frame(out_frame)

                if cache.is_empty():
                    dlogger.debug(blue(f"[V][I][CUDA] cache is empty, break"))
                    break

        if verbose:
            dlogger.debug(blue(f"[V][I][CUDA] Ended"))

        # Send a poison pill ?
        self.consumer.put(None)
