from __future__ import annotations
from enum import IntEnum
from queue import Queue
import time

from .cuda_temporalfix_thread import _CACHE_STATE

try:
    import tensorrt as trt
except:
    try:
        import tensorrt_rtx as trt
    except:
        pass
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
    TensorRtSession,
)
from gpu_algos import (
    gpu_resize_to_,
    gpu_resize_,
)


class TrtTemporalFrameCacheMirroring:

    def __init__(self, window_size: int = 7) -> None:
        self.window_size: int = window_size
        self.radius: int = window_size // 2
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []          # (1,3,H,W), chronological order
        self.feats: list[Tensor | None] = []     # (1,4,H,W) or None while not yet encoded
        self.tail: list[Tensor] = []             # end-of-clip mirror padding (images)
        self.feat_tail: list[Tensor | None] = [] # end-of-clip mirror padding (features)
        self.pending: Tensor | None = None       # image whose feature is not computed yet
        self._zero_feat: Tensor | None = None    # placeholder for the slots with no feature
        self._masks: dict[tuple, Tensor] = {}    # GPU mask cache, a few patterns only
        self.pad_i: int = 0
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.feats.clear()
        self.tail.clear()
        self.feat_tail.clear()
        self.pending = None
        self.pad_i = 0
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING))


    def is_empty(self) -> bool:
        return bool(self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT))


    def has_pending(self) -> bool:
        """True if the last appended frame has not been encoded yet."""
        return self.pending is not None


    def _enter_emptying(self) -> None:
        # Mirror padding: f(N-2), f(N-3), f(N-4). These frames were already encoded
        # (only the last frame can still be pending), so their features are available.
        self.tail = [self.tensors[-(k + 1)] for k in range(1, self.radius + 1)]
        self.feat_tail = [self.feats[-(k + 1)] for k in range(1, self.radius + 1)]
        self.pad_i = 0
        self.state = _CACHE_STATE.EMPTYING


    def _push(self, frame: NnFrame) -> None:
        self.tensors.append(frame.tensor)
        self.feats.append(None)                  # encoded by the next engine call
        self.frames.append(frame)
        self.pending = frame.tensor


    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self._zero_feat is None:
            _, _, h, w = frame.tensor.shape
            self._zero_feat = torch.zeros(
                (1, 4, h, w), dtype=frame.tensor.dtype, device=frame.tensor.device
            )

        if self.pending is not None:
            raise ValueError("cache: previous frame was not encoded (commit_feat missing)")

        if self.state == _CACHE_STATE.INIT:
            self.tensors, self.feats, self.frames = [], [], []
            self._push(frame)
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self._push(frame)
            if len(self.frames) == self.radius + 1:
                # [f0 f1 f2 f3] -> [f3 f2 f1 f0 f1 f2 f3]
                # f3 is pending: both of its slots (first and last) are replaced by commit_feat
                self.tensors = self.tensors[self.radius:0:-1] + self.tensors
                self.feats = self.feats[self.radius:0:-1] + self.feats
                if frame.last:
                    self._enter_emptying()
                else:
                    self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")
            self._push(frame)
            if frame.last:
                self._enter_emptying()

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def commit_feat(self, feat: Tensor) -> None:
        """Store the feature of the pending frame in every slot that references it."""
        if self.pending is None:
            return
        # Only one frame can be pending, so every None refers to it
        self.feats = [feat if f is None else f for f in self.feats]
        self.pending = None


    def _mask(self, pattern: tuple[int, ...], dtype: torch.dtype, device) -> Tensor:
        m = self._masks.get(pattern)
        if m is None:
            m = torch.tensor(pattern, dtype=dtype, device=device).view(1, 7, 1, 1, 1)
            self._masks[pattern] = m
        return m


    def get_window(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """
        Returns (x, feats, new_frame, new_mask) for the engine, or None if not ready.
          x:         (1,7,3,H,W)   order [0,-3,-2,-1,+1,+2,+3] (center first)
          feats:     (1,7,4,H,W)   zeros where the feature is not known yet
          new_frame: (1,3,H,W)     frame to encode (a dummy one if nothing is pending)
          new_mask:  (1,7,1,1,1)   1 where the slot takes the feature of new_frame
        """
        if self.state == _CACHE_STATE.EMPTYING:
            while len(self.tensors) < self.window_size:
                self.tensors.append(self.tail[self.pad_i])
                self.feats.append(self.feat_tail[self.pad_i])
                self.pad_i += 1
        elif self.state != _CACHE_STATE.PROCESSING:
            return None

        c = self.radius
        order = [c, *range(c), *range(c + 1, self.window_size)]   # center first

        t = [self.tensors[i] for i in order]
        f = [self._zero_feat if self.feats[i] is None else self.feats[i] for i in order]
        pattern = tuple(int(self.feats[i] is None) for i in order)

        x = torch.cat(t, dim=0).unsqueeze(0)
        feats = torch.cat(f, dim=0).unsqueeze(0)
        new_frame = self.pending if self.pending is not None else t[0]
        mask = self._mask(pattern, x.dtype, x.device)
        return x, feats, new_frame, mask


    def current_frame(self) -> NnFrame | None:
        if self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING):
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            self.tensors.pop(0)
            self.feats.pop(0)
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None



class TrtTemporalInferenceThread(BaseThread):
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
        self.cache = TrtTemporalFrameCacheMirroring(window_size=model.arch.infer_type.inputs)
        self.trt_session: TensorRtSession = nnlib.session(model=model)
        self.trt_session.initialize(
            device=device,
            dtype=dtype,
            warmup=False,
        )

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
            verbose_prefix: str = "[V][I][TRT]"
            dlogger.debug(blue(f"{verbose_prefix} Cuda TemporalInferenceThread"))
        in_queue: Queue = self.in_queue
        cache: TrtTemporalFrameCacheMirroring = self.cache

        session = self.trt_session
        cuda_stream = torch.cuda.Stream(session.device)
        session_dtype: torch.dtype = session.dtype
        context, engine = session.context, session.engine

        # Resolve the IO tensors once, not at every frame
        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        is_input = {n: engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in io_names}
        is_dynamic = {n: -1 in tuple(engine.get_tensor_shape(n)) for n in io_names if is_input[n]}

        dummy: dict[str, Tensor] = {}

        def run_engine(x, feats, new_frame, mask) -> tuple[Tensor, Tensor]:
            h, w = x.shape[-2:]
            # Fresh output buffers at every call: `out` goes to the encoder thread and
            # `new_feat` is kept in the cache, so neither can be overwritten by the next call.
            out = torch.empty((1, 3, h, w), dtype=session_dtype, device=x.device)
            new_feat = torch.empty((1, 4, h, w), dtype=session_dtype, device=x.device)
            bound = {
                "x": x, "feats": feats, "new_frame": new_frame, "new_mask": mask,
                "output": out, "new_feat": new_feat,
            }
            for name in io_names:
                t = bound[name]
                context.set_tensor_address(name, t.data_ptr())
                if is_input[name] and is_dynamic[name]:
                    context.set_input_shape(name, tuple(t.shape))
            context.execute_async_v3(stream_handle=cuda_stream.cuda_stream)
            return out, new_feat


        scale = self.trt_session.model.scale

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
                        verbose_prefix: str = f"[V][I][TRT][{frame.f_no}]"

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
                    frame.tensor = frame.tensor.to(dtype=session_dtype).contiguous()
                    if verbose:
                        dlogger.debug(f"cache {frame.f_no}: {frame.tensor.data_ptr()}")

                    cache.append(frame=frame)


                # elif verbose:
                #     dlogger.debug(blue(f"[V][I][CUDA] emptying"))

                if not cache.is_ready():
                    # Warm-up: the first radius frames never go through a real window.
                    # Run the engine only to get their feature (mask = 0, output discarded).
                    if not dummy:
                        _, _, h, w = frame.tensor.shape
                        dummy["x"] = torch.zeros((1, 7, 3, h, w), dtype=session_dtype, device=frame.tensor.device)
                        dummy["feats"] = torch.zeros((1, 7, 4, h, w), dtype=session_dtype, device=frame.tensor.device)
                        dummy["mask"] = torch.zeros((1, 7, 1, 1, 1), dtype=session_dtype, device=frame.tensor.device)
                    _, new_feat = run_engine(dummy["x"], dummy["feats"], frame.tensor, dummy["mask"])
                    cache.commit_feat(new_feat)
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
                x, feats, new_frame, mask = window
                out_tensor, new_feat = run_engine(x, feats, new_frame, mask)
                # no-op while emptying (nothing pending)
                cache.commit_feat(new_feat)

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
