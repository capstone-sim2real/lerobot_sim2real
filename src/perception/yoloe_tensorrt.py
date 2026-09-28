"""Small TensorRT runtime for the fixed-prompt YOLOE segmentation engine.

TensorRT and CUDA imports stay lazy so the normal application and tests do not
need Jetson-only packages. This module never opens a camera or robot device.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    gain: float
    pad_left: int
    pad_top: int
    input_shape: tuple[int, int]
    original_shape: tuple[int, int]


def letterbox_bgr(frame: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Return normalized RGB NCHW and the exact resize/padding transform."""
    if frame.ndim != 3 or frame.shape[2] != 3 or size < 32:
        raise ValueError("Expected a BGR image and size >= 32")
    h, w = frame.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    right, bottom = dw - left, dh - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    ) / 255.0
    return tensor, LetterboxInfo(
        gain=gain, pad_left=left, pad_top=top,
        input_shape=(size, size), original_shape=(h, w),
    )


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    result = boxes.astype(np.float32, copy=True)
    result[:, [0, 2]] = (result[:, [0, 2]] - info.pad_left) / info.gain
    result[:, [1, 3]] = (result[:, [1, 3]] - info.pad_top) / info.gain
    h, w = info.original_shape
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, w)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, h)
    return result


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0), axis=1)
    area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_b = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou: float,
                maximum: int) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < maximum:
        index = int(order[0])
        keep.append(index)
        order = order[1:]
        if order.size:
            order = order[_iou_one_to_many(boxes[index], boxes[order]) <= iou]
    return np.asarray(keep, dtype=np.int64)


def decode_yoloe_segmentation(
    detections: np.ndarray,
    prototypes: np.ndarray,
    info: LetterboxInfo,
    confidence: float,
    iou: float,
    maximum: int,
) -> list[dict]:
    """Decode the fixed exported contract: xyxy, score, class, 32 coefficients."""
    rows = np.asarray(detections)[0]
    proto = np.asarray(prototypes)[0]
    if rows.ndim != 2 or rows.shape[1] != 6 + proto.shape[0]:
        raise ValueError("Unexpected YOLOE TensorRT output contract")
    rows = rows[rows[:, 4] >= confidence]
    if not len(rows):
        return []
    keep = nms_indices(rows[:, :4], rows[:, 4], iou, maximum)
    rows = rows[keep]
    boxes = scale_boxes(rows[:, :4], info)

    # Ultralytics retina-mask path: combine prototypes, remove letterbox pad,
    # resize logits to the original image, threshold, then crop to each box.
    ph, pw = proto.shape[1:]
    original_h, original_w = info.original_shape
    gain = min(ph / original_h, pw / original_w)
    pad_w = (pw - round(original_w * gain)) / 2
    pad_h = (ph - round(original_h * gain)) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    right, bottom = pw - round(pad_w + 0.1), ph - round(pad_h + 0.1)
    logits = rows[:, 6:] @ proto.reshape(proto.shape[0], -1)
    logits = logits.reshape(-1, ph, pw)[:, top:bottom, left:right]

    results = []
    for row, box, logit in zip(rows, boxes, logits):
        full = cv2.resize(logit, (original_w, original_h), interpolation=cv2.INTER_LINEAR) > 0
        x1, y1, x2, y2 = box
        columns = np.arange(original_w)[None, :]
        lines = np.arange(original_h)[:, None]
        full &= (columns >= x1) & (columns < x2) & (lines >= y1) & (lines < y2)
        if full.any():
            results.append({
                "bbox_px": box.tolist(),
                "confidence": float(row[4]),
                "class_id": int(row[5]),
                "mask": full.astype(np.uint8),
            })
    return results


class TensorRTEngine:
    """Static-shape TensorRT 10 engine using cuda-python for memory transfer."""

    def __init__(self, path: str | Path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._cuda = cudart
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self._engine is None:
            raise RuntimeError("Could not deserialize TensorRT engine")
        self._context = self._engine.create_execution_context()
        self._stream = self._check(cudart.cudaStreamCreate())
        self.arrays: dict[str, np.ndarray] = {}
        self._pointers: dict[str, int] = {}
        self._host_pointers: dict[str, int] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError("Dynamic TensorRT engines are not supported")
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            host_pointer = self._check(cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault))
            buffer = (ctypes.c_byte * nbytes).from_address(int(host_pointer))
            array = np.ctypeslib.as_array(buffer).view(dtype).reshape(shape)
            pointer = self._check(cudart.cudaMalloc(nbytes))
            if not self._context.set_tensor_address(name, int(pointer)):
                raise RuntimeError(f"Could not bind TensorRT tensor {name}")
            self.arrays[name], self._pointers[name] = array, pointer
            self._host_pointers[name] = host_pointer
        expected = {"images", "output0", "output1"}
        if set(self.arrays) != expected or self.arrays["images"].shape != (1, 3, 640, 640):
            raise ValueError(f"Unexpected engine tensors: {self.arrays.keys()}")

    def _check(self, result):
        if result[0] != self._cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error: {result[0]}")
        return result[1] if len(result) == 2 else None

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if tensor.shape != self.arrays["images"].shape or tensor.dtype != np.float32:
            raise ValueError("TensorRT input must be float32 NCHW 1x3x640x640")
        cudart = self._cuda
        np.copyto(self.arrays["images"], tensor)
        start = time.perf_counter()
        self._check(cudart.cudaMemcpyAsync(
            self._pointers["images"], self.arrays["images"].ctypes.data, tensor.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream,
        ))
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT inference failed")
        for name in ("output0", "output1"):
            array = self.arrays[name]
            self._check(cudart.cudaMemcpyAsync(
                array.ctypes.data, self._pointers[name], array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream,
            ))
        self._check(cudart.cudaStreamSynchronize(self._stream))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self.arrays["output0"].copy(), self.arrays["output1"].copy(), elapsed_ms

    def close(self) -> None:
        if getattr(self, "_stream", None) is None:
            return
        for pointer in self._pointers.values():
            self._check(self._cuda.cudaFree(pointer))
        for pointer in self._host_pointers.values():
            self._check(self._cuda.cudaFreeHost(pointer))
        self._check(self._cuda.cudaStreamDestroy(self._stream))
        self._pointers.clear()
        self._host_pointers.clear()
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
