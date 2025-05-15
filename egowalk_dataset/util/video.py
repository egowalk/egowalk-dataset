import contextlib
import av
import av.container
import av.stream
import numpy as np

from pathlib import Path
from typing import Union, Literal, List, Optional
from egowalk_dataset.misc.indexing import (StartingSequence,
                                           EndingSequence,
                                           IndependentSequence)


IdxType = Union[int, StartingSequence, EndingSequence, IndependentSequence]
FormatType = Literal["rgb", "depth"]


def convert_raw_frame(frame: av.VideoFrame,
                      format: Literal["rgb", "depth"]) -> np.ndarray:
    if format == "rgb":
        return frame.to_ndarray(format="rgb24")
    elif format == "depth":
        frame = frame.to_ndarray(format="gray16le")
        frame = frame.astype(np.float32) / 1000.0
        frame[frame == 0.] = np.nan
        return frame
    else:
        raise ValueError(f"Invalid format: {format}")


def calculate_time_idx(stream: av.stream.Stream,
                       frame_idx: int) -> int:
    return int(frame_idx * stream.time_base.denominator / stream.base_rate.numerator)


def collect_frames_sequence(container: av.container.Container,
                            stream: av.stream.Stream,
                            seq: Union[StartingSequence, EndingSequence],
                            fmt: FormatType) -> Optional[np.ndarray]:
    frames = []
    idxs = [seq.start_idx if isinstance(seq, StartingSequence) else seq.end_idx]
    step = seq.step if isinstance(seq, StartingSequence) else -seq.step
    for _ in range(0, seq.n_steps):
        idxs.append(idxs[-1] + step)
    if isinstance(seq, EndingSequence):
        idxs.reverse()

    idxs = [calculate_time_idx(stream, idx) for idx in idxs]
    idxs = [idx for idx in idxs if 0 <= idx < stream.frames]
    if len(idxs) == 0:
        return None
    
    cnt = 0
    container.seek(idxs[0], stream=stream)
    for frame in container.decode(stream):
        if frame.pts == idxs[cnt]:
            frames.append(frame)
            cnt += 1
            if cnt == len(idxs):
                break

    return np.stack([convert_raw_frame(frame, fmt) for frame in frames], axis=0)


def collect_frames_independent_seq(container: av.container.Container,
                                   stream: av.stream.Stream,
                                   seq: Union[IndependentSequence, int],
                                   fmt: FormatType) -> Optional[np.ndarray]:
    frames = []
    total_frames = stream.frames
    if total_frames == 0:
        total_frames = np.inf

    if isinstance(seq, IndependentSequence):
        idxs = [calculate_time_idx(stream, idx) for idx in seq.indices if 0 <= idx < total_frames]
        if len(idxs) == 0:
            return None
        single = False
    else:
        if not (0 <= seq < total_frames):
            return None
        idxs = [calculate_time_idx(stream, seq)]
        single = True

    for idx in idxs:
        container.seek(idx, stream=stream)
        for frame in container.decode(stream):
            if frame.pts == idx:
                frames.append(frame)
                break

    frames = [convert_raw_frame(frame, fmt) for frame in frames]
    if single:
        return frames[0]
    else:
        return np.stack(frames, axis=0)


def decode_frame(container: Union[av.container.Container, str, Path],
                 frame_idx: Union[IdxType, List[IdxType]],
                 fmt: FormatType) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    if isinstance(container, (str, Path)):
        container = av.open(str(container))
        close_container = True
    else:
        close_container = False

    result = []
    if not isinstance(frame_idx, list):
        frame_idx = [frame_idx]
        single = True
    else:
        single = False

    stream = container.streams.video[0]

    try:
        for idx in frame_idx:
            if isinstance(idx, int) or isinstance(idx, IndependentSequence):
                target = collect_frames_independent_seq(container, stream, idx, fmt)
            elif isinstance(idx, StartingSequence) or isinstance(idx, EndingSequence):
                target = collect_frames_sequence(container, stream, idx, fmt)
            else:
                raise ValueError(f"Invalid frame index: {idx}")
            result.append(target)
    
    except Exception as e:
        raise e
    finally:
        if close_container:
            container.close()

    return result[0] if single else result
