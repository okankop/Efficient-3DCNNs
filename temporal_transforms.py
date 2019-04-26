import random
import math


class LoopPadding(object):

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample
        out = frame_indices

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        out = frame_indices[:clip_duration]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (clip_duration // 2))
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        rand_end = max(0, vid_duration - clip_duration - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames
