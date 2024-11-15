import cv2
import numpy as np


def thicken_paths(frames, thick_indices, thickness=7):
    valid_frames_thickened = []

    for frame in frames:
        try:
            ov = np.squeeze(frame.overlay, axis=0)

            frame_index = np.unique(ov)[1]

            assert ov.shape[0] == ov.shape[1]

            if frame_index in thick_indices:
                valid_frames_thickened.append(frame)
            #     kernel = np.ones((thickness, thickness), np.uint8) * frame_index
            #     frame.overlay = np.expand_dims(cv2.dilate(ov, kernel, iterations=1), axis=0)
            #
            #     # Path to thicken case
            #     valid_frames_thickened.append(frame)

            # Crossover area, do not thicken
            valid_frames_thickened.append(frame)

        except IndexError as _:
            pass

    return valid_frames_thickened
