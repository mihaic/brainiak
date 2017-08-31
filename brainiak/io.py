#  Copyright 2017 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""I/O functionality."""

__all__ = [
    "load_boolean_mask",
    "load_images_from_dir",
    "load_labels",
    "save_as_nifti_file",
]

from pathlib import Path
from typing import Callable, Iterable, List, Union

import logging
import nibabel as nib
import numpy as np
import pkg_resources

from bids.grabbids import BIDSLayout
from nibabel.nifti1 import Nifti1Pair
from nibabel.spatialimages import SpatialImage

from .image import mask_image, SingleConditionSpec

logger = logging.getLogger(__name__)


def load_images_from_dir(in_dir: Union[str, Path], suffix: str = "nii.gz",
                         ) -> Iterable[SpatialImage]:
    """Load images from directory.

    For efficiency, returns an iterator, not a sequence, so the results cannot
    be accessed by indexing.

    For every new iteration through the images, load_images_from_dir must be
    called again.

    Parameters
    ----------
    in_dir:
        Path to directory.
    suffix:
        Only load images with names that end like this.

    Yields
    ------
    SpatialImage
        Image.
    """
    if isinstance(in_dir, str):
        in_dir = Path(in_dir)
    files = sorted(in_dir.glob("*" + suffix))
    for f in files:
        logger.debug(
            'Starting to read file %s', f
        )
        yield nib.load(str(f))


def load_images(image_paths: Iterable[Union[str, Path]]
                ) -> Iterable[SpatialImage]:
    """Load images from paths.

    For efficiency, returns an iterator, not a sequence, so the results cannot
    be accessed by indexing.

    For every new iteration through the images, load_images must be called
    again.

    Parameters
    ----------
    image_paths:
        Paths to images.

    Yields
    ------
    SpatialImage
        Image.
    """
    for image_path in image_paths:
        if isinstance(image_path, Path):
            string_path = str(image_path)
        else:
            string_path = image_path
        logger.debug(
            'Starting to read file %s', string_path
        )
        yield nib.load(string_path)


def load_boolean_mask(path: Union[str, Path],
                      predicate: Callable[[np.ndarray], np.ndarray] = None
                      ) -> np.ndarray:
    """Load boolean nibabel.SpatialImage mask.

    Parameters
    ----------
    path
        Mask path.
    predicate
        Callable used to create boolean values, e.g. a threshold function
        ``lambda x: x > 50``.

    Returns
    -------
    np.ndarray
        Boolean array corresponding to mask.
    """
    if not isinstance(path, str):
        path = str(path)
    data = nib.load(path).get_data()
    if predicate is not None:
        mask = predicate(data)
    else:
        mask = data.astype(np.bool)
    return mask


def load_labels(path: Union[str, Path]) -> List[SingleConditionSpec]:
    """Load labels files.

    Parameters
    ----------
    path
        Path of labels file.

    Returns
    -------
    List[SingleConditionSpec]
        List of SingleConditionSpec stored in labels file.
    """
    condition_specs = np.load(str(path))
    return [c.view(SingleConditionSpec) for c in condition_specs]


def save_as_nifti_file(data: np.ndarray, affine: np.ndarray,
                       path: Union[str, Path]) -> None:
    """Create a Nifti file and save it.

    Parameters
    ----------
    data
        Brain data.
    affine
        Affine of the image, usually inherited from an existing image.
    path
        Output filename.
    """
    if not isinstance(path, str):
        path = str(path)
    img = Nifti1Pair(data, affine)
    nib.nifti1.save(img, path)


def load_bids(bids_root: Path, task: str = None, space: str = None
              ) -> Iterable[np.ndarray]:
    """Create masked images from a BIDS dataset.

    Parameter
    ---------
    bids_root
        Path to root of BIDS dataset.
    task
        Task to load. Must be specified for multi-task datasets.
    space
        Registration space to load. Must be specified for multi-space datasets.

    Yields
    ------
    np.ndarray
        Masked image.
    """
    config_path = pkg_resources.resource_filename("brainiak", "bids.json")
    layout = BIDSLayout(str(bids_root), config=config_path)
    subjects = layout.get_subjects()
    if len(layout.get_tasks()) > 1 and task is None:
        raise ValueError("Must specify task for multi-task datasets.")
    general_args = {"modality": "func"}
    spaces = layout.get_spaces()
    if space is not None:
        if space not in spaces:
            raise ValueError("Given space not in dataset.")
        general_args["space"] = space
    tasks = layout.get_tasks()
    if task is not None:
        if task not in tasks:
            raise ValueError("Given task not in dataset.")
        general_args["task"] = task
    for s in subjects:
        subject_args = general_args.copy()
        subject_args["subject"] = s
        image_args = subject_args.copy()
        image_args["type"] = "preproc"
        images = layout.get(**image_args)
        assert len(images) == 1, images
        image = nib.load(images[0].filename)
        mask_args = subject_args.copy()
        mask_args["type"] = "brainmask"
        masks = layout.get(**mask_args)
        assert len(masks) == 1, masks
        mask = load_boolean_mask(masks[0].filename)
        yield mask_image(image, mask)
