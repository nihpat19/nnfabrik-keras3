import json
import os
import numpy as np
import h5py
import keras
import tifffile
import nibabel as nib
import glob
import math

class DeepGenerator(keras.utils.Sequence):
    """
    This class instantiante the basic Generator Sequence object
    from which all Deep Interpolation generator should be generated.

    Parameters:
    json_path: a path to the json file used to parametrize the generator

    Returns:
    None
    """

    def __init__(self, generator_param):
        self.json_data = generator_param
        self.local_mean = 1
        self.local_std = 1

    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape[1:]

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape[1:]

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return [np.array([]), np.array([])]

    def __get_norm_parameters__(self, idx):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample

        Parameters:
        idx index of the sample

        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std

class SequentialGenerator(DeepGenerator):
    """This generator stores shared code across generators that have a
    continous temporal direction upon which start_frame, end_frame,
    pre_frame,... are used to to generate a list of samples. It is an
    intermediary class that is meant to be extended with details of
    how datasets are loaded."""

    def __init__(self, generator_param):
        "Initialization"
        super().__init__(generator_param)

        # We first store the relevant parameters
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = True

        if "pre_post_omission" in self.json_data.keys():
            self.pre_post_omission = self.json_data["pre_post_omission"]
        else:
            self.pre_post_omission = 0

        # load parameters that are related to training jobs
        self.batch_size = self.json_data["batch_size"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        # Loading limit parameters
        self.start_frame = self.json_data["start_frame"]
        self.end_frame = self.json_data["end_frame"]

        # start_frame starts at 0
        # end_frame is compatible with negative frames. -1 is the last
        # frame.

        # We initialize the epoch counter
        self.epoch_index = 0

    def _update_end_frame(self, total_frame_per_movie):
        """Update end_frame based on the total number of frames available.
        This allows for truncating the end of the movie when end_frame is
        negative."""

        # This is to handle selecting the end of the movie
        if self.end_frame < 0:
            self.end_frame = total_frame_per_movie+self.end_frame
        elif total_frame_per_movie <= self.end_frame:
            self.end_frame = total_frame_per_movie-1

    def _calculate_list_samples(self, total_frame_per_movie):

        # We first cut if start and end frames are too close to the edges.
        self.start_sample = np.max([self.pre_frame
                                    + self.pre_post_omission,
                                    self.start_frame])
        self.end_sample = np.min([self.end_frame, total_frame_per_movie - 1 -
                                  self.post_frame - self.pre_post_omission])

        if (self.end_sample - self.start_sample+1) < self.batch_size:
            raise Exception("Not enough frames to construct one " +
                            str(self.batch_size) + " frame(s) batch between " +
                            str(self.start_sample) +
                            " and "+str(self.end_sample) +
                            " frame number.")

        # +1 to make sure end_samples is included
        self.list_samples = np.arange(self.start_sample, self.end_sample+1)

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def on_epoch_end(self):
        """We only increase index if steps_per_epoch is set to positive value.
        -1 will force the generator to not iterate at the end of each epoch."""
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def __len__(self):
        "Denotes the total number of batches"
        return math.ceil(len(self.list_samples) / self.batch_size)

    def generate_batch_indexes(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        if end_ind < self.list_samples.shape[0]:
            indexes = np.arange(start_ind, end_ind)
            shuffle_indexes = self.list_samples[indexes]
        else:
            shuffle_indexes = self.list_samples[start_ind:]
        return shuffle_indexes



class SingleTifGenerator(SequentialGenerator):
    """This generator is used when dealing with a single tif file storing a
    continous movie recording. Each frame can be arbitrary (x,y) size but
    should be consistent through training. a maximum of 1000 frames are pulled
    from the beginning of the movie to estimate mean and std."""

    def __init__(self, generator_param):
        "Initialization"
        super().__init__(generator_param)

        self.raw_data_file = self.json_data["train_path"]

        with tifffile.TiffFile(self.raw_data_file) as tif:
            self.raw_data = tif.asarray()

        self.total_frame_per_movie = self.raw_data.shape[0]

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)

        average_nb_samples = np.min([self.total_frame_per_movie, 1000])
        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

    def __getitem__(self, index):
        shuffle_indexes = self.generate_batch_indexes(index)

        input_full = np.zeros(
            [
                self.batch_size,
                self.raw_data.shape[1],
                self.raw_data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [self.batch_size, self.raw_data.shape[1],
             self.raw_data.shape[2], 1],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # X : (n_samples, *dim, n_channels)

        input_full = np.zeros(
            [
                1,
                self.raw_data.shape[1],
                self.raw_data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [1, self.raw_data.shape[1],
             self.raw_data.shape[2], 1], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index !=
                                      index_frame - index_padding]
            input_index = input_index[input_index !=
                                      index_frame + index_padding]

        data_img_input = self.raw_data[input_index, :, :]
        data_img_output = self.raw_data[index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float32") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float32") - self.local_mean
        ) / self.local_std
        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0],
                    : img_out_shape[1], 0] = data_img_output

        return input_full, output_full

def generator_function(seed: int, **config):
    training_generator_params = config.get("generator_params")
    test_generator_params = config.get("test_generator_params")
    return {
        "training_generator": SingleTifGenerator(training_generator_params),
        "test_generator": SingleTifGenerator(test_generator_params)
    }