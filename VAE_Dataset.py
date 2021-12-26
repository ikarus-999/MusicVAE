import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import pretty_midi
import collections


class MusicVAEDataset(Sequence):
    def __init__(self, dataframe, split_set: str, batch_size=64):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.split_set = split_set
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(list(self.dataframe[self.dataframe['split'] == self.split_set]['midi_filename'].values)))) / self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len())

    def __read_midi(self, filename):
        pm = pretty_midi.PrettyMIDI(filename)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)
        # 시작 시간에 따른 정렬
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        # prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['velocity'].append(note.velocity)
            notes['start'].append(start)
            notes['end'].append(end)
            # notes['step'].append(start - prev_start)
            # notes['duration'].append(end - start)
            # prev_start = start

        result = np.array([value for key, value in notes.items()]).T
        return result

    def __get_sample(self, df_temp, split_set, idx_temp):
        midi_file_lists = list(df_temp[df_temp['split'] == split_set]['midi_filename'].values)

        data = [self.__read_midi(fname) for i, fname in enumerate(midi_file_lists)]

        return np.array(data[idx_temp])

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        midi_data = [self.__get_sample(self.dataframe, self.split_set, idx) for idx in indexes]

        return tf.convert_to_tensor(midi_data)
