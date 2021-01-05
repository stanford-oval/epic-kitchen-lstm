from .video_record import VideoRecord


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]
        self._temp_verb = -1
        self._temp_noun = -1
        self._temp_valid = False

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return self._series['start_frame'] - 1

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def temp_label(self):
        assert self._temp_valid
        return {'verb': self._temp_verb if self._temp_valid else -1,
                'noun': self._temp_noun if self._temp_valid else -1}

    def set_temp_label(self, verb, noun):
        self._temp_verb = verb
        self._temp_noun = noun
        self._temp_valid = True

    def invalidate_temp(self):
        self._temp_valid = False

    @property
    def metadata(self):
        return {'narration_id': self._index}