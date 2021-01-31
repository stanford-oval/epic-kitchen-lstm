from typing import Sequence

from torch.utils.data import Sampler

from slowfast.datasets import Epickitchens


class SequentialBatchSampler(Sampler[Sequence[int]]):

    def __init__(self, dataset: Epickitchens, batch_num: int):
        super().__init__(Epickitchens)
        video_list = []
        # noinspection PyProtectedMember
        for idx, entry in enumerate(dataset._video_records):
            if len(video_list) == 0 or entry.untrimmed_video_name != video_list[-1]['video_name']:
                video_list += [{
                    "start_idx": idx,
                    "end_idx": idx,
                    "video_name": entry.untrimmed_video_name
                }]
            else:
                video_list[-1]["end_idx"] = idx

        start_num = min(len(video_list), batch_num)

        cur_list = [(v_id, 0) for v_id in range(start_num)]

        next_v_id = start_num

        sequence = []

        while len(cur_list) != 0:
            sequence += [[video_list[v_id]["start_idx"] + sub_idx for v_id, sub_idx in cur_list]]
            new_list = []
            for v_id, sub_idx in cur_list:
                start_idx = video_list[v_id]["start_idx"]
                end_idx = video_list[v_id]["end_idx"]
                if start_idx + sub_idx == end_idx:
                    # need to switch to next video
                    if next_v_id != len(video_list):
                        # switch to next unless there is no more video
                        new_list += [(next_v_id, 0)]
                        next_v_id += 1

                else:
                    new_list += [(v_id, sub_idx + 1)]
            cur_list = new_list

        # noinspection PyProtectedMember
        assert len([item for sub_seq in sequence for item in sub_seq]) == len(dataset._video_records)

        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        return self.sequence.__iter__()
