import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class CarotidarteryDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.carotidartery_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])


    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/bounding_boxes_vessel.txt'.format(self.base_path, sequence_name)
        
        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        occlusion_label_path = '{}/{}/zeros.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/zeros.txt'.format(self.base_path, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        # Shrink the bounding boxes
        def shrink_bounding_box(bbox, shrink_factor=0.1):
            x_center, y_center, width, height = bbox
            shrunk_width = width * (1 - shrink_factor)
            shrunk_height = height * (1 - shrink_factor)
            
            return [x_center, y_center, round(shrunk_width, 1), round(shrunk_height, 1)]

        target_class = ""
        
        # shrink the bounding boxes %10 for the getting better accuracy
        ground_truth_rect_shrunk = np.array([shrink_bounding_box(bbox) for bbox in ground_truth_rect])
        

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        # use for expanded, not shrunk
        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        
        # use for expanded, not shrunk
        return Sequence(sequence_name, frames_list, 'carotidartery', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

        # use for not expanded, shrunk
        # frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect_shrunk.shape[0] + 1)]

        # use for not expanded, shrunk
        # return Sequence(sequence_name, frames_list, 'carotidartery', ground_truth_rect_shrunk.reshape(-1, 4),
        #                 object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['carotid-1', 'carotid-5', 'carotid-9', "carotid-16"]
        return sequence_list
