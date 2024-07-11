import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import sys

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import glob
import re

from lib.test.utils.load_text import load_text

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        self.base_path = env.carotidartery_path
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
            if isinstance(bbox, dict):
                for obj_id, box in bbox.items():
                    x, y, w, h = [int(v) for v in box]
                    cv.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            else:
                x, y, w, h = [int(v) for v in bbox]
                cv.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            return image

        def save_image_with_bbox(image, predicted_bbox, ground_truth_bbox, seq_name, frame_num):
            output_dir = os.path.join("test", "visual_results", seq_name)
            os.makedirs(output_dir, exist_ok=True)
            image_with_predicted_bbox = draw_bounding_box(image.copy(), predicted_bbox, color=(255, 0, 0), thickness=2)
            image_with_both_bboxes = draw_bounding_box(image_with_predicted_bbox, ground_truth_bbox, color=(0, 255, 0), thickness=2)
            cv.imwrite(os.path.join(output_dir, f"{frame_num + 1}.png"), cv.cvtColor(image_with_both_bboxes, cv.COLOR_RGB2BGR))

        
        def create_video_from_images(seq_name, fps=30):
            visual_results_folder = os.path.join("test", "visual_results")
            video_results_folder = os.path.join("test", "video_results")
            
            os.makedirs(video_results_folder, exist_ok=True)
            
            image_folder = os.path.join(visual_results_folder, seq_name)
            video_path = os.path.join(video_results_folder, f"{seq_name}.mp4")
            
            images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
            
            def numerical_sort(value):
                parts = re.split(r'(\d+)', value)
                return [int(part) if part.isdigit() else part for part in parts]
            
            images.sort(key=numerical_sort)
            
            if not images:
                print(f"No images found in {image_folder} to create video.")
                return
            
            first_image_path = os.path.join(image_folder, images[0])
            frame = cv.imread(first_image_path)
            height, width, _ = frame.shape
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video = cv.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for image in images:
                image_path = os.path.join(image_folder, image)
                frame = cv.imread(image_path)
                video.write(frame)
            
            video.release()
            print(f"Video saved at {video_path}")

        def combine_videos(output_path, fps=30):
            video_results_folder = os.path.join("test", "video_results")
            video_files = glob.glob(os.path.join(video_results_folder, "*.mp4"))
            
            if not video_files:
                print("No video files found to combine.")
                return
            
            # Sort video files numerically by their sequence names
            def numerical_sort(value):
                parts = re.split(r'(\d+)', os.path.basename(value))
                return [int(part) if part.isdigit() else part for part in parts]
            
            video_files.sort(key=numerical_sort)
            
            # Read the first video to get dimensions
            first_video = cv.VideoCapture(video_files[0])
            if not first_video.isOpened():
                print(f"Error opening video file {video_files[0]}")
                return
            
            frame_width = int(first_video.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(first_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            first_video.release()
            
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 files
            combined_video = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            for video_file in video_files:
                video = cv.VideoCapture(video_file)
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    combined_video.write(frame)
                video.release()
            
            combined_video.release()
            print(f"Combined video saved at {output_path}")

        def _read_image(image_file):
            if isinstance(image_file, str):
                im = cv.imread(image_file)
                return cv.cvtColor(im, cv.COLOR_BGR2RGB)
            elif isinstance(image_file, list) and len(image_file) == 2:
                return decode_img(image_file[0], image_file[1])
            else:
                raise ValueError("type of image_file should be str or list")

        anno_path = '{}/{}/bounding_boxes_vessel.txt'.format(self.base_path, seq.name)
        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        # Initialize
        image = self._read_image(seq.frames[0])
        init_info['seq_name'] = seq.name
        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        save_image_with_bbox(image, init_default['target_bbox'], ground_truth_rect[0], init_info['seq_name'], 0)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = _read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            save_image_with_bbox(image, out['target_bbox'], ground_truth_rect[frame_num], init_info['seq_name'], frame_num)
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        create_video_from_images(init_info['seq_name'])
        # After processing all sequences, combine videos
        # 
        # combine_videos(output_path=os.path.join("test", "video_results", "combined_video.mp4"))

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



