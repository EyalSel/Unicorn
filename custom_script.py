import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import os
from absl import app, flags
from itertools import chain, repeat
from pathlib import Path
from unicorn.data import ValTransform
from unicorn.data.datasets.datasets_wrapper import Dataset
from unicorn.evaluators import MOTEvaluator
from unicorn.exp import ExpTrack

flags.DEFINE_multi_string(
    'avi_files',
    default=None,
    required=False,
    help="List of avi files to generate ground truth for")

flags.DEFINE_string(
    'external_detections_base_path',
    default=None,
    required=False,
    help=("A path to detections made by another model to be used instead of "
          "Unicorn's own detector."))

FLAGS = flags.FLAGS


class MEVASensor:
    """
    Copied from ad_config_search
    """

    def __init__(self, data_path, eager):
        self.data_path = data_path
        cap = cv2.VideoCapture(str(data_path))
        self.total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = True
        self.eager = eager
        self.released = False
        if self.eager:
            self.all_data = []
            ret, frame = cap.read()
            while ret:
                self.all_data.append({"center_camera_feed": frame})
                ret, frame = cap.read()
            cap.release()
        else:
            self.cap = cap
            self.next_frame = 0

    def total_num_frames(self):
        return self.total_length

    def __len__(self):
        return self.total_num_frames()

    def get_frame(self, frame_index):
        if self.eager:
            return self.all_data[frame_index]
        else:
            assert self.next_frame == frame_index, (self.next_frame,
                                                    frame_index)
            if frame_index == self.total_num_frames():
                raise IndexError()
            ret, frame = self.cap.read()
            assert ret, frame_index
            if frame_index == self.total_num_frames() - 1:
                self.cap.release()
                self.released = True
            self.next_frame += 1
            return {"center_camera_feed": frame[:, :, ::-1]}

    def __del__(self):
        if not self.released:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.get_frame(self.next_frame)["center_camera_feed"]
        except IndexError:
            raise StopIteration()
        return {
            "frame": self.next_frame,
            "img": result,
            "height": self.height,
            "width": self.width,
            "data_path": self.data_path
        }


class Exp(ExpTrack):
    """
    Copied from exps/default/unicorn_track_large_mot_challenge.py
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split(".")[0]
        self.backbone_name = "convnext_large"
        self.in_channels = [384, 768, 1536]
        self.pretrain_name = "unicorn_det_convnext_large_800x1280"
        self.mot_test_name = "motchallenge"
        self.num_classes = 1
        self.mhs = False

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        assert not is_distributed

        valdataset = MOTMEVADataset(img_size=self.test_size, )

        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset,
                                                 **dataloader_kwargs)

        return val_loader


class MOTMEVADataset(Dataset):
    """
    Adapted from MOTDataset in data/datasets/mot.py
    """

    def __init__(
            self,
            img_size=(608, 1088),
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by
        COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.next_index = 0
        if FLAGS.avi_files is None:
            files = list(Path("../ad-config-search/MEVA").glob("*.avi"))
        else:
            files = [
                Path("../ad-config-search/MEVA") / fn for fn in FLAGS.avi_files
            ]
        readers = [MEVASensor(str(f), eager=False) for f in files]
        self.total_length = sum(
            [reader.total_num_frames() for reader in readers])
        self.iterator_chained = chain(*[
            zip(repeat(video_id), reader)
            for video_id, reader in enumerate(readers)
        ])
        self.external_detections = None
        if FLAGS.external_detections_base_path is not None:
            # There is no standardized naming system at the moment, so this is
            # an ad-hoc conversion from the AVI file name to the DETA detection
            # predictions.
            video_names = [f.name.replace(".avi", "") for f in files]
            self.external_detections = [
                np.load(
                    Path(FLAGS.external_detections_base_path) /
                    f"preds--{name}__DETA.pl.npy") for name in video_names
            ]

        # for single class inference, used by coco for detection output
        # organization
        self.class_ids = [1]

        self.preproc = ValTransform()

    def __len__(self):
        return self.total_length

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and
        pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0
                    to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        assert index == self.next_index, (index, self.next_index)
        self.next_index += 1

        video_id, img_dict = next(self.iterator_chained)
        img = img_dict["img"]
        fake_image_path = Path(img_dict["data_path"]).name.replace(".avi", "")
        fake_image_path = fake_image_path + "/img{}.jpg".format(
            str(img_dict["frame"]).zfill(5))
        external_detections = (
            0 if self.external_detections is None else
            self.external_detections[video_id][img_dict["frame"] - 1])
        img_info = (img_dict["height"], img_dict["width"], img_dict["frame"],
                    video_id, fake_image_path, self.external_detections
                    is not None, external_detections)
        # imgs, _, info_imgs, ids
        # imgs, None, (height, width, frame_id, video_id, file_name), index
        # where file_name = "video_name/img_{index}.jpg" fake path template

        if self.preproc is not None:
            img, _ = self.preproc(img, None, self.input_dim)
        return img, [-np.inf], img_info, np.array([index])


class Args:
    min_box_area = 100


# exp.output_dir ++ comes from BaseExp
# exp.get_model ++
# exp.get_eval_loader ++
# exp.test_conf ++
# exp.nmsthre ++
# exp.num_classes ++
# exp.test_size ++
# exp.grid_sample ++


def main(_):
    exp = Exp()
    experiment_name = "MEVA-dir"
    is_distributed = False
    cudnn.benchmark = True
    rank = 0
    file_name = os.path.join(exp.output_dir, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    model = exp.get_model(load_pretrain=False)

    ckpt_file = (
        "Unicorn_outputs/unicorn_track_large_mot_challenge/best_ckpt.pth")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"],
                                                          strict=False)
    print("missing keys:", missing_keys)
    print("unexpected keys", unexpected_keys)

    val_loader = exp.get_eval_loader(1, False, False)
    evaluator = MOTEvaluator(args=Args,
                             dataloader=val_loader,
                             img_size=exp.test_size,
                             confthre=exp.test_conf,
                             nmsthre=exp.nmsthre,
                             num_classes=exp.num_classes,
                             mask_thres=0.3)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    trt_file = None
    decoder = None

    # start evaluate

    *_, summary = evaluator.evaluate_omni(model,
                                          is_distributed,
                                          False,
                                          trt_file,
                                          decoder,
                                          exp.test_size,
                                          results_folder,
                                          grid_sample=exp.grid_sample)


if __name__ == '__main__':
    app.run(main)
