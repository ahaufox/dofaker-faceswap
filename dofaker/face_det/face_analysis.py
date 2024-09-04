'''
The following code references:: https://github.com/deepinsight/insightface
'''

import glob
import os.path as osp
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
import onnxruntime

from insightface import model_zoo
from insightface.utils import ensure_available
from insightface.app.common import Face

from dofaker.utils import download_file, get_model_url
import insightface as i

__all__ = ['FaceAnalysis']


class FaceAnalysis:

    def __init__(self,
                 name='buffalo_l',
                 root='weights',
                 allowed_modules=None,
                 **kwargs):
        self.model_dir, _ = download_file(get_model_url(name),
                                          save_dir=root,
                                          overwrite=False)
        print('model dir:', self.model_dir)
        onnxruntime.set_default_logger_severity(3)
        print('device:',onnxruntime.get_device())

        self.models = {}
        # print(self.model_dir)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                raise RuntimeError('model not recognized:{}'.format(onnx_file))
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models.keys() and (allowed_modules is None
                                                        or model.taskname
                                                        in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file,
                      model.taskname)
                del model
        assert 'detection' in self.models.keys() ,"detection model load error"
        self.det_model = self.models['detection']
        

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None, "det_size can't be None."
        self.det_size = det_size

        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id,
                              input_size=det_size,
                              det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)
        # del self.models #用完删除 节约内存
    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype('int')
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype('int')
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age),
                            (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)
        return dimg
