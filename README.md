# SRGA: Semantic Region-Guided Transferable Attacks on Vision-Language Pretraining Models

## 1. Install dependencies

See in `requirements.txt`.

## 2. Prepare datasets and models

Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).

Download the YOLO model weights,  [YOLOv5](https://github.com/ultralytics/yolov5) or [YOLOv8](https://github.com/ultralytics/ultralytics) (the pre-trained weights are provided in the official release pages of each repository). Set the root path of the model weights in `./root`. The checkpoints of the fine-tuned YOLO models are accessible in [YOLOv8 Official Checkpoints](https://github.com/ultralytics/assets/releases), [YOLOv5 Official Checkpoints](https://github.com/ultralytics/yolov5/releases), [CLIP-YOLO Fusion Model](https://huggingface.co/models?search=yolo-clip) and so on.

## 3.Modify the dictionary mapping in attacker.py

Modify the dictionary mapping of YOLO categories based on the downloaded YOLO model weights, and adjust the dictionary for semantics-related mappings according to different datasets (Flickr30k / MSCOCO).

## 4. Attack evaluation

From ALBEF to others models on the Flickr30k dataset:

```
python eval.py --config ./configs/Retrieval_flickr.yaml \
--source_model ALBEF  --source_ckpt ./checkpoint/albef_retrieval_flickr.pth \
--original_rank_index ./std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

