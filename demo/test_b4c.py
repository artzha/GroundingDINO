import os
import pdb
import torch
from torch.utils.data import DataLoader
from brains4cars_dataset import B4CDataset
from groundingdino.util.inference import load_model, load_image, predict, Model
from tqdm import tqdm
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import supervision as sv
import cv2

import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap

import fiftyone as fo
import pickle

CLASSES = [
    "Car", "Bicycle", "Person", "Traffic Sign", "Traffic Light", "Date"
]

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def load_image(img_np):
    # load image
    image_pil = Image.fromarray(img_np)  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def dump_dictionary_to_pickle(dictionary, file_path):
    """
    Dump a Python dictionary to a pickle file.

    Parameters:
        dictionary : dict
            The Python dictionary to be dumped.
        file_path : str
            The path to the pickle file where the dictionary will be stored.
    """
    print("Saving label to ", file_path)
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

def main(
    root_dir: str = '/home/arthur/AMRL/Datasets/Brains4Cars', #'/media/arthur/ExtremePro/kitti_format/coda',
    text_prompt: str =  "Car, Bicycle, Person, Traffic Sign, Traffic Light, Date",
    box_threshold: float = 0.3, 
    text_threshold: float = 0.3,
    export_dataset: bool = False,
    view_dataset: bool = False,
    export_annotated_images: bool = True,
    weights_path : str = "./groundingdino_swinb_cogcoor.pth",
    config_path: str = "./groundingdino/config/GroundingDINO_SwinB.cfg.py",
    subsample: int = None,
    cpu_only: bool = False,
    output_dir: str = "./outputs_paper"
):
    global CLASSES
    CLASSES = [objcls.lower() for objcls in CLASSES]
    model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)
    load_model(config_path, weights_path)

    root_path = Path(root_dir)
    training_data = B4CDataset(CLASSES, root_path, 'road_camera', 'train', create_dataset=False)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, collate_fn=training_data.collate_fn, pin_memory=True)
    validation_data = B4CDataset(CLASSES, root_path, 'road_camera', 'val', create_dataset=False)
    val_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False, collate_fn=training_data.collate_fn, pin_memory=True)
    testing_data = B4CDataset(CLASSES, root_path, 'road_camera', 'test', create_dataset=False)
    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False, collate_fn=training_data.collate_fn, pin_memory=True)
    dataloader_list = [val_dataloader, test_dataloader]

    for dataloader in dataloader_list:
        for img, img_file in tqdm(dataloader):
            if "20141115_095633_1607_1757" not in img_file[0]:
                continue

            # if "20141123_153324_15_164" not in img_file[0]:
            #     continue

            # image_pil, image = load_image(sample_img[0].astype(dtype=np.uint8))

            # boxes_filt, pred_phrases = get_grounding_output(
            #     model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only
            # )
            # import pdb; pdb.set_trace()
            detections_all = None

            # #Multi class prediction
            # detections = model.predict_with_classes(
            #     image=img[0].astype(dtype=np.uint8),
            #     classes=CLASSES,
            #     box_threshold=box_threshold,
            #     text_threshold=text_threshold
            # )
            # # Filter detections
            # class_ids = detections.class_id
            # class_none_mask = class_ids!=None
            # detections.xyxy = detections.xyxy[class_none_mask, :]
            # detections.class_id = class_ids[class_none_mask]
            # detections.confidence = detections.confidence[class_none_mask]

            # # fuse detections
            # if detections_all is None:
            #     detections_all = detections
            # else:
            #     if len(detections.confidence)==0:
            #         continue
            #     try:
            #         detections_all.xyxy = np.vstack( (detections_all.xyxy, detections.xyxy))
            #         detections_all.class_id = np.concatenate( (detections_all.class_id, detections.class_id))
            #         detections_all.confidence = np.concatenate( (detections_all.confidence, detections.confidence) )
            #     except Exception as e:
            #         print("error occured")
            #         import pdb; pdb.set_trace()
            #         print("next")

            # Single class prediction
            for objid, objcls in enumerate(CLASSES):
                detections = model.predict_with_classes(
                    image=img[0].astype(dtype=np.uint8),
                    classes=[objcls],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                # Filter detections
                class_ids = np.array([objid]*len(detections.class_id))
                class_none_mask = class_ids!=None
                detections.xyxy = detections.xyxy[class_none_mask, :]
                detections.class_id = class_ids[class_none_mask]
                detections.confidence = detections.confidence[class_none_mask]

                # fuse detections
                if detections_all is None:
                    detections_all = detections
                else:
                    if len(detections.confidence)==0:
                        continue
                    try:
                        detections_all.xyxy = np.vstack( (detections_all.xyxy, detections.xyxy))
                        detections_all.class_id = np.concatenate( (detections_all.class_id, detections.class_id))
                        detections_all.confidence = np.concatenate( (detections_all.confidence, detections.confidence) )
                    except Exception as e:
                        print("error occured")
                        import pdb; pdb.set_trace()
                        print("next")

            if detections_all is not None:
                print("detections all", detections_all)
                try:
                    detections_all.class_id = detections_all.class_id.astype(np.uint8)
                except Exception as e:
                    import pdb; pdb.set_trace()
                detections_all_dict = {
                    "xyxy": detections_all.xyxy,
                    "class_id": detections_all.class_id,
                    "confidence": detections_all.confidence
                }
                pickle_file_list = img_file[0].split('/')
                action = pickle_file_list[-3]
                avi_name = pickle_file_list[-2]
                filename = pickle_file_list[-1].replace(".jpg", ".pkl")
                pickle_file_list.insert(-3, "labels")
                pickle_file_dir = "/".join(pickle_file_list[:-1])
                if not os.path.exists(pickle_file_dir):
                    print(f'Creating labels directory {pickle_file_dir}')
                    os.makedirs(pickle_file_dir)
            else:
                print("No detections for image ", img_file[0])
    
            pickle_file_path = os.path.join(pickle_file_dir, filename)
            dump_dictionary_to_pickle(detections_all_dict, pickle_file_path)
            
            box_annotator = sv.BoxAnnotator()
            input_img = img[0][:,:,::-1] # Flip BGR to RGB before annotating
            annotated_image = box_annotator.annotate(scene=input_img.astype(dtype=np.uint8), detections=detections_all)

            annotated_pil_img = Image.fromarray(annotated_image, "RGB")

            filename_only = filename.replace('.pkl', '.jpg')
            output_img_dir = os.path.join(output_dir, action, avi_name)
            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
           
            output_img_path = os.path.join(output_img_dir, filename_only)
            annotated_pil_img.save(output_img_path)
            annotated_pil_img.save(os.path.join(output_dir,"pred.jpg"))



if __name__ == "__main__":
    main()