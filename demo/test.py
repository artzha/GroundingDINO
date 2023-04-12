import os
import pdb
import torch
from torch.utils.data import DataLoader
from coda_dataset import CODataset
from groundingdino.util.inference import load_model, load_image, predict, Model
from tqdm import tqdm
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap

import fiftyone as fo

CLASS_IDS = [str(i) for i in range(0, 55)]

CLASSES =  [  
    # Dynamic Classes
    "Car"                   ,
    "Pedestrian"            ,
    "Bike"                  ,
    # # # Newly added
    "Tree Trunk"            ,
    "Pole"                  ,
    "Sign"                  ,
    "Chair"                 ,
    "Table"
    # "Motorcycle"            ,
    # "Golf Cart"             ,
    # "Truck"                 ,
    # "Scooter"               ,
    # # Static Classes
    # "Tree Trunk"            ,
    # "Traffic Sign"          ,
    # "Canopy"                ,
    # "Traffic Light"         ,
    # "Bike Rack"             ,
    # "Bollard"               ,
    # "Construction Barrier"  ,
    # "Parking Kiosk"         ,
    # "Mailbox"               ,
    # "Fire Hydrant"          ,
    # # Static Class Mixed
    # "Freestanding Plant"    ,
    # "Pole"                  ,
    # "Informational Sign"    ,
    # "Door"                  ,
    # "Fence"                 ,
    # "Railing"               ,
    # "Cone"                  ,
    # "Chair"                 ,
    # "Bench"                 ,
    # "Table"                 ,
    # "Trash Can"             ,
    # "Newspaper Dispenser"   ,
    # # Static Classes Indoor
    # "Room Label"            ,
    # "Stanchion"             ,
    # "Sanitizer Dispenser"   ,
    # "Condiment Dispenser"   ,
    # "Vending Machine"       ,
    # "Emergency Aid Kit"     ,
    # "Fire Extinguisher"     ,
    # "Computer"              ,
    # "Television"            ,
    # "Other"                 ,
    # "Horse"                 ,
    # # New Classes
    # "Pickup Truck"          ,
    # "Delivery Truck"        ,
    # "Service Vehicle"       ,
    # "Utility Vehicle"       ,
    # "Fire Alarm"            ,
    # "ATM"                   ,
    # "Cart"                  ,
    # "Couch"                 ,
    # "Traffic Arm"           ,
    # "Wall Sign"             ,
    # "Floor Sign"            ,
    # "Door Switch"           ,
    # "Emergency Phone"       ,
    # "Dumpster"              ,
    # "Vacuum Cleaner"        
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

def main(
    root_dir: str = '/robodata/arthurz/Benchmarks/unsupda/ST3D/data/coda',
    text_prompt: str =  "Car, Pedestrian, Bike, Motorcycle, Golf Cart, Truck, Scooter, Tree Trunk, Traffic Sign, Canopy, Traffic Light, Bike Rack, Bollard, Construction Barrier, Parking Kiosk, Mailbox, Fire Hydrant, Freestanding Plant, Pole, Informational Sign, Door, Fence, Railing, Cone, Chair, Bench, Table, Trash Can, Newspaper Dispenser, Room Label, Stanchion, Sanitizer Dispenser, Condiment Dispenser, Vending Machine, Emergency Aid Kit, Fire Extinguisher, Computer, Television, Other, Horse, Pickup Truck, Delivery Truck, Service Vehicle, Utility Vehicle, Fire Alarm, ATM, Cart, Couch, Traffic Arm, Wall Sign, Floor Sign, Door Switch, Emergency Phone, Dumpster, Vacuum Cleaner.",
    box_threshold: float = 0.3, 
    text_threshold: float = 0.20,
    export_dataset: bool = False,
    view_dataset: bool = False,
    export_annotated_images: bool = True,
    weights_path : str = "./groundingdino_swinb_cogcoor.pth",
    config_path: str = "./groundingdino/config/GroundingDINO_SwinB.cfg.py",
    subsample: int = None,
    cpu_only: bool = False,
    output_dir: str = "./outputs"
):
    global CLASSES
    CLASSES = [objcls.lower() for objcls in CLASSES]
    # model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)
    # load_model(config_path, weights_path)

    root_path = Path(root_dir)
    training_data = CODataset(CLASSES, root_path, 'training')
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, collate_fn=training_data.collate_fn, pin_memory=True)
    
    # Load GT dataset to fiftyone
    for _, _, sample_img_file, gt_label, gt_bbox in tqdm(train_dataloader):
        import pdb; pdb.set_trace()
        sample = fo.Sample(filepath=sample_img_file[0])
        detections = []
        
        for obj_idx, obj_label in enumerate(gt_label):
            objlabel_id = CLASSES.index(obj_label.lower())

            # Bounding box coordinates should be relative values
            # in [0, 1] in the following format:
            # [top-left-x, top-left-y, width, height]
            import pdb; pdb.set_trace()
            h, w = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
            bounding_box = [gt_bbox[0], gt_bbox[1], w, h]

            detections.append(
                fo.Detection(label=objlabel_id, bounding_box=bounding_box)
            )

        # Store detections in a field name of your choice
        sample["ground_truth"] = fo.Detections(detections=detections)

        samples.append(sample)

        import pdb; pdb.set_trace()

    for sample_idx, sample_img, sample_img_file, gt_label, gt_bbox in tqdm(train_dataloader):
        # image_pil, image = load_image(sample_img[0].astype(dtype=np.uint8))

        # boxes_filt, pred_phrases = get_grounding_output(
        #     model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only
        # )
        # import pdb; pdb.set_trace()
        detections_all = None
        for objid, objcls in enumerate(CLASSES):
            detections = model.predict_with_classes(
                image=sample_img[0].astype(dtype=np.uint8),
                classes=[objcls],
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            # fuse detections
            if detections_all is None:
                detections_all = detections
            else:
                if len(detections.confidence)==0:
                    continue
                    # import pdb; pdb.set_trace()

                try:
                    # print("objid ", objid)
                    detections_all.xyxy = np.vstack( (detections_all.xyxy, detections.xyxy))
                    detections_all.class_id = np.concatenate( (detections_all.class_id, np.array([objid]*len(detections.class_id)) ))
                    detections_all.confidence = np.concatenate( (detections_all.confidence, detections.confidence) )
                except Exception as e:
                    print("error occured")
                    import pdb; pdb.set_trace()
                    print("next")

        
        detections_all.class_id = detections_all.class_id.astype(np.uint8)
        box_annotator = sv.BoxAnnotator()
        # import pdb; pdb.set_trace()
        annotated_image = box_annotator.annotate(scene=sample_img[0].astype(dtype=np.uint8), detections=detections_all)
        # import pdb; pdb.set_trace()
        # # visualize pred
        # size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes_filt,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": pred_phrases,
        # }
        # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        annotated_pil_img = Image.fromarray(annotated_image, "RGB")
        annotated_pil_img.save(os.path.join(output_dir, "pred.jpg"))



if __name__ == "__main__":
    main()