import cv2
import numpy as np
import torch
import os
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
from collections import Counter

SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

BLOCK_LIST_CLASSES = [
    "ceiling", "floor", "chair", "table", "person", "sofa", "bed", "shelf", "cabinet",
    "potted_plant", "plant", "monitor", "screen", "computer", "clock"
]

ALLOW_LIST_CLASSES = ["wall", "column", "panel", "building"]
BLOCK_LIST_IDS = []
ALLOW_LIST_IDS = []
MIN_MASK_AREA_RATIO = 0.02 
MAX_IMAGE_WIDTH = 1280

def run_pipeline(args):
    global ALLOW_LIST_IDS, BLOCK_LIST_IDS
    print("...")
    if not os.path.exists(args.room):
        print(f"Error: Room image not found: {args.room}")
        return
    if not os.path.exists(SAM_MODEL_PATH):
        print(f"Error: SAM model not found: {SAM_MODEL_PATH}")
        print("Please download 'sam_vit_b_01ec64.pth' and place it in the same directory.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading SegFormer model (for class identification)...")
    try:
        processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
        segformer_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(device)
    except Exception as e:
        print(f"Error loading SegFormer: {e}. Check internet connection and 'transformers' library.")
        return
    
    id2label = segformer_model.config.id2label
    label2id = segformer_model.config.label2id
    ALLOW_LIST_IDS = [label2id[name] for name in ALLOW_LIST_CLASSES if name in label2id]
    BLOCK_LIST_IDS = [label2id[name] for name in BLOCK_LIST_CLASSES if name in label2id]
    print(f"Allowing classes: {ALLOW_LIST_CLASSES} (IDs: {ALLOW_LIST_IDS})")
    print(f"Blocking classes: {BLOCK_LIST_CLASSES} (IDs: {BLOCK_LIST_IDS})")


    print("Loading SAM model (for precise masks)...")
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=12,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.90,
        )
    except Exception as e:
        print(f"Error loading SAM: {e}. Check 'segment_anything' library.")
        return
    print("Models loaded successfully.")

    room_pil = Image.open(args.room).convert("RGB")
    room_np = np.array(room_pil)
    room_h, room_w = room_np.shape[:2]
    room_total_pixels = room_w * room_h
    MIN_MASK_AREA = room_total_pixels * MIN_MASK_AREA_RATIO

    if room_w > MAX_IMAGE_WIDTH:
        new_w = MAX_IMAGE_WIDTH
        new_h = int(room_h * (MAX_IMAGE_WIDTH / room_w))
        print(f"Resizing image from {room_w}x{room_h} to {new_w}x{new_h}")
        room_np = cv2.resize(room_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        room_h, room_w = new_h, new_w

    print("Running SegFormer to get class labels...")
    inputs = processor(images=room_np, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer_model(**inputs)

    logits = F.interpolate(outputs.logits, size=(room_h, room_w), mode="bilinear", align_corners=False)
    predicted_labels_map = logits.argmax(1).squeeze().cpu().numpy()

    print("Running SAM to get precise masks...")
    all_masks = mask_generator.generate(room_np)
    all_masks = sorted(all_masks, key=lambda x: x["area"], reverse=True)

    print(f"Found {len(all_masks)} potential masks. Filtering...")
    final_combined_mask = np.zeros((room_h, room_w), dtype=np.uint8)
    detected_count = 0

    for i, mask_data in enumerate(all_masks):
        area = mask_data["area"]
        if area < MIN_MASK_AREA:
            continue

        mask_bool = mask_data["segmentation"]
        labels_in_mask = predicted_labels_map[mask_bool]
        
        if labels_in_mask.size == 0:
            continue
        
        top_labels_with_counts = Counter(labels_in_mask).most_common(5)
        top_label_ids = [item[0] for item in top_labels_with_counts]
        top_label_names = [id2label.get(label_id, "unknown") for label_id in top_label_ids]

        is_blocked = False
        is_allowed = False

        if any(label_id in BLOCK_LIST_IDS for label_id in top_label_ids):
            is_blocked = True
        if any(label_id in ALLOW_LIST_IDS for label_id in top_label_ids):
            is_allowed = True
        
        if is_allowed and not is_blocked:
            print(f"Accepting mask {i} (Area: {area}, Top Labels: {top_label_names})")
            final_combined_mask[mask_bool] = 255 
            detected_count += 1
            if args.single_wall:
                print("Single wall mode: stopping after first match.")
                break
        else:
            if is_blocked:
                 print(f"Rejecting mask {i} (Area: {area}, Top Labels: {top_label_names}) -> Contains blocked class.")

    # Save Final Mask
    if detected_count > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        final_combined_mask = cv2.morphologyEx(final_combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        cv2.imwrite(args.output_mask, final_combined_mask)
        print(f"Successfully found {detected_count} wall regions and saved mask to {args.output_mask}")
    else:
        print("No suitable wall regions were found after filtering.")
        cv2.imwrite(args.output_mask, final_combined_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Wall Finder Pipeline")    
    parser.add_argument("--room", type=str, required=True, help="Path to the room image")
    parser.add_argument("--output_mask", type=str, required=True, help="Path to save the final B&W mask")
    parser.add_argument(
        "--single_wall",
        action="store_true",
        help="If set, finds only the largest detected wall"
    )
    args = parser.parse_args()
    run_pipeline(args)
