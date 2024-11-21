# '''from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# # Initialize COCO ground truth api for keypoint annotations
# ann_file = 'path/to/person_keypoints_val2017.json'  # Replace with path to the downloaded file
# coco_gt = COCO(ann_file)


# # Assuming 'detections' is your list of predicted keypoints for each person in the image
# coco_dt = coco_gt.loadRes(detections)
# coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()'''

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import json

# # Load ground truth and detection annotations
# coco_gt = COCO("path/to/ground_truth_annotations.json")
# coco_dt = coco_gt.loadRes("path/to/your_annotations.json")

# # Initialize COCO evaluation for 'keypoints' or 'bbox' or 'segm' based on what you want to evaluate
# coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")

# # Specify the image IDs you want to evaluate (or use all available in the ground truth)
# image_ids = sorted(coco_gt.getImgIds())

# # Run evaluation
# coco_eval.params.imgIds = image_ids
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Paths to your files
ground_truth_file = "gt_ann.json"
detection_file = "detect_ann.json"

# Initialize COCO ground truth and detection
coco_gt = COCO(ground_truth_file)
coco_dt = coco_gt.loadRes(detection_file)
# Check lengths of keypoints in ground truth
for annotation in coco_gt.dataset['annotations']:
    print(f"Ground Truth Keypoints Length: {len(annotation['keypoints'])}")

# Check lengths of keypoints in detections
for detection in coco_dt.anns.values():
    print(f"Detection Keypoints Length: {len(detection['keypoints'])}")

# Check for mismatched keypoints
for detection in coco_dt.anns.values():
    if len(detection['keypoints']) != 51:  # 17 keypoints * 3 (x, y, visibility)
        print(f"Detection ID {detection['id']} has {len(detection['keypoints'])} keypoints.")

 
# Initialize COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")

# Set maxDets to 50
coco_eval.params.maxDets = [50]  # Set maxDets to 50

# Print to confirm
print("Max Dets:", coco_eval.params.maxDets)

# Specify the image IDs you want to evaluate (or use all available in the ground truth)
image_ids = sorted(coco_gt.getImgIds())

# Run evaluation
coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
''' 
# Initialize COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')

# Set maxDets to 50
coco_eval.params.maxDets = [5]  # You can also set it as [50, 100, 200] for multiple thresholds
#! if your model only detects 10 objects but maxDets is set to 50, the evaluation may not yield valid results.

# Specify the image IDs you want to evaluate (or use all available in the ground truth)
#image_ids = sorted(coco_gt.getImgIds())

# Run evaluation
#coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()''' 
 

