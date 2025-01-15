'''Pipe ZED detections into json files for COCO detections.'''

import pyzed.sl as sl
import cv2
import numpy as np
import json

NUM_FRAMES = 500

def main():
    '''Init, close, and capture data from ZED camera.'''
    #with open("body_tracking_outputCLASS3.txt", "w") as file:

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    body_params = sl.BodyTrackingParameters()
    # Different model can be chosen, optimizing the runtime or the accuracy
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    # Optimize the person joints position, requires more computations
    body_params.enable_body_fitting = True

    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

        print("Body tracking: Loading Module...")

        err = zed.enable_body_tracking(body_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Enable Body Tracking : "+repr(err)+". Exit program.")
            zed.close()
            exit()
        bodies = sl.Bodies()
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        # For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
        # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
        body_runtime_param.detection_confidence_threshold = 40
        
    #Capture the data.
    for _ in range(NUM_FRAMES): #setup num frames
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            if bodies.is_new:
                body_array = bodies.body_list
                print(str(len(body_array)) + " Person(s) detected\n")

                
                for idx, body in enumerate(body_array):
                    #!skip if 1) not OK status or 2) conf < certain threshold?? not sure on this for now.
                    print(f"Person {idx + 1} attributes:")
                    print(" Confidence (" + str(int(body.confidence)) + "/100)")
                    #TODO: if confidence??
                    
                    if body_params.enable_tracking:
                        print(" Tracking ID: " + str(int(body.id)) + " tracking state: " + repr(body.tracking_state) + " / " + repr(body.action_state))
                        if repr(body.tracking_state) != "OK":
                            continue
                    
                    #!process bounding boxes, keypoints, conf score, area, width, height
                    #can transport and streamline these processes from the ipynb file
                    # for it in body.keypoint_2d:
                        #     print("    " + str(it))
                        #     file.write("    " + str(it) + "\n")

                    #!Flatten keypoints
                        # Flattened keypoints array with visibility flag `2` and removing keypoint at index 1
                        flattened_keypoints = []
                        for idx, kp in enumerate(body.keypoint_2d):
                            if idx == 1:  # Skip the keypoint at index 1
                                continue
                            flattened_keypoints.extend([kp[0], kp[1], 2])  # Append x, y, and visibility

                        # Print the flattened array
                        print(flattened_keypoints)

                    #TODO: put right bodies in corresponding JSON files, right now: 
                    #1. First Person is GT_aann
                    #2. Everyone else is a detection
                    if idx == 0:
                        #2D keypoints.
                        with open("gtgtgt_ann.json", "r") as file:
                            data = json.load(file)

                        # Example modifications
                        # 1. Change the width of the first image
                        data["images"][0]["width"] = 1920

                        # 2. Update the keypoints of the first annotation
                        data["annotations"][0]["keypoints"] = [800.0, 300.0, 2, 750.0, 320.0, 2, ...]

                        # 3. Modify the description in the info section
                        data["info"]["description"] = "Updated 2D Keypoint Annotations"

                        # Save the updated JSON back to the file
                        with open("ground_truth.json", "w") as file:
                            json.dump(data, file, indent=4)
                    else: #otherwise we open a different file

                        file.write("\n")
                        print("\n")

    # Close the camera
    zed.disable_body_tracking()
    zed.close()


if __name__ == "__main__":
    main()
