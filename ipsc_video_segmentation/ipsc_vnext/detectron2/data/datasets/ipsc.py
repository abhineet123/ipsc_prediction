def register_as_coco():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("ipsc-all_frames_roi_g2_0_38-train", {},
                            "~/data/ipsc/well3/all_frames_roi/all_frames_roi_g2_0_38-train.json",
                            "~/data//ipsc/well3/all_frames_roi")

    register_coco_instances("ipsc-all_frames_roi_g2_0_38-val", {},
                            "~/data/ipsc/well3/all_frames_roi/all_frames_roi_g2_0_38-val.json",
                            "/data//ipsc/well3/all_frames_roi")

    register_coco_instances("ipsc-all_frames_roi_g2_39_53", {},
                            "~/data/ipsc/well3/all_frames_roi/all_frames_roi_g2_39_53.json",
                            "~/data//ipsc/well3/all_frames_roi")

