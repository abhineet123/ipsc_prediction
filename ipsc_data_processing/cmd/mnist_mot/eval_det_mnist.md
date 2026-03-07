# n-1
## r50       @ n-1-->eval_det_mnist
### pool       @ r50/n-1-->eval_det_mnist
python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_2/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-pool_2 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_4/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-pool_4 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_8/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-pool_8 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_16/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-pool_16 iw=0 det_nms=0 n_proc=1 

### set_zero_1_2_3       @ r50/n-1-->eval_det_mnist
python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_set_zero_1_2_3/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-set_zero_1_2_3 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_2_set_zero_1_2_3/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-set_zero_1_2_3-pool_2 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_4_set_zero_1_2_3/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-set_zero_1_2_3-pool_4 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_8_set_zero_1_2_3/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-set_zero_1_2_3-pool_8 iw=0 det_nms=0 n_proc=1 

python3 eval_det.py cfg=mnist:n1 img_paths=lists/mnist_mot_rgb_512_1k_9600_1_var_test_1_10.txt det_paths=log/swi/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn/best_bbox_mAP_on_test_1_10_pool_16_set_zero_1_2_3/csv gt_csv_name=annotations.csv save_suffix=mnist_mot-r50-test_1_10-set_zero_1_2_3-pool_16 iw=0 det_nms=0 n_proc=1 
