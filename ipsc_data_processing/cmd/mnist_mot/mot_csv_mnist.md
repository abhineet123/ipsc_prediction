<!-- MarkdownTOC -->

- [mot_csv_to_xml_coco](#mot_csv_to_xml_coco_)
    - [640-1       @ mot_csv_to_xml_coco](#640_1___mot_csv_to_xml_coc_o_)
    - [640-3       @ mot_csv_to_xml_coco](#640_3___mot_csv_to_xml_coc_o_)
    - [640-5       @ mot_csv_to_xml_coco](#640_5___mot_csv_to_xml_coc_o_)
    - [512       @ mot_csv_to_xml_coco](#512___mot_csv_to_xml_coc_o_)
        - [n-1       @ 512/mot_csv_to_xml_coco](#n_1___512_mot_csv_to_xml_coc_o_)
        - [n-3       @ 512/mot_csv_to_xml_coco](#n_3___512_mot_csv_to_xml_coc_o_)
- [mot_to_csv](#mot_to_cs_v_)

<!-- /MarkdownTOC -->

<a id="mot_csv_to_xml_coco_"></a>
# mot_csv_to_xml_coco
<a id="640_1___mot_csv_to_xml_coc_o_"></a>
## 640-1       @ mot_csv_to_xml_coco-->mot_csv_mnist
python mot_csv_to_xml_coco.py cfg=mnist:640-1:12_1000:train:zip:csv
<a id="640_3___mot_csv_to_xml_coc_o_"></a>
## 640-3       @ mot_csv_to_xml_coco-->mot_csv_mnist
python mot_csv_to_xml_coco.py cfg=mnist:640-3:12_1000:train:zip:csv
<a id="640_5___mot_csv_to_xml_coc_o_"></a>
## 640-5       @ mot_csv_to_xml_coco-->mot_csv_mnist
python mot_csv_to_xml_coco.py cfg=mnist:640-5:12_1000:train:zip:csv

<a id="512___mot_csv_to_xml_coc_o_"></a>
## 512       @ mot_csv_to_xml_coco-->mot_csv_mnist
<a id="n_1___512_mot_csv_to_xml_coc_o_"></a>
### n-1       @ 512/mot_csv_to_xml_coco-->mot_csv_mnist
python mot_csv_to_xml_coco.py root_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations ignore_invalid=0 show_img=0 sample=30 save_img_seq=1 mode=2 start_id=0 end_id=999 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt json_fname=mnist_mot_rgb_512_1k_9600_1_var-train.json out_root_suffix=json

python mot_csv_to_xml_coco.py root_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations ignore_invalid=0 show_img=0 save_img_seq=1 mode=2 start_id=1000 end_id=-1 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt json_fname=mnist_mot_rgb_512_1k_9600_1_var-val.json out_root_suffix=json sample=300

python mot_csv_to_xml_coco.py root_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations ignore_invalid=0 show_img=0 save_img_seq=1 mode=2 start_id=1000 end_id=-1 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt json_fname=mnist_mot_rgb_512_1k_9600_1_var-test.json out_root_suffix=json sample=30

python mot_csv_to_xml_coco.py root_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations ignore_invalid=0 show_img=0 save_img_seq=0 mode=2 start_id=1000 end_id=1001 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt json_fname=mnist_mot_rgb_512_1k_9600_1_var-test_dummy.json out_root_suffix=json sample=300

python mot_csv_to_xml_coco.py root_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations ignore_invalid=0 show_img=0 save_img_seq=0 mode=2 start_id=1000 end_id=1009 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt json_fname=mnist_mot_rgb_512_1k_9600_1_var-test_1_10.json out_root_suffix=json sample=30

<a id="n_3___512_mot_csv_to_xml_coc_o_"></a>
### n-3       @ 512/mot_csv_to_xml_coco-->mot_csv_mnist
python mot_csv_to_xml_coco.py root_dir=/data/MNIST_MOT_RGB_512x512_3_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_3_var.txt data_type=annotations mode=1 ignore_invalid=0 start_id=0 label=digit show_img=0 sample=30 save_img_seq=1 mode=2 start_id=0 end_id=-1 vid_ext=mp4 class_names_path=lists/classes/mnist_mot.txt

<a id="mot_to_cs_v_"></a>
# mot_to_csv
python mot_to_csv.py root_dir=/data/MNIST_MOT_RGB_512x512_1_1000_9600_var img_dir=Images list_file_name=lists/mnist_mot_rgb_512_1k_9600_1_var.txt data_type=annotations mode=2 ignore_invalid=0 start_id=0 end_id=-1 show_img=0 class_names_path=lists/classes/mnist_mot.txt out_root_suffix=json sample=30 save_video=0 vid_ext=mp4
