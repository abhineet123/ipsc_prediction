<!-- MarkdownTOC -->

- [mnist-640-1       @ mnist_mot](#mnist_640_1___mnist_mo_t_)
    - [frame-0_1       @ mnist-640-1](#frame_0_1___mnist_640_1_)
    - [frame-2_3       @ mnist-640-1](#frame_2_3___mnist_640_1_)
    - [frame-4_5       @ mnist-640-1](#frame_4_5___mnist_640_1_)
- [mnist-640-3       @ mnist_mot](#mnist_640_3___mnist_mo_t_)
- [mnist-640-5       @ mnist_mot](#mnist_640_5___mnist_mo_t_)
- [mnist-512-3       @ mnist_mot](#mnist_512_3___mnist_mo_t_)

<!-- /MarkdownTOC -->

<a id="mnist_640_1___mnist_mo_t_"></a>
# mnist-640-1       @ mnist_mot-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-1:12_1000:train:gz
python xml_to_coco.py cfg=mnist:640-1:12_1000:test:gz
<a id="frame_0_1___mnist_640_1_"></a>
## frame-0_1       @ mnist-640-1-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-1:12_1000:train:frame-0_1:gz
<a id="frame_2_3___mnist_640_1_"></a>
## frame-2_3       @ mnist-640-1-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-1:12_1000:train:frame-2_3:gz
python xml_to_coco.py cfg=mnist:640-1:12_1000:test:frame-2_3:gz
<a id="frame_4_5___mnist_640_1_"></a>
## frame-4_5       @ mnist-640-1-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-1:12_1000:train:frame-4_5:gz
python xml_to_coco.py cfg=mnist:640-1:12_1000:test:frame-2_3:gz

<a id="mnist_640_3___mnist_mo_t_"></a>
# mnist-640-3       @ mnist_mot-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-3:12_1000:train:gz
python xml_to_coco.py cfg=mnist:640-3:12_1000:test:gz
<a id="mnist_640_5___mnist_mo_t_"></a>
# mnist-640-5       @ mnist_mot-->xml_to_coco_mnist
python xml_to_coco.py cfg=mnist:640-5:12_1000:train:gz
python xml_to_coco.py cfg=mnist:640-5:12_1000:test:gz

python xml_to_coco.py cfg=mnist:640-5:12_1000:test:seq-0_5:frame-0_5:gz

<a id="mnist_512_3___mnist_mo_t_"></a>
# mnist-512-3       @ mnist_mot-->xml_to_coco_mnist
python xml_to_coco.py root_dir=/data/mnist_mot/512_3_1000_9600_var/Images seq_paths=lists/mnist_mot_rgb_512_1k_9600_3_var.txt class_names_path=lists/classes/mnist_mot.txt output_json=mnist_mot_rgb_512_1k_9600_3_var.json ignore_invalid_label=0 val_ratio=0 start_id=0 end_id=999 skip_invalid=0 enable_masks=0
__val__
python xml_to_coco.py root_dir=/data/MNIST_MOT_RGB_512x512_3_1000_9600_var/Images seq_paths=lists/mnist_mot_rgb_512_1k_9600_3_var.txt class_names_path=lists/classes/mnist_mot.txt output_json=mnist_mot_rgb_512_1k_9600_3_var.json ignore_invalid_label=0 val_ratio=1 start_id=1000 end_id=-1 skip_invalid=0 enable_masks=0 samples_per_seq=0.01
