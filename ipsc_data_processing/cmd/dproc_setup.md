# virtualenv
mkvirtualenv -p python3.10 dproc
alias dproc='workon dproc'
alias ac='workon dproc'

## pycharm       @ virtualenv-->dproc_setup
Add interpreter > On WSL

# packages
python -m pip install -r dproc_requirements.txt

python -m pip install imagesize paramparse numpy opencv-python matplotlib pandas tqdm prettytable tabulate scikit-learn pycocotools pyperclip compress_json

sudo apt-get install python3.10-tk


