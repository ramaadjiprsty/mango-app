import gdown
url = "https://drive.google.com/uc?id=1oUiG-b21tOpUAAAsWuq0ElDxPF_SPlSA"
output = "model/mango_leaf_disease_simple.h5"
gdown.download(url, output, quiet=False)
