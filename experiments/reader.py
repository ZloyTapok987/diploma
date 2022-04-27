from scipy.io import loadmat
from bert_serving.client import BertClient

NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

def mnistNums():
    arr = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    bc = BertClient()
    for num in arr:
        embed = bc.encode([num])
        embed.tofile("mnist_text/" + num + ".csv", sep=',', format='%10.5f')


mnistNums()


#image_labels = loadmat('imagelabels.mat')['labels'][0]
#bc = BertClient()
#embed = bc.encode(["cat"])
#embed.tofile("cat.csv", sep=',', format='%10.5f')
# get the embedding
##count = 1
#for label in image_labels:
#    print(NAMES[label - 1])
   # embedding = bc.encode([NAMES[label - 1]])
   # embedding.tofile('input_encoded_labels/image_{0}.csv'.format(count), sep=',', format='%10.5f')
#    count = count + 1

#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
#pip install --ignore-installed --upgrade $TF_BINARY_URL

#then bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len 50