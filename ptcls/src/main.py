import argparse
from pathlib import Path
import torch
from ptcls.config import Config

from ptcls.dataloader.w2ner import DataManger
from ptcls.models.w2ner import Model
from ptcls.trainer.w2ner import Trainer


root = Path("/home/hsc/ner")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_path", type=str,
    default=root / "pretrained_model/chinese_wwm_ext_L-12_H-768_A-12/")
parser.add_argument(
    "--dataset_path", type=str,
    default=root / "dataset/cluener/w2ner/")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
else:
    device = torch.device("cpu")

print("Device", device)

# random.seed(config.seed)
# np.random.seed(config.seed)
# torch.manual_seed(config.seed)
# torch.cuda.manual_seed(config.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

print("Loading Data")
pretrained_path = Path(args.pretrained_path)
dataset_path = Path(args.dataset_path)

dm = DataManger(dataset_path / "labels.txt", pretrained_path)
config = Config("w2ner")

train_loader = dm.load(
    dataset_path / "train.json", config.batch_size, True, True)
dev_loader = dm.load(
    dataset_path / "dev.json", config.batch_size, False, False)

test_file = dataset_path / "test.json"
if test_file.exists():
    test_loader, test_data = dm.load(
        dataset_path / "test.json", config.batch_size, False, False, True)
else:
    test_loader = None
    test_data = []

if args.epochs:
    config.epochs = args.epochs
if args.batch_size:
    config.batch_size = args.batch_size
config.label_num = len(dm.label2id)
config.updates_total = len(train_loader) * config.epochs

print("Config: ", config)
print(f"Batch Num: {len(train_loader)}")
print("Loading Model")
# model = Model(config, pretrained_path)
print("Loading Trainer")
# trainer = Trainer(model, config, device)
print("Training")
# trainer.train(train_loader, dev_loader, test_loader)

if test_loader is not None:
    print("Loading trained model")
    # trainer.load(config.out_path)
    print("Testing")
    # trainer.predict("Final", test_loader, test_data, dm.id2label)
