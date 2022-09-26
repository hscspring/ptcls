from typing import List, Dict, Union
import json
from pathlib import Path
import pnlp


def read_file_to_list_dict(inp_file: Union[Path, str]) -> List[Dict]:
    res = []
    for line in pnlp.read_lines(inp_file):
        item = json.loads(line.strip())
        res.append(item)
    return res


def write_list_dict_to_file(out_file: Union[Path, str], data: List[Dict]):
    fo = open(out_file, "w")
    for item in data:
        line = json.dumps(item, ensure_ascii=False)
        line += "\n"
        fo.write(line)


class ResumeZh:

    def __init__(self, path: Path):
        ...


class ClueNer:

    def __init__(self, path: Path):
        self.path = path

    def to_w2ner(self, file_name: str) -> List[Dict]:
        file_path = self.path / file_name
        lst = read_file_to_list_dict(file_path)
        res = []
        for im in lst:
            text = im["text"]
            label = im["label"]
            ent_res = []
            for tag, ent_dict in label.items():
                for val, locs in ent_dict.items():
                    for loc in locs:
                        idx = list(range(loc[0], loc[1] + 1))
                        v = {"index": idx, "type": tag, "value": val}
                        ent_res.append(v)
            new = {"sentence": list(text), "ner": ent_res, "word": []}
            res.append(new)
        out_root = self.path / "w2ner"
        pnlp.check_dir(out_root)
        pnlp.write_json(out_root / file_name, res)


if __name__ == "__main__":
    root = Path("/home/hsc/ner/dataset")
    cn = ClueNer(root / "cluener")
    cn.to_w2ner("train.json")
