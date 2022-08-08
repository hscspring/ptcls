import os
from pathlib import Path

import pnlp

root = Path(os.path.abspath(__file__)).parent


class Config:

    def __init__(self, model: str):
        fname = model + ".json"
        config_file = root / fname
        try:
            dct = pnlp.read_json(config_file)
        except Exception as e:
            msg = f"ptcls: config file {config_file} "
            msg += "is not a valid json file. "
            msg += f"Reason: {e}"
            raise ValueError(msg)
        for key, val in dct.items():
            setattr(self, key, val)

    def __repr__(self):
        dct = self.__dict__.items()
        return f"Config: {dct}"
