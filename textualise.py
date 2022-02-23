"""
    Handle Text in a semi-professional manner
"""
import re
from dataclasses import dataclass
import pandas as pd

@dataclass
class DialogueSplitter:
    text_file : str


    def __speaker__(self, arg: str) -> str:
        res = re.search(r'([A-Za-z\ ]{2,})\:\n', arg)
        if res:
            return res.group(0)
        else:
            return None


    def __call__(self):

        self.index = {}

        with open(self.text_file) as f:
            while True:
                line = f.readline()
                if not line:
                    break

                row  = []
                speaker = self.__speaker__(line)
                if speaker:
                    print(f"SPEAKER {speaker}")
                else:
                    if line == "\n":
                        print("end of dialogue")
                        # TODO: pop row into index
                    else:
                        row.append(line.strip())
