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

        frames = []

        with open(self.text_file) as f:

            text = ""
            role = None

            while True:
                line = f.readline()
                if not line:
                    break

                speaker = self.__speaker__(line)
                if speaker:
                    role = speaker.strip()

                else:
                    if line == "\n":
                        text = re.sub("[.,!?\\-]", '', text.lower())
                        frames.append([role, text.strip()])
                        role = None
                        text = ""

                    else:
                        text += line.strip() + " "

        return pd.DataFrame(data = frames, columns = ['speaker', 'text'])

