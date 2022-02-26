import textualise
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
import pandas as pd
import torch
import model

"""
    This is a Language Modelling Task, using Shakespearing English
    In order to see how BERT works, what its mechanisms and designs are
    how the Attention Layers, The Masking, and The Encoding Layers work,
    why they work the way they work, and how I can use it later in other
    projects and research.

    It is meant to be used to understand the structure, decisions, design
    and mathematics behind it.
"""

def create_bert_vocab():
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
    tokenizer.train(['shakespeare.txt'],
                    vocab_size =12000,
                    show_progress = True,
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                    limit_alphabet=1000,
                    wordpieces_prefix="##"
    )
    tokenizer.save('shakespeare_bert_tokenizer.json', pretty=True)


# TODO: randomly pad up to 15% of the NON-MASKED tensor values
#       1. get the non-zero size (e.g., the unmasked and unpadded ones)
#       2. pick up to 15% of their indices
#       3. replace those indices with Zeros
#
def random_mask(arg):
    #non_zeros = 
    pass


def load_tensors(df: pd.DataFrame, tokenizer):
    """Load the Entire DataSet from Strings, into:
        - input ids (token ids)
        - segment ids (used for question - answering) are loaded but not used
        - attention mask

    Then move that data from a List, into a GPU Tensor, and allocate to
    a PyTorch DataLoader.

    **NOTE** Huggingface returns `None` for padding that doesn't below to any
    sentence, and then indexes sentences for question-answering.

    Since I am not doing this, here (I'm only modelling the domain) I don't need
    to pass segment indices, so I manually pass 1 for everything of attention
    and 0 for all else
    """
    data  = []
    batch = tokenizer.encode_batch(df['text'].tolist())
    for encoded in batch:
        #print(f"Tokens: {encoded.tokens}")
        #print(f"Token IDS (Input ID): {encoded.ids}")
        #print(f"Sequence IDS: {encoded.sequence_ids}")
        #print(f"Attention Mask: {encoded.attention_mask}")
        #print(f"Word IDS {encoded.word_ids}")
        #print(f"Type IDS {encoded.type_ids}")

        tok_ids = torch.ShortTensor(encoded.ids)
        seg_ids = encoded.sequence_ids
        seg_ids = [1 if x == 0 else 0 for x in seg_ids]
        seg_ids = torch.ShortTensor(seg_ids)
        msk_ids = torch.BoolTensor(encoded.attention_mask)
        t_row   = torch.stack((tok_ids, seg_ids, msk_ids))
        data.append(t_row)
    return data


def datasplit(df: pd.DataFrame) -> tuple:
    df_train = df.sample(frac = 0.70)
    df_test  = df.drop(df_train.index)
    return df_train, df_test


def training(vocab_size, tokenizer, data):
    bert = model.BERT(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size = 32,
                                         shuffle = True,
                                         pin_memory = False)

    for epoch in range(100):
        optimizer.zero_grad()
        for batch_ndx, sample in enumerate(loader):
            # TODO: randomly mask the 1st tensor which contains the token ids
            #       then upload to GPU and propagate
            #       
            #       See the rest of the code online
            #       and implement a similar approach, using tensorboard to monitor
            #       and track the experiments
            #
            print(batch_ndx, sample)


if __name__ == "__main__":
    """
        One problem we'll have with Shakespeare is that the longest
        dialogue is actually 537 words long, whereas the hard limit
        that this BERT model has is 512 words long. Clearly we can create
        a Model that is larger (1024 words long) but that means much more time
        spent training.

        Below we load the data from text into a DataFrame, then split it
        to 70% Train, 30% Test
    """
    df = textualise.DialogueSplitter("shakespeare.txt")()
    train_df, test_df = datasplit(df)

    """
        For BERT Tokenization I am using the Huggingface Bert Tokenizer.
        I've trained a custom one, but we can also use a pre-trained one.
        For more info, see:
            https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

    From Huggingface we note that:
        - IDS (or Input IDS in latter version) is the actual Token vector.
        - Word IDS mean **word position** IDS e.g., the position of the token in the sentence
        - Type IDS mean **segment** IDS 
        - Attention Mask defines the Unmakes (content) of the Tensor, whereas
          Masked is used to specify padding which isn't useful

    Below we load a saved tokenizer; if we need to retrain the tokenizer then
    uncomment the line `create_bert_vocab`
    """
    #create_bert_vocab()
    tokenizer = Tokenizer.from_file('shakespeare_bert_tokenizer.json')
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    max_length = 512
    tokenizer.enable_padding(length=512)
    tokenizer.enable_truncation(max_length=512)
    vocab_size = len(tokenizer.get_vocab())

    """
    We now iterate all the train data (DataFrame) encoding it all at once
    and populating the GPU memory with tensors which contain the token ids, segment ids and mask ids.
    """
    train_data = load_tensors(train_df, tokenizer)
    training(vocab_size, tokenizer, train_data)
