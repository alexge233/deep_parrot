import textualise
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import model
import random

torch.set_printoptions(linewidth=200)

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


def random_mask(input_ids, masks):
    """Randomly pad **UP TO 15%** of the NON-MASKED tensor values.
    We use Python's PRNG to chose how many NON-MASKED values we'll pad
    and then we use Python's PRNG to chose which indices to randomly MASK.

    I tried with numpy but wasn't happy about it, so using Python for this.
    Should be done in-memory so is most likely performant.

    We use the Mask Tensor instead of the Raw values (which is what is happening now)
    """
    full_size = input_ids.size()[0]
    non_zeros = torch.nonzero(input_ids).squeeze()
    max_idx   = non_zeros.size()[0] -1
    r_num     = random.randint(0, int(max_idx * 0.15))
    r_indices = [random.randint(0,max_idx) for x in range(0, r_num)]

    if len(r_indices) > 0 and masks.size()[0] == 512:
        return masks.index_fill(0, torch.tensor(r_indices), 0)
    else:
        return masks


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
        tok_ids = torch.IntTensor(encoded.ids)
        seg_ids = encoded.sequence_ids
        seg_ids = [1 if x == 0 else 0 for x in seg_ids]
        seg_ids = torch.IntTensor(seg_ids)
        msk_ids = torch.BoolTensor(encoded.attention_mask)
        t_row   = torch.stack((tok_ids, seg_ids, msk_ids))
        data.append(t_row)
    return data


def datasplit(df: pd.DataFrame) -> tuple:
    df_train = df.sample(frac = 0.70)
    df_test  = df.drop(df_train.index)
    return df_train, df_test


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

    """
    Setup Model, Criterion, Optimizer, Data Loader.
    Start putting all data on GPU now, and note that I've only got a GTX1080Ti
    so, my hard GPU limit is 12Gb which won't run with a batch of 16 or 32.
    Batch of 8 runs, might be able to squeeze a few more but that's it.
    """
    network = model.BERT(vocab_size).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    loader = torch.utils.data.DataLoader(train_data,
                                         batch_size = 24,
                                         shuffle = True,
                                         pin_memory = False)

    for epoch in range(40):
        optimizer.zero_grad()
        losses = []
        for batch_ndx, samples in enumerate(loader):
            """
            Sample is a 3-dim tensor
            1st Dim is the Batch

            2nd Dim is the Sample Tensor

            3rd Dim the the Tensor values:
                - input ids
                - segment ids
                - masks


            We need to manually randomly mask the token input ids and then
            upload all the data to CUDA memory.
            """
            input_ds  = torch.zeros(samples.size()[0], samples.size()[2], dtype=torch.int)
            segments  = torch.zeros(samples.size()[0], samples.size()[2], dtype=torch.int)
            masks_pos = torch.zeros(samples.size()[0], samples.size()[2], dtype=torch.long)
            masked_ts = torch.zeros(samples.size()[0], samples.size()[2], dtype=torch.long)

            for i, x in enumerate(samples):
                masks_pos[i] = random_mask(x[0], x[2])
                input_ds[i]  = x[0]
                segments[i]  = x[1]
                masked_ts[i] = x[2]

            """
            Propagate, then calculate loss and finally backpropagate.
            We do an update of the loss per batch but we keep a record per epoch.
            """
            logits_lm, _ = network(input_ds.to('cuda'), segments.to('cuda'), masks_pos.to('cuda'))
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_ts.to('cuda'))
            loss_lm.backward()
            losses.append(loss_lm.item())

        loss = sum(losses) / len(losses)
        losses.clear()

        print('Epoch:', '%04d' % (epoch + 1), 'CE Loss =', '{:.6f}'.format(loss))
        optimizer.step()

    # finally save it. We;ll evaluate it manually later
    torch.save(network.state_dict(), "bert_model")

