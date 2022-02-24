from tokenizers import BertWordPieceTokenizer
import torch


def create_bert_vocab():
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
    tokenizer.train(['shakespeare.txt'],
                    vocab_size=30000,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                    limit_alphabet=1000,
                    wordpieces_prefix="##"
                   )
    tokenizer.save('shakespeare-bert.json')


if __name__ == "__main__":
    create_bert_vocab()
    bert_encoder = BertWordPieceTokenizer('shakespeare-bert.json')


