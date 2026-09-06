"""Token IDs referencing the standard special-token layout of docs/TOKENIZER.md."""

# byte values occupy ids 0..255
# special tokens occupy ids 256..255+num_special
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259

SPECIAL_IDS = {
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<unk>": UNK_ID,
}