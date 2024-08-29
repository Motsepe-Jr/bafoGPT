import io
import json
import sentencepiece as spm


class ZuluTokenizer:
    def __init__(self, pretraining_file, model_file="zulu_tokenizer.model", vocab_size=40000):
        self.pretraining_file = pretraining_file
        self.model_file = model_file
        self.vocab_size = vocab_size
        self.model = io.BytesIO()
        self.sp = None 

    def read_text_file(self):
        """Generates lines of text from the input file."""
        with open(self.pretraining_file, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()

    def train_model(self):
        """Trains the SentencePiece model with the given configurations."""
        spm.SentencePieceTrainer.train(
            sentence_iterator=self.read_text_file(),
            model_writer=self.model,
            shuffle_input_sentence=False,
            max_sentence_length=8192,
            model_type="BPE",
            vocab_size=self.vocab_size,
            split_digits=True,
            split_by_unicode_script=True,
            byte_fallback=True,
            allow_whitespace_only_pieces=True,
            remove_extra_whitespaces=False,
            normalization_rule_name="nmt_nfkc",
        )
        self.save_model()

    def save_model(self):
        """Saves the trained SentencePiece model to a file."""
        with open(self.model_file, "wb") as f:
            f.write(self.model.getvalue())

    def load_model(self):
        """Loads the SentencePiece model from the model file."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_file)

    def tokenize_sentence(self, sentence):
        """Tokenizes a sentence using the loaded SentencePiece model."""
        if self.sp is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        return self.sp.EncodeAsPieces(sentence)

    def print_tokens(self, sentence):
        """Prints tokenized sentence with token indices."""
        tokens = self.tokenize_sentence(sentence)
        print("Tokenized sentence:")
        for i, token in enumerate(tokens, 1):
            print(f"{i}: {token}")

    def get_vocab_size(self):
        """Returns the size of the vocabulary."""
        if self.sp is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        return self.sp.GetPieceSize()

    def save_tokens_to_json(self, output_file="tokenization/zulu_tokenizer/zulu_tokens.json"):
        """Saves all tokens from the vocabulary to a JSON file."""
        if self.sp is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        tokens = [self.sp.IdToPiece(i) for i in range(self.get_vocab_size())]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=4)

        print(f"Saved {len(tokens)} tokens to {output_file}")


if __name__ == "__main__":
    zulu_pretraining_file = "pretrain_dataset/files/pretrain_dataset.txt"
    tokenizer = ZuluTokenizer(pretraining_file=zulu_pretraining_file,
                              model_file="tokenization/zulu_tokenizer/tokenizer.model" # do not porvide the mdoel path when you are training a new tokenizer
                              ) 

    tokenizer.train_model()

    tokenizer.load_model()

    sentence = "Iminyamangabomu sekuyinto evamile neyenzeka nsuku zonke emazweni amaningi asathuthuka"
    tokenizer.print_tokens(sentence)

    tokenizer.save_tokens_to_json()

    vocab_size = tokenizer.get_vocab_size()
    print(f"Number of tokens in the vocabulary: {vocab_size}")
