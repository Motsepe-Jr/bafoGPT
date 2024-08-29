import unittest
from transformers import AutoTokenizer

class TestCustomTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("tokenization/expanded_tokenizer")
        cls.text = """
                        Published online by Cambridge University Press
                        Inkelenkele yakwaXhosa
                        NoXiniha, Hahabe, noMenziwa?
                        ONqabisile, noManxa, noNukwa?
                        Zileleph' izintombi zikaXhosa,
                        OSuthu, Nojoki, noNomalizo?
                        Khuluma ngitshele! Baphi oHintsa
                        Namakhosi kaXhosa nezinkehli
                        Ezazigqok' amanqe zibhinc' intshe,
                        Zidlokovul' eQonce nakwaMonti?
                        Bonke sebenjengamahlamvu wona
                        Apheshulwa ngumoy' umnhla nomzansi.
                        Sebenjengephupho namagam' abo
                        Ahlaba njengomkhonto: anomlingo.
                        Nongqawuze, wena mprofethikazi!
                        Ziphi izintombi zayizolo na?
                        Ziphi izintombi zakuthangi na?
                        Nomhlambi wamathole nezimvana?
                        Okush' okudala okuhl' okubi!
                        Yebo, konke lokho, nikubekephi?
                        Konke kupheshuliwe yiw' umoya,
                        Kwashabalalis' okwefu nenkungu.
                        Zindaba zakho Nongqawuze ndini!
                        """

    def test_basic_tokenization(self):
        tokens = self.tokenizer.tokenize(self.text)
        print("tokens: ", tokens)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

    def test_vocab_size(self):
        expected_vocab_size = 288256
        actual_vocab_size = self.tokenizer.vocab_size
        print("expected_vocab :", expected_vocab_size, "actual_vocab :",actual_vocab_size)
        self.assertEqual(actual_vocab_size, expected_vocab_size, 
                         f"Vocab size mismatch. Expected {expected_vocab_size}, but got {actual_vocab_size}")

    def test_encoding_decoding(self):
        
        encoded = self.tokenizer.encode(self.text, add_special_tokens=False)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(self.text, decoded)

    def test_padding(self):
        texts = ["Zindaba zakho Nongqawuze ndini!", "Kwashabalalis' okwefu nenkungu.", "Yebo, konke lokho, nikubekephi?"]
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        self.assertEqual(encoded.input_ids.shape[1], len(self.tokenizer.encode(texts[2])))

    def test_batch_encoding(self):
        texts = ["Zindaba zakho Nongqawuze ndini!", "Kwashabalalis' okwefu nenkungu.", "Yebo, konke lokho, nikubekephi?"]
        batch_encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.assertEqual(len(batch_encoded.input_ids), len(texts))

if __name__ == "__main__":
    unittest.main()