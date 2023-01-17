from transformers import AutoTokenizer

from soft_optim.fine_tune import create_dataset


class TestCreateDataset:
    def test_tokenizes_text(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        text = first_example["text"]
        input_ids = first_example["input_ids"]
        expected_input_ids = tokenizer.encode(text)
        assert input_ids == expected_input_ids
        
    def test_adds_labels(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        input_ids = first_example["input_ids"]
        labels = first_example["labels"]
        assert input_ids == labels
        