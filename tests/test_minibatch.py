import unittest
from dataclasses import dataclass, is_dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from trlx.pipeline import MiniBatchIterator
from trlx.pipeline.offline_pipeline import (
    ILQLRolloutStorage,
    ILQLSeq2SeqRolloutStorage,
    PromptPipeline,
)


@dataclass
class DataclassBatch:
    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


class DummyDataset(Dataset, DataclassBatch):
    def __init__(self, num_samples):
        self.query_tensors = torch.randn(num_samples, 64)
        self.response_tensors = torch.randn(num_samples, 64)
        self.logprobs = torch.randn(num_samples, 1)
        self.values = torch.randn(num_samples, 1)
        self.rewards = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.query_tensors)

    def __getitem__(self, idx) -> DataclassBatch:
        return DataclassBatch(
            query_tensors=self.query_tensors[idx],
            response_tensors=self.response_tensors[idx],
            logprobs=self.logprobs[idx],
            values=self.values[idx],
            rewards=self.rewards[idx],
        )


def collate_fn(batch):
    return DataclassBatch(
        query_tensors=torch.stack([sample.query_tensors for sample in batch]),
        response_tensors=torch.stack([sample.response_tensors for sample in batch]),
        logprobs=torch.stack([sample.logprobs for sample in batch]),
        values=torch.stack([sample.values for sample in batch]),
        rewards=torch.stack([sample.rewards for sample in batch]),
    )


class BaseTestMiniBatchIterator(unittest.TestCase):
    def check_mini_batch(self, mb, expected_mini_batch_size):
        if is_dataclass(mb):
            mb = mb.__dict__
        for key, value in mb.items():
            self.assertEqual(value.size(0), expected_mini_batch_size)


class TestMiniBatchDL(BaseTestMiniBatchIterator):
    def test_batch(self):
        batch = DataclassBatch(
            torch.tensor([1]), torch.tensor([2]), torch.tensor([3]), torch.tensor([4]), torch.tensor([5])
        )
        self.assertTrue(is_dataclass(batch))
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in batch.__dict__.values()))

    def test_minibatch_iterator(self):
        # Create Dummy Dataset and DataLoader
        dummy_dataset = DummyDataset(32)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

        iterator = MiniBatchIterator(dummy_dataloader, mb_size=4, num_mb=2)
        for minibatches in iterator:
            for minibatch in minibatches:
                self.assertIsInstance(minibatch, DataclassBatch)
                self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))
                self.check_mini_batch(minibatch, 4)

    def test_minibatch_iterator_with_undivisible_mbsize(self):
        # Create Dummy Dataset and DataLoader
        dummy_dataset = DummyDataset(32)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

        iterator = MiniBatchIterator(dummy_dataloader, mb_size=3, num_mb=3)

        for minibatches in iterator:
            for minibatch in minibatches[:-1]:
                self.assertIsInstance(minibatch, DataclassBatch)
                self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))
                self.check_mini_batch(minibatch, 3)

            # last minibatch has only 2 samples
            minibatch = minibatches[-1]
            self.assertIsInstance(minibatch, DataclassBatch)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))
            self.check_mini_batch(minibatch, 2)

    def test_minibatch_iterator_with_remainder(self):
        # Create Dummy Dataset and DataLoader
        dummy_dataset = DummyDataset(36)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

        iterator = MiniBatchIterator(dummy_dataloader, mb_size=2, num_mb=4)

        for i in range(4):
            minibatches = next(iterator)
            for minibatch in minibatches[:-1]:
                self.assertIsInstance(minibatch, DataclassBatch)
                self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))
                self.check_mini_batch(minibatch, 2)

        # last iteration has only 2 minibatches
        minibatches = next(iterator)
        self.assertEqual(len(minibatches), 2)
        for minibatch in minibatches:
            self.assertIsInstance(minibatch, DataclassBatch)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))
            self.check_mini_batch(minibatch, 2)

    def test_minibatch_iterator_with_smaller_dataset(self):
        # Create Dummy Dataset and DataLoader with size smaller than batch size
        dummy_dataset = DummyDataset(6)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)

        iterator = MiniBatchIterator(dummy_dataloader, mb_size=2, num_mb=4)

        minibatches = next(iterator)

        for minibatch in minibatches:
            self.assertIsInstance(minibatch, DataclassBatch)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in minibatch.__dict__.values()))

        with self.assertRaises(StopIteration):
            minibatches = next(iterator)

    def test_minibatch_content(self):
        dummy_dataset = DummyDataset(32)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

        iterator = MiniBatchIterator(dummy_dataloader, mb_size=4, num_mb=2)

        idx = 0
        for minibatches in iterator:
            for minibatch in minibatches:
                for key in minibatch.__dict__.keys():
                    original_data = getattr(dummy_dataset, key)
                    start_idx = idx * minibatch.__dict__[key].size(0)
                    end_idx = start_idx + minibatch.__dict__[key].size(0)
                    expected_data = original_data[start_idx:end_idx]

                    # Check if the tensor content in the minibatch is consistent with the original dataset
                    self.assertTrue(torch.all(torch.eq(minibatch.__dict__[key], expected_data)))
                idx += 1

        # Test if the iterator covered all the samples in the dataset
        self.assertEqual(idx * iterator.mb_size, len(dummy_dataset))


class TestMiniBatchIteratorWithPromptPipeline(BaseTestMiniBatchIterator):
    def test_minibatch_iterator_with_prompt_pipeline(self):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Create prompts
        prompts = ["This is a test prompt."] * 32

        prompt_pipeline = PromptPipeline(prompts, max_prompt_length=20, tokenizer=tokenizer)

        prompt_dataloader = prompt_pipeline.create_loader(batch_size=8, shuffle=True)

        iterator = MiniBatchIterator(prompt_dataloader, mb_size=4, num_mb=2)
        for minibatches in iterator:
            for minibatch in minibatches:
                self.assertTrue("input_ids" in minibatch)
                self.assertTrue("attention_mask" in minibatch)
                self.assertTrue(isinstance(minibatch["input_ids"], torch.Tensor))
                self.assertTrue(isinstance(minibatch["attention_mask"], torch.Tensor))
                self.check_mini_batch(minibatch, 4)


class TestMiniBatchIteratorWithILQLRollouts(BaseTestMiniBatchIterator):
    def create_dummy_tensors(self, num_samples):
        input_ids = torch.randint(0, 100, (num_samples, 10))
        attention_mask = torch.randint(0, 2, (num_samples, 10))
        rewards = torch.randn(num_samples, 1)
        states_ixs = torch.randint(0, 100, (num_samples, 1))
        actions_ixs = torch.randint(0, 100, (num_samples, 1))
        dones = torch.randint(0, 2, (num_samples, 1), dtype=torch.bool)

        return input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones

    def test_minibatch_iterator_with_ilql_rollout_storage(self):
        # Create dummy data
        input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones = self.create_dummy_tensors(32)

        # Create ILQLRolloutStorage instance
        ilql_rollout_storage = ILQLRolloutStorage(input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones)

        ilql_dataloader = ilql_rollout_storage.create_loader(batch_size=8)

        iterator = MiniBatchIterator(ilql_dataloader, mb_size=4, num_mb=2)

        for minibatches in iterator:
            self.assertEqual(len(minibatches), 2)
            for minibatch in minibatches:
                self.check_mini_batch(minibatch, expected_mini_batch_size=4)

    def test_minibatch_iterator_with_ilql_seq2seq_rollout_storage(self):
        # Create dummy data
        input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones = self.create_dummy_tensors(32)
        decoder_input_ids = torch.randint(0, 100, (32, 10))

        # Create ILQLSeq2SeqRolloutStorage instance
        ilql_seq2seq_rollout_storage = ILQLSeq2SeqRolloutStorage(
            input_ids, attention_mask, decoder_input_ids, rewards, states_ixs, actions_ixs, dones
        )

        ilql_seq2seq_dataloader = ilql_seq2seq_rollout_storage.create_loader(batch_size=8)

        iterator = MiniBatchIterator(ilql_seq2seq_dataloader, mb_size=4, num_mb=2)

        for minibatches in iterator:
            self.assertEqual(len(minibatches), 2)
            for minibatch in minibatches:
                self.check_mini_batch(minibatch, expected_mini_batch_size=4)


if __name__ == "__main__":
    unittest.main()
