import multiprocessing
from itertools import chain

import datasets
from transformers import AutoTokenizer, PretrainedConfig


def load_and_preprocess_data(
    model_name: str,
    seq_length: int | None,
    dataset_name: str,
    dataset_subset: str | None,
    config: PretrainedConfig,
) -> datasets.Dataset:
    """
    Prepare data for manual CausalLM training (without trainer).

    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = datasets.load_dataset(dataset_name, dataset_subset)

    seq_length = seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    column_names = data['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):  # noqa: ANN001, ANN202
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples: dict[str, list]) -> dict[str, list]:
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.  # noqa: E501
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.  # noqa: E501
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f'Grouping texts in chunks of {seq_length}',
    )

    return lm_datasets['train']
