import os
from tqdm import tqdm
from datasets import load_dataset
from utils import SquadExample, SquadFeatures, SquadResult
class Loader:
    def __init__(self, tokenizer, local_rank):
        self.tokenizer = tokenizer
        self.local_rank = local_rank
    def read_from_file(self, data_path):
        pass
    def load(self):
        pass

class SquadLoader(Loader):

    def read_from_file(self, data_dir, is_eval):
        if data_dir:
            data_files = {}
            data_files['train'] = os.path.join(data_dir, "train-v2.0.json")
            data_files['validation'] = os.path.join(data_dir, "dev-v2.0.json")
            datasets = load_dataset('json', data_files=data_files, field="data")
        else:
            datasets = load_dataset("squad_v2")
        self.datasets = datasets
        ## convert data to examples
        examples = []
        if is_eval:
            dataset = datasets["validation"]
        else:
            dataset = datasets["train"]
        for data in tqdm(dataset):
            qas_id = data['id']
            question_text = data['question']
            paragraph_text = data["context"]
            answer_text = data["answers"]["text"]
            start_position = data['answers']["answer_start"]
            is_impossible = (len(answer_text) == 0)
            answer_text = None if is_impossible else answer_text[0]
            start_position = None if is_impossible else start_position[0]
            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                paragraph=paragraph_text,
                answer_text=answer_text,
                start_position=start_position,
                is_impossible=is_impossible)
            examples.append(example)

    def convert_examples_to_features(
            self,
            examples,
            tokenizer,
            max_seq_length,
            doc_stride,
            max_query_length,
            is_training,
            padding_strategy="max_length",
            threads=1,
            tqdm_enabled=True,
    ):
        """
        Converts a list of examples into a list of features that can be directly given as input to a model. It is
        model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
        Args:
            examples: list of :class:`~transformers.data.processors.squad.SquadExample`
            tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
            max_seq_length: The maximum sequence length of the inputs.
            doc_stride: The stride used when the context is too large and is split across several features.
            max_query_length: The maximum length of the query.
            is_training: whether to create features for model evaluation or model training.
            padding_strategy: Default to "max_length". Which padding strategy to use
            threads: multiple processing threads.
            tqdm_enabled: whether enable tqdm
        Returns:
            list of :class:`~transformers.data.processors.squad.SquadFeatures`
        Example::
            processor = SquadV2Processor()
            examples = processor.get_dev_examples(data_dir)
            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
            )
        """
        # Defining helper methods
        features = []

        threads = min(threads, cpu_count())
        from functools import partial
        with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
            annotate_ = partial(
                convert_example_to_features,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                padding_strategy=padding_strategy,
                is_training=is_training,
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert squad examples to features",
                    disable=not tqdm_enabled,
                )
            )

        new_features = []
        unique_id = 1000000000
        example_index = 0
        for example_features in tqdm(
                features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1
            example_index += 1
        features = new_features
        del new_features
        dataset = convert_features_to_dataset(features, is_training)
        return features, dataset

    def convert_features_to_dataset(features, is_training):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )
        return dataset

    def convert_example_to_features(
            example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
    ):
        """Convert one single example to feature"""
        features = []
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(
            example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
        )

        # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
        # in the way they compute mask of added tokens.
        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            # Define the side we want to truncate / pad and the text/pair sorting
            if tokenizer.padding_side == "right":
                texts = truncated_query
                pairs = span_doc_tokens
                truncation = TruncationStrategy.ONLY_SECOND.value
            else:
                texts = span_doc_tokens
                pairs = truncated_query
                truncation = TruncationStrategy.ONLY_FIRST.value

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                texts,
                pairs,
                truncation=truncation,
                padding=padding_strategy,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][
                                     : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                            len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                        tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict or (
                    "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
            ):
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implementation also keeps the classification token (set to 0)
            p_mask = np.ones_like(span["token_type_ids"])
            if tokenizer.padding_side == "right":
                p_mask[len(truncated_query) + sequence_added_tokens:] = 0
            else:
                p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

            pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
            special_token_indices = np.asarray(
                tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
            ).nonzero()

            p_mask[pad_token_indices] = 1
            p_mask[special_token_indices] = 1

            # Set the cls index to 0: the CLS index can be used for impossible answers
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                InputFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,
                    # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    qas_id=example.qas_id,
                )
            )
        return features

    def squad_convert_example_to_features_init(tokenizer_for_convert):
        global tokenizer
        tokenizer = tokenizer_for_convert



data_files = {'data':"../../datasets/squad_v2/dev-v2.0.json"}
dataset = load_dataset('json', data_files="../../datasets/squad_v2/dev-v2.0.json",field="data")