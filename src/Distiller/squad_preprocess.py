import os
from tqdm import tqdm
from datasets import load_dataset


class Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph,
                 answer_text=None,
                 start_position=None,
                 is_impossible=None):
        """
            Construct a Extractive QA(squad style) example
            Args:
                qas_id: Unique id for the example
                question_text: text of questions
                paragraph: context sentences
                orig_answer_text: the answer text
                start_position: start_position of the answer in the paragraph
                is_impossible: if it is impossible to get answer from the paragraph
        """
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = paragraph
        self.answer_text = answer_text
        self.start_position = start_position
        self.is_impossible = is_impossible

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position]
            self.end_position = char_to_word_offset[
                min(start_position + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"qas_id: {self.qas_id}"
        s += f", question_text: {self.question_text}"
        s += f", paragraph: {self.context_text}"
        if self.start_position:
            s += f", start_position: {self.start_position}"
        if self.end_position:
            s += f", end_position: {self.end_position}"
        if self.is_impossible:
            s += f", is_impossible: {self.is_impossible}"
        return s

class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits

# class SquadFeatures:


def convert_data_to_examples(dataset, aug=None):
    examples = []
    for data in tqdm(dataset):
        qas_id = data['id']
        question_text = data['question']
        paragraph_text = data["context"]
        answer_text = data["answers"]["text"]
        start_position = data['answers']["answer_start"]
        is_impossible = (len(answer_text) == 0)
        answer_text = None if is_impossible else answer_text[0]
        start_position = None if is_impossible else start_position[0]
        example = Example(
            qas_id=qas_id,
            question_text=question_text,
            paragraph=paragraph_text,
            answer_text=answer_text,
            start_position=start_position,
            is_impossible=is_impossible)
        examples.append(example)
    return examples


def read_and_aug(args, aug=None, data_dir=None):
    if data_dir:
        data_files = {}
        data_files['train'] = os.path.join(data_dir,"train-v2.0.json")
        data_files['validation'] = os.path.join(data_dir, "dev-v2.0.json")
        datasets = load_dataset('json', data_files=data_files, field="data")
    else:
        datasets = load_dataset("squad_v2")

    if args.do_train:
        input_data = datasets["train"]
        train_examples = convert_data_to_examples(input_data, aug)





def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False