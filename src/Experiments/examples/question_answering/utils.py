import logging
import collections
from collections import OrderedDict
import math
import numpy as np
import json
import six
from scipy.misc import logsumexp
from collections import Counter, OrderedDict
import re
import argparse
import json
import sys
import nltk
logger = logging.getLogger(__name__)

def write_predictions_google(tokenizer, all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    #logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0
        null_ls = 0
        #null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            result_start_ls = log_softmax1d(result.start_logits)
            result_end_ls = log_softmax1d(result.end_logits)
            start_indexes = _get_best_indexes(result_start_ls, n_best_size)
            end_indexes   = _get_best_indexes(result_end_ls, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant

            #feature_null_score =   log_sigmoid(result.cls_logits) #result.start_logits[0] + result.end_logits[0]
            #feature_HasAns_score = log_sigmoid(-result.cls_logits)
            span_start_ls = result_start_ls #+ feature_HasAns_score
            span_end_ls   = result_end_ls #+ feature_HasAns_score

            #if feature_null_score < score_null:
            #    score_null = feature_null_score
            #    min_null_feature_index = feature_index
            #    null_ls = feature_null_score #result.start_logits[0]
            #    #null_end_logit = result.end_logits[0]


            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=span_start_ls[start_index],
                            end_logit=span_end_ls[end_index]))


        #if FLAGS.version_2_with_negative:
        #prelim_predictions.append(
        #    _PrelimPrediction(
        #        feature_index=min_null_feature_index,
        #        start_index=0,
        #        end_index=0,
        #        start_logit=null_ls/2,
        #        end_logit=null_ls/2))


        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, tokenizer, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it

        #if FLAGS.version_2_with_negative:
        #if "" not in seen_predictions:
        #    nbest.append(
        #        _NbestPrediction(
        #            text="",
        #            start_logit=null_ls/2,
        #            end_logit=null_ls/2))


        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        #index_best_non_null_entry = None
        for (i,entry) in enumerate(nbest):
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    #index_best_non_null_entry = i
                    best_non_null_entry = entry

        probs = np.exp(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1


        #if not FLAGS.version_2_with_negative:
        #    all_predictions[example.qas_id] = nbest_json[0]["text"]
        #else:
            # predict "" iff the null score - the score of best non-null > threshold

        #if best_non_null_entry is None:
        #    score_diff = 999999
        #else:
        #    score_diff = np.exp(score_null) - np.exp((best_non_null_entry.start_logit + best_non_null_entry.end_logit))
        #scores_diff_json[example.qas_id] = float(score_diff)
        ##scores_diff_json[example.qas_id] = float(np.exp(score_null))
        #if score_diff > config.args.null_score_diff_threshold:
        #    all_predictions[example.qas_id] = ""
        #else:
        #    all_predictions[example.qas_id] = best_non_null_entry.text

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w",encoding='utf-8') as writer:
        writer.write(json.dumps(all_predictions, indent=4,ensure_ascii=False) + "\n")

    #with open(output_nbest_file, "w") as writer:
    #    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    #with open(output_null_log_odds_file,"w") as writer:
    #    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json


def log_softmax1d(scores):
    if not scores:
        return []
    x = np.array(scores)
    z = logsumexp(x)
    return x-z

def log_sigmoid(score):
    return math.log(1/(1+math.exp(-score)))

def _get_best_indexes(logits, n_best_size, offset=0):
    """Get the n-best logits from a list."""
    sorted_indices = np.argsort(logits)[::-1] + offset
    return list(sorted_indices[:n_best_size])

def get_final_text(pred_text, orig_text, tokenizer, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text



# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
	in_str = (in_str).lower().strip()
	segs_out = []
	temp_str = ""
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
	for char in in_str:
		if rm_punc and char in sp_char:
			continue
		if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
			if temp_str != "":
				ss = nltk.word_tokenize(temp_str)
				segs_out.extend(ss)
				temp_str = ""
			segs_out.append(char)
		else:
			temp_str += char

	#handling last part
	if temp_str != "":
		ss = nltk.word_tokenize(temp_str)
		segs_out.extend(ss)

	return segs_out


# remove punctuation
def remove_punctuation(in_str):
	in_str = str(in_str).lower().strip()
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
	out_segs = []
	for char in in_str:
		if char in sp_char:
			continue
		else:
			out_segs.append(char)
	return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
	m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
	mmax = 0
	p = 0
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				m[i+1][j+1] = m[i][j]+1
				if m[i+1][j+1] > mmax:
					mmax=m[i+1][j+1]
					p=i+1
	return s1[p-mmax:p], mmax

#
def evaluate(ground_truth_file, prediction_file):
	f1 = 0
	em = 0
	total_count = 0
	skip_count = 0
	for instance in ground_truth_file["data"]:
		#context_id   = instance['context_id'].strip()
		#context_text = instance['context_text'].strip()
		for para in instance["paragraphs"]:
			for qas in para['qas']:
				total_count += 1
				query_id    = qas['id'].strip()
				query_text  = qas['question'].strip()
				answers 	= [x["text"] for x in qas['answers']]

				if query_id not in prediction_file:
					sys.stderr.write('Unanswered question: {}\n'.format(query_id))
					skip_count += 1
					continue

				prediction 	= (prediction_file[query_id])
				f1 += calc_f1_score(answers, prediction)
				em += calc_em_score(answers, prediction)

	f1_score = 100.0 * f1 / total_count
	em_score = 100.0 * em / total_count
	return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
	f1_scores = []
	for ans in answers:
		ans_segs = mixed_segmentation(ans, rm_punc=True)
		prediction_segs = mixed_segmentation(prediction, rm_punc=True)
		lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
		if lcs_len == 0:
			f1_scores.append(0)
			continue
		precision 	= 1.0*lcs_len/len(prediction_segs)
		recall 		= 1.0*lcs_len/len(ans_segs)
		f1 			= (2*precision*recall)/(precision+recall)
		f1_scores.append(f1)
	return max(f1_scores)


def calc_em_score(answers, prediction):
	em = 0
	for ans in answers:
		ans_ = remove_punctuation(ans)
		prediction_ = remove_punctuation(prediction)
		if ans_ == prediction_:
			em = 1
			break
	return em

