from collections import defaultdict, OrderedDict
import html
import json
import logging
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

import gluonnlp as nlp
import mxnet as mx
import numpy as np

from mosestokenizer import MosesDetokenizer

class Hypotheses():
    """A wrapper around hypotheses for a single utterance
    """
    
    def __init__(self, sents: List[str], scores: List[float], vocab: Optional[nlp.Vocab] = None, tokenizer: Any = None):
        self.sents = sents
        self.scores = scores
        self.tokenizer = tokenizer
        self._vocab = vocab


    def _generate_ln(self, alpha=0.6, tokenizer=None, ln_type='gnmt'):
        self.sent_lens = np.zeros(shape=(len(self.sents),))
        for idx, sent in enumerate(self.sents):
            if ln_type == 'gnmt':
                self.sent_lens[idx] = (5 + len(tokenizer(sent)))**alpha / (5 + 1)**alpha
            elif ln_type == 'length':
                self.sent_lens[idx] = len(tokenizer(sent))
            else:
                raise ValueError("Invalid length normalization type '{}'".format(ln_type))
        return self.sent_lens


    def rescore(self, hyps_list: List[Any], scales: List[float], ln=None, ln_type=None) -> Any:
        """This implements rescoring as:
        s_final = (1-scale)*s_orig + scale*s_new
        
        Args:
            new_scores (TYPE): List of list of new scores
            scales (float, optional): Scale to apply to each list of new scores
        
        Returns:
            TYPE: Description
        """

        scale_total = 0.0
        final_scores = np.zeros((len(self.scores),))
        length_penalties = np.ones((len(self.scores), len(hyps_list)))
        if ln is not None:
            for idx, hyps in enumerate(hyps_list):
                length_penalties[:,idx] = self._generate_ln(alpha=ln, ln_type=ln_type, tokenizer=hyps.tokenizer)
        for idx, (hyps, scale) in enumerate(zip(hyps_list, scales)):
            scale_total += scale
            final_scores += scale*(np.array(hyps.scores) / length_penalties[:,idx])
        final_scores += np.array(self.scores)

        new_idxs = (-final_scores).argsort()
        # Reindex and create a new hypotheses object
        return Hypotheses([self.sents[i] for i in new_idxs], [final_scores[i] for i in new_idxs])



class Corpus(OrderedDict):
    """A ground truth corpus (dictionary of ref sentences)
    """

    @classmethod
    def from_file(cls, fp: TextIO, **kwargs) -> Any:
        if Path(fp.name).suffix == '.json':
            # A ESPNet JSON file with hypotheses and reference
            obj_dict = json.load(fp)
            return cls.from_dict(obj_dict, **kwargs)
        else:
            # A text file (probably LM training data per line)
            return Corpus.from_text(fp, **kwargs)


    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Dict[str, Any]], max_utts: int = None) -> Any:
        """Loads reference texts from the format of Shin et al. (JSON)
        
        Args:
            fp (TextIO): JSON file object
        """

        # Just a dictionary for now
        # but equipped with this factory method
        corpus = cls()

        item_list = sorted(obj_dict.items())
        if max_utts is not None:
            item_list = item_list[:max_utts]
        for utt_id, hyps_dict in item_list:
            # hyps_dict key-values look like:
            # 'ref': "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
            # 'hyp_100' {'score': -10.107752799987793, 'text': ' mister quillter is the apostle of the middle classes and weir glad to welcome his gospel'}
            corpus[utt_id] = hyps_dict['ref'].strip()

        return corpus


    @classmethod
    def from_text(cls, fp: Iterable[str], max_utts=None):
        corpus = cls()
        # For text files, utterance ID is just the zero-indexed line number
        idx = 0
        for line in fp:
            if max_utts is not None and idx >= max_utts:
                break
            corpus[idx] = line.strip()
            idx += 1
        return corpus


    # TODO: WORK ON THIS
    def _get_num_words_in_utt(self, utt: str) -> int:
        return len(utt.split(' '))


    def get_num_words(self) -> Tuple[List[int], int]:

        max_words_per_utt = 0
        num_words_list = []

        for utt_id, utt in self.items():
            num_words = self._get_num_words_in_utt(utt)
            num_words_list.append(num_words)
            if num_words > max_words_per_utt:
                max_words_per_utt = num_words

        return num_words_list, max_words_per_utt


    @staticmethod
    def _edit_distance(a: str, b: str) -> int:

        s1 = a.split(' ')
        s2 = b.split(' ')

        # Implementation is from:
        # https://stackoverflow.com/a/32558749
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


    @staticmethod
    def _word_len(sent: str) -> None:
        raise NotImplementedError


    def wer_score(self, preds: Any) -> float:
        edit_dist = 0.0
        num_ref_words = 0.0
        # TODO: Here we aggregate all edits and then 
        for utt_id, hyps in preds.items():
            pred_sent = hyps.sents[0]
            edit_dist += self._edit_distance(pred_sent, self[utt_id])
            num_ref_words += self._get_num_words_in_utt(self[utt_id])
        return edit_dist / num_ref_words


    def mbleu_score(self, preds: Any):
        logging.warn("Note: we escape tokens in the reference.")
        with NamedTemporaryFile('wt') as fp_ref, NamedTemporaryFile('wt') as fp_hyp:
            for utt_id, hyps in preds.items():
                ref = self[utt_id]
                fp_ref.write(html.unescape(ref) + '\n')

                hyp = hyps.sents[0]
                fp_hyp.write(html.unescape(hyp) + '\n')

            fp_ref.flush()
            fp_hyp.flush()
            logging.warn("Hypothesis file written to '{}'".format(fp_hyp.name))
            logging.warn("Example:\nref: {}\nhyp: {}".format(ref, hyp))

            import sacrebleu
            # mbleu_output = subprocess.check_output("sacrebleu {} < {}".format(fp_ref.name, fp_hyp.name), shell=True)
            mbleu_output = subprocess.check_output("perl multi-bleu.perl {} < {}".format(fp_ref.name, fp_hyp.name), shell=True)
            logging.warn(mbleu_output)

        # TODO: Return the BLEU score
        return None


    def detok_copy(self) -> Any:
        logging.warn("Creating a Moses-detokenized copy of this corpus!")
        detokenize_inner = MosesDetokenizer('en')
        detokenize = lambda x: detokenize_inner(x.split())
        corpus = Corpus()
        for utt_id, sent in self.items():
            corpus[utt_id] = detokenize(sent)
        return corpus


class Predictions(OrderedDict):
    """A dictionary of hypotheses
    """

    # When "deserializing" a corpus into predictions, what token separates the utt id from the number?
    SEPARATOR = '--'


    @classmethod
    def from_file(cls, fp: TextIO, **kwargs):
        suffix = Path(fp.name).suffix
        if suffix == '.json':
            obj_dict = json.load(fp)
            return cls.from_dict(obj_dict, **kwargs)
        elif suffix == '.nobpe':
            return cls.from_nmt(fp, **kwargs)
        else:
            raise ValueError("Hypothesis file of type '{}' is not supported".format(suffix))


    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Dict[str, Any]], max_utts: Optional[int] = None, vocab: Optional[nlp.Vocab] = None, tokenizer = None):
        """Loads hypotheses from the format of Shin et al. (JSON)
        
        Args:
            fp (str): JSON file name
            max_utts (None, optional): Number of utterances to process
            vocab (None, optional): Vocabulary
        
        Returns:
            TYPE: Description
        """

        # Just a dictionary for now
        # but equipped with this factory method
        preds = cls()

        item_list = sorted(obj_dict.items())
        if max_utts is not None:
            item_list = item_list[:max_utts]
        for utt_id, hyps_dict in item_list:

            num_hyps = 0
            for key in hyps_dict.keys():
                if key.startswith("hyp_"):
                    num_hyps += 1

            sents = [None]*num_hyps
            scores = [None]*num_hyps
            # hyps_dict key-values look like:
            # 'ref': "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
            # 'hyp_100' {'score': -10.107752799987793, 'text': ' mister quillter is the apostle of the middle classes and weir glad to welcome his gospel'}
            for hyp_id, hyp_data in hyps_dict.items():
                if not hyp_id.startswith('hyp_'):
                    continue
                # 'hyp_100' --> 99
                idx = int(hyp_id.split('_')[1]) - 1
                sents[idx] = hyp_data['text'].strip()
                scores[idx] = hyp_data['score']

            hyps = Hypotheses(sents, scores, vocab, tokenizer)
            preds[utt_id] = hyps

        return preds


    @classmethod
    def from_nmt(cls, fp: TextIO, max_utts=None, vocab=None, tokenizer=None):
        """Loads hypotheses from Toan's NMT beam output format
        
        Args:
            fp (TextIO): .nobpe filename
            max_utts (None, optional): Number of utterances to process
            vocab (None, optional): Vocabulary
        
        Returns:
            Predictions: Initialized predictions object
        """

        # Just a dictionary for now
        # but equipped with this factory method
        preds = cls()

        pair_idx = 0
        sents = []
        scores = []
        # TODO: Assumes newline at the end
        for line_idx, line in enumerate(fp):
            if max_utts is not None and max_utts <= pair_idx:
                break
            line = line.strip()
            if line == '':
                hyps = Hypotheses(sents, scores, vocab, tokenizer)
                preds[pair_idx] = hyps
                pair_idx += 1
                sents = []
                scores = []
                continue

            line_parts = line.split()
            neg_log_prob = float(line_parts[-1])

            # TEMPORARY: FOR CATCHING IMPROPER PROCESSING, e.g. ... gedi-25.58
            neg_log_prob_ln_str = line_parts[-2]
            str_parts = neg_log_prob_ln_str.split('-')
            # Were they adjoined?
            if len(str_parts[0]) > 0:
                neg_log_prob_ln_str = '-' + str_parts[-1]
                # logging.warn("Line {}: LN score '{}' was found, treating as '{}'".format(line_idx+1, line_parts[-2], neg_log_prob_ln_str))
            neg_log_prob_ln = float(neg_log_prob_ln_str)

            hyp = ' '.join(line_parts[:-2])

            sents.append(hyp)
            scores.append(neg_log_prob_ln)

        return preds



    def to_corpus(self) -> Corpus:

        corpus = Corpus()
        for utt_id, hyps in self.items():
            for idx, sent in enumerate(hyps.sents):
                corpus["{}{}{}".format(utt_id, self.SEPARATOR, idx+1)] = sent

        return corpus


    def to_json(self, fp: TextIO):

        json_dict = {}

        for utt_id, hyps in self.items():

            json_dict[utt_id] = {}

            for idx, (sent, score) in enumerate(zip(hyps.sents, hyps.scores)):
                json_dict[utt_id]["hyp_{}".format(idx+1)] = {
                    "score": float(score),
                    "text": sent
                }

        json.dump(json_dict, fp, indent=2, separators=(',', ': '), sort_keys=True)


class ScoredCorpus(OrderedDict):


    @classmethod
    def from_corpus_and_scores(cls, corpus: Corpus, scores: List[float]) -> OrderedDict:
        scored_corpus = ScoredCorpus()
        for (idx, text), score in zip(corpus.items(), scores):
            scored_corpus[idx] = {'score': score, 'text': text}
        return scored_corpus


    @classmethod
    def from_files(cls, corpus_file: Path, score_file: Path, max_utts: Optional[int] = None) -> OrderedDict:
        """Creates a ScoredCorpus from separate text and score files

        Args:
            corpus_file (Path): A file with a sentence per line
            score_file (Path): A file with either a float or a list of floats/null per line
            max_utts (Optional[int], optional): Number

        Returns:
            ScoredCorpus: Resulting object
        """
        corpus = Corpus.from_file(corpus_file.open('rt'), max_utts=max_utts)
        scores = [json.loads(line) for idx, line in enumerate(score_file.open('rt')) if max_utts is None or idx < max_utts]
        return ScoredCorpus.from_corpus_and_scores(corpus, scores)


    def to_file(self, fp: TextIO, scores_only: bool = False):
        for idx, data in self.items():
            # Either a float or a list of floats/null
            line = "{}\n".format(json.dumps(data['score']))
            if not scores_only:
                line = "{} ".format(data['text']) + line
            fp.write(line)


    def to_predictions(self) -> OrderedDict: 

        hyp_dict = defaultdict(dict)

        for key, val in self.items():
            data_key = key.split(Predictions.SEPARATOR)
            if not (len(data_key) == 2 and data_key[1].isdigit()):
                raise ValueError("This ScoredCorpus cannot be deserialized into Predictions")
            utt_id = data_key[0]
            hyp_num = int(data_key[1])
            hyp_dict[utt_id]['hyp_{}'.format(hyp_num)] = val

        return Predictions.from_dict(hyp_dict)
