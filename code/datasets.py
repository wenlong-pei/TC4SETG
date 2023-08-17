import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


# def read_examples(filename, stage):
#     """Read examples from filename."""
#     # filename: e.g. $data_dir/$lang/train/
#     # stage: e.g. train
#     examples = []
#     data = pd.read_csv(filename)
#     nls = data['title'].tolist()
#     codes = data['answer'].tolist()
#     for idx in range(len(nls)):
#         examples.append(
#             Example(
#                 idx=idx,
#                 source=nls[idx],
#                 target=codes[idx],
#             )
#         )
#     return examples

def read_examples(filename, stage):
    """Read examples from filename."""
    # filename: e.g. $data_dir/$lang/train/
    # stage: e.g. train
    examples = []
    data = pd.read_csv(filename,encoding='ISO8859-1')
    nls = data['input_text'].tolist()
    codes = data['target_text'].tolist()

    for idx in range(len(nls)):
        examples.append(
            Example(
                idx=idx,
                source=str(nls[idx]),
                target=str(codes[idx]),
            )
        )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)
        #source_tokens=tokenizer.encode_plus(source_tokens, return_tensors='pt') # gpt2

        source_tokens = source_tokens[:max_source_length - 2]
        #print(source_tokens)
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        #source_tokens=list(filter(None, source_tokens)) # 过滤掉None
        #print(source_tokens)
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        #source_ids=list(filter(None, source_ids)) # 过滤掉None

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        #target_tokens = list(filter(None, target_tokens))  # 过滤掉None
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        source_tokens = source_tokens[1:-2]
        target_tokens = target_tokens[1:-2]
        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features
