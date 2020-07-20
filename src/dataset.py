import os
import random
import torch


class ExampleTriple():
    def __init__(self, ind, text_head, text_relation, text_tail, label):
        self.ind = ind
        self.text_head = text_head
        self.text_relation = text_relation
        self.text_tail = text_tail
        self.label = label


class KGExample():
    def __init__(self, data_dir):
        self.labels = set()
        self.data_dir = data_dir
        self.length = 0

    def __len__(self):
        return self.length

    def _read_tsv(self, data_dir):
        '''获取给定路径下的文件，并按照行分割，存入列表并返回'''
        with open(data_dir, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                line = line.rstrip("\n")
                lines.append(line)
        return lines

    def _create_example(self, str_type):
        entity2text = {}
        with open(os.path.join(self.data_dir, "entity2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                if len(line) == 2:
                    entity2text[line[0]] = line[1]
        entities = entity2text.keys()

        relation2test = {}
        with open(os.path.join(self.data_dir, "relation2text.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n").split("\t")
                if len(line) == 2:
                    relation2test[line[0]] = line[1]

        data_lines = self._read_tsv(os.path.join(self.data_dir, f"{str_type}.tsv"))
        line_str_set = set("\t".join(line) for line in data_lines)

        examples = []
        for i, line in enumerate(data_lines):
            line = line.rstrip("\n").split("\t")
            head, relation, tail = line[:3]
            text_head, text_relation, text_tail = entity2text[head], relation2test[relation], entity2text[tail]
            if str_type == "dev" or str_type == "test":
                ind = f"{str_type}-{i}"
                label = line[-1]
                if label == "1":
                    label = 1
                elif label == "-1":
                    label = 0
                self.labels.add(label)
                self.length += 1
                examples.append(
                    ExampleTriple(
                        ind=ind,
                        text_head=text_head,
                        text_relation=text_relation,
                        text_tail=text_tail,
                        label=label
                    )
                )
            elif str_type == "train":
                ind = f"{str_type}-{i}"
                label = 1
                self.labels.add(label)
                self.length += 1
                examples.append(
                    ExampleTriple(
                        ind=ind,
                        text_head=text_head,
                        text_relation=text_relation,
                        text_tail=text_tail,
                        label=label
                    )
                )

                rdm = random.random()
                if rdm <= 0.5:  # 替换head
                    while True:
                        entities_list = list(entities)
                        entities_list.remove(head)
                        wrong_head = random.choice(entities_list)
                        if f"\t{wrong_head}\t{relation}\t{tail}\n" not in line_str_set:
                            break
                    self.length += 1
                    examples.append(
                        ExampleTriple(
                            ind=f"{str_type}_wrong-{i}",
                            text_head=entity2text[wrong_head],
                            text_relation=text_relation,
                            text_tail=text_tail,
                            label=0
                        )
                    )
                elif rdm > 0.5:
                    while True:
                        entities_list = list(entities)
                        entities_list.remove(tail)
                        wrong_tail = random.choice(entities_list)
                        if f"\t{head}\t{relation}\t{wrong_tail}\n" not in line_str_set:
                            break
                    self.length += 1
                    examples.append(
                        ExampleTriple(
                            ind=f"{str_type}_wrong-{i}",
                            text_head=text_head,
                            text_relation=text_relation,
                            text_tail=entity2text[wrong_tail],
                            label=0
                        )
                    )
        return examples

    def get_train_example(self):
        return self._create_example("train")

    def get_test_example(self):
        return self._create_example("test")

    def get_dev_example(self):
        return self._create_example("dev")


class KGData():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _read_tsv(self, data_dir):
        '''获取给定路径下的文件，并按照行分割，存入列表并返回'''
        with open(data_dir, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                line = line.rstrip("\n")
                lines.append(line)
        return lines

    def get_relations(self):
        relations = []
        with open(os.path.join(self.data_dir, "relation.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                relations.append(line)
        return relations

    def get_entities(self):
        entities = []
        with open(os.path.join(self.data_dir, "entities.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                entities.append(line)
        return entities

    def get_labels(self):
        return [0, 1]

    def get_train_triples(self):
        return self._read_tsv(os.path.join(self.data_dir, "train.tsv"))

    def get_test_triples(self):
        return self._read_tsv(os.path.join(self.data_dir, "test.tsv"))

    def get_dev_triples(self):
        return self._read_tsv(os.path.join(self.data_dir, "dev.tsv"))


class KGDataset():
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        ind = example.ind
        text_head = example.text_head
        text_relation = example.text_relation
        text_tail = example.text_tail
        label = example.label

        token_head = self.tokenizer.tokenize(text_head)
        token_relation = self.tokenizer.tokenize(text_relation)
        token_tail = self.tokenizer.tokenize(text_tail)

        while True:
            len_h, len_r, len_t = len(text_head), len(text_relation), len(text_tail)
            if len_h + len_r + len_t <= self.max_len-4:
                break
            elif len_h > len_r and len_h > len_t:
                token_head.pop()
            elif len_r > len_h and len_r > len_t:
                token_relation.pop()
            else:
                token_tail.pop()
        tokens = ['[CLS]'] + token_head + ['[SEP]']
        token_type_ids = [0] * len(tokens)

        tokens += token_relation + ['[SEP]']
        token_type_ids += [1] * (len(token_relation) + 1)

        tokens += token_tail + ['[SEP]']
        token_type_ids += [0] * (len(token_tail) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding  = self.max_len - len(input_ids)
        if padding > 0:
            input_ids += [0] * padding
            attention_mask += [0] * padding
            token_type_ids += [0] * padding
        if label == 0:
            label = [1, 0]
        elif label == 1:
            label = [0, 1]
        return {
            "tokens": tokens,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)

        }



if __name__ == '__main__':
    import transformers
    kd = KGExample("../data/FB13")
    kd_dev = kd.get_dev_example()
    tokenizer = transformers.BertTokenizer.from_pretrained("../bert-base-tiny-uncased/vocab.txt")
    kds = KGDataset(kd_dev, tokenizer, 128)
    print(kds[0])


