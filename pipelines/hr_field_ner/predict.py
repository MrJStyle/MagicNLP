import paddle

from paddlenlp.transformers import BertTokenizer, BertForTokenClassification

from data import Data
from config import Config


config = Config()


def parse_decodes(input_words, id2label, tokenizer, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        input_ids = input_words[idx]["input_ids"]
        sent = "".join([tokenizer.vocab.idx_to_token[id_] for id_ in input_ids][1: end])
        tags = [id2label[x] for x in decodes[idx][1: end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.endswith('-B'):
                    tags_out.append(t.split('-')[0])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            tags.pop()
            # sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def predict():
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)

    data = Data(config.tag_path)
    test_ds = data.load_dataset(config.test_data)
    test_data_loader = data.make_dataloader(
        ds=test_ds,
        tokenizer=tokenizer,
        no_entity_id=data.tag_nums - 1,
        max_seq_len=config.max_seq_length,
        batch_size=config.batch_size
    )

    model = BertForTokenClassification.from_pretrained(
        config.model_name_or_path,
        num_classes=data.tag_nums
    )

    if config.checkpoint_output_dir:
        model_dict = paddle.load(config.model_checkpoint_file)
        model.set_dict(model_dict)

    model.eval()
    pred_list = []
    len_list = []

    for step, batch in enumerate(test_data_loader):
        input_ids, token_type_ids, lengths, labels = batch
        logit = model(input_ids, token_type_ids)
        pred = paddle.argmax(logit, axis=2)

        pred_list.append(pred.numpy())
        len_list.append(lengths.numpy())

    preds = parse_decodes(test_ds, data.index_tag_map, tokenizer, pred_list, len_list)

    file_path = "results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
    # Print some examples
    print(
        "The results have been saved in the file: %s, some examples are shown below: "
        % file_path)
    print("\n".join(preds[:10]))


if __name__ == '__main__':
    predict()
