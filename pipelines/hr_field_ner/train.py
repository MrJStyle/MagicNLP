import os
import time

import paddle

from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import BertTokenizer, BertForTokenClassification, LinearDecayWithWarmup

from data import Data
from config import Config

# parser = argparse.ArgumentParser()
# config: Config = parse_argument(parser)


config = Config()


def train():
    ignore_label = -1

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)

    data = Data(config.tag_path)
    train_ds = data.load_dataset(config.train_data)
    train_data_loader = data.make_dataloader(
        train_ds, tokenizer, data.tag_nums - 1, config.max_seq_length, config.batch_size
    )

    model = BertForTokenClassification.from_pretrained(config.model_name_or_path, num_classes=data.tag_nums)

    num_training_steps = config.max_steps if config.max_steps > 0 else len(
        train_data_loader) * config.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(config.learning_rate, num_training_steps,
                                         config.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=config.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=list(data.tag_index_map.keys()))

    global_step = 0
    last_step = config.num_train_epochs * len(train_data_loader)
    tic_train = time.time()
    for epoch in range(config.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, _, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = loss_fct(logits, labels)
            avg_loss = paddle.mean(loss)
            if global_step % config.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       config.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % config.save_steps == 0 or global_step == last_step:
                # evaluate(model, loss_fct, metric, test_data_loader,
                #          label_num)
                paddle.save(model.state_dict(),
                            os.path.join(config.checkpoint_output_dir,
                                         "model_%d.pdparams" % global_step))


if __name__ == '__main__':
    # data = Data("/home/mrj/Code/Personal/Python/magic_nlp/pipelines/hr_field_ner/tag.dict")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # ds = data.load_dataset("/home/mrj/Code/Personal/Python/magic_nlp/pipelines/hr_field_ner/data/train_cert.tsv")
    # dl = data.make_dataloader(ds, tokenizer, 10, 32)
    # print(next(iter(dl)))

    # print(config)
    # print(Data.load_label_vocab("/home/mrj/Code/Personal/Python/magic_nlp/pipelines/hr_field_ner/tag.dict"))

    train()
