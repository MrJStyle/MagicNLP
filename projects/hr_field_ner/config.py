from argparse import ArgumentParser

from pydantic import BaseModel, Field
from paddlenlp.transformers import BertTokenizer

from magic_nlp.models import Device


class Config(BaseModel):
    model_name_or_path: str = Field("bert-base-chinese")
    checkpoint_output_dir: str = Field("checkpoint")
    model_checkpoint_file: str = Field("")
    tag_path: str = Field("tag.dict")
    train_data: str = Field("")
    test_data: str = Field("")
    max_seq_length: int = Field(128)
    batch_size: int = Field(16)
    learning_rate: float = Field(5e-5)
    weight_decay: float = Field(0.0)
    adam_epsilon: float = Field(1e-8)
    max_grad_norm: float = Field(1.0)
    num_train_epochs: int = Field(2)
    max_steps: int = Field(-1)
    warmup_steps: int = Field(0)
    logging_steps: int = Field(1)
    save_steps: int = Field(100)
    seed: int = Field(42)
    device: Device = Field(Device.CPU)


def parse_argument(parser: ArgumentParser) -> Config:
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
             + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys()))
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--tag_path", default="tag.dict", type=str, help="sequence label info")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu."
    )

    args = parser.parse_args()

    return Config(**args.__dict__)

