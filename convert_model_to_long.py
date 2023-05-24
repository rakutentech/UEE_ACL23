'''
Based on: https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

python3 convert_model_to_long.py \
	--pretrained-model base_model/japanese-roberta-base \
	--save-model-dir base_model/roberta-long-japanese-seq4096
'''

import logging
import argparse
import copy
from transformers import RobertaForMaskedLM
from transformers import T5Tokenizer
from transformers.modeling_longformer import LongformerSelfAttention

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


def create_long_model(model, tokenizer, save_model_to, attention_window, max_pos):
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
#     model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos) #giving error, don't need it

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", type=str)
    ap.add_argument("--save-model-dir", type=str)
    ap.add_argument("--tokenizer-dir", type=str, default=None)
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--attention-window", type=int, default=512)
    args = ap.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model)
    model = RobertaForMaskedLM.from_pretrained(args.pretrained_model)
    model,tokenizer = create_long_model(model, tokenizer, args.save_model_dir, args.attention_window, args.max_pos)
    # print(model)
    
    ## load the model again
    from transformers import LongformerForMaskedLM
    model = LongformerForMaskedLM.from_pretrained(args.save_model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.save_model_dir)
    # print(model)
