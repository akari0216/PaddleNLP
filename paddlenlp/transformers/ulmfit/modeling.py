# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model

__all__ = [
    "ULMFiTModel",
    "ULMFiTPretrainedModel",
    "ULMFiTForPretraining",
    "ULMFiTPretrainingCriterion",
    "ULMFiTForSequenceClassification"
]


def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    x = paddle.empty(shape=sz)
    x = paddle.full(x.shape, 1 - p)
    x = paddle.bernoulli(x)
    x = paddle.divide(x, paddle.to_tensor(1 - p))
    return x


# Cell
class RNNDropout(nn.Layer):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p=0.5):
        super(RNNDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        return x * dropout_mask(x.detach(), [x.shape[0], 1, *x.shape[2:]],
                                self.p)


# Cell
class WeightDropout(nn.Layer):
    "A module that wraps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module, weight_p, layer_names='weight_hh_l0'):
        super(WeightDropout, self).__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, [
            layer_names
        ]

    def _setweights(self):
        "Apply dropout to the raw weights."
        # dropout
        if self.training:
            old_dict = self.module.state_dict()
            wgt = old_dict["weight_hh_l0"]
            drop_w = nn.functional.dropout(wgt, p=self.weight_p)
            old_dict["weight_hh_l0"] = drop_w
            old_dict["0.cell.weight_hh"] = drop_w
            self.module.set_state_dict(old_dict)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore", category=UserWarning)
            res = self.module(*args)
            return res

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()

    def _do_nothing(self):
        pass


# Cell
class EmbeddingDropout(nn.Layer):
    "Apply dropout with probability `embed_p` to an embedding layer `emb`."

    def __init__(self, emb, embed_p):
        super(EmbeddingDropout, self).__init__()
        self.emb, self.embed_p = emb, embed_p

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.shape[0], 1)
            mask = dropout_mask(self.emb.weight.detach(), size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight

        if scale: masked_embed.mul_(scale)
        padding_idx = self.emb._padding_idx
        if padding_idx is None: padding_idx = -1

        padding_idx = -1
        return nn.functional.embedding(words.astype("int64"), masked_embed,
                                       padding_idx, self.emb._sparse)


# Cell
def awd_lstm_lm_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [
        nn.Sequential(rnn, dp)
        for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)
    ]
    groups = [
        groups +
        [nn.Sequential(model[0].encoder, model[0].encoder_dp, model[1])]
    ]
    return [p for p in groups.parameters()]


# Cell
awd_lstm_lm_config = dict(emb_sz=400,
                          n_hid=1152,
                          n_layers=3,
                          pad_token=1,
                          bidir=False,
                          output_p=0.1,
                          hidden_p=0.15,
                          input_p=0.25,
                          embed_p=0.02,
                          weight_p=0.2,
                          tie_weights=True,
                          out_bias=True)


# Cell
def awd_lstm_clas_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [
        nn.Sequential(model[0].module.encoder, model[0].module.encoder_dp)
    ]
    groups += [
        nn.Sequential(rnn, dp)
        for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)
    ]
    groups = [groups + [model[1]]]
    return [p for p in groups.parameters()]


# Cell
awd_lstm_clas_config = dict(emb_sz=400,
                            n_hid=1152,
                            n_layers=3,
                            pad_token=1,
                            bidir=False,
                            output_p=0.4,
                            hidden_p=0.3,
                            input_p=0.4,
                            embed_p=0.05,
                            weight_p=0.5)


#每个class和forward下面均要写出文档

layer_norm_eps = 1e-6

#配置模型初始参数值，模型下载路径，以及初始化方法
class ULMFiTPretrainedModel(PretrainedModel):
    base_model_prefix = "ulmfit"
    model_config_file = "model_config.json"
    #预训练权重配置
    pretrained_init_configuration = {

    }

    resource_files_names = {"model_state":"model_state.pdparams"}

    #需要上传的权重
    pretrained_resource_files_map = {

    }

    def init_weights(self, layer):
        """Initialize the weights."""

        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.ulmfit.config[
                        "initializer_range"],
                    shape =layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = layer_norm_eps

#模型核心实现，这里将AWD_LSTM替换为ULMFiTModel
@register_base_model
class ULMFiTModel(ULMFiTPretrainedModel):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182"
    initrange = 0.1

    def __init__(self,
                 vocab_sz,
                 emb_sz,
                 n_hid,
                 n_layers,
                 pad_token=1,
                 hidden_p=0.2,
                 input_p=0.6,
                 embed_p=0.1,
                 output_p=0.1,
                 weight_p=0.5,
                 bidir=False,
                 tie_weights=False,
                 bias=True):
        super(ULMFiTModel, self).__init__()
        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.pad_token = pad_token
        self.bs = 1
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz,
                                    padding_idx=pad_token)  #pad_token
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.LayerList([
            self._one_rnn(emb_sz if l == 0 else n_hid,
                          (n_hid if l != n_layers - 1 else emb_sz) //
                          self.n_dir, bidir, weight_p, l)
            for l in range(n_layers)
        ])
        self.encoder.weight.set_value(
            paddle.uniform(shape=self.encoder._size,
                           min=-self.initrange,
                           max=self.initrange))
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.LayerList(
            [RNNDropout(hidden_p) for l in range(n_layers)])
        self.embed_p = embed_p
        self.reset()

    def forward(self, inp, from_embeds=False):
        bs, sl = inp.shape[:2] if from_embeds else inp.shape
        if bs != self.bs: self._change_hidden(bs)
        if not from_embeds:
            inp = self.encoder_dp(inp)

        output = self.input_dp(inp)

        new_hidden = []
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            output, new_h = rnn(output, self.hidden[l])
            new_hidden.append((new_h[0].detach(), new_h[1].detach()))
            if l != self.n_layers - 1: output = hid_dp(output)

        self.hidden = new_hidden
        return output

    def _change_hidden(self, bs):
        self.hidden = [
            self._change_one_hidden(l, bs) for l in range(self.n_layers)
        ]
        self.bs = bs

    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        "Return one of the inner rnn"
        direct = "bidirectional" if bidir else "forward"
        rnn = nn.LSTM(n_in, n_out, 1, time_major=False, direction=direct)
        return WeightDropout(rnn, weight_p)

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.n_hid
              if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        s = paddle.zeros(shape=[self.n_dir, self.bs, nh])
        return (s, s)

    def _change_one_hidden(self, l, bs):
        if self.bs < bs:
            nh = (self.n_hid
                  if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            s = paddle.zeros(shape=[self.n_dir, bs - self.bs, nh])
            return tuple(paddle.concat([h, s], axis=1) for h in self.hidden[l])
        if self.bs > bs:
            return (self.hidden[l][0][:, :bs], self.hidden[l][1][:, :bs])
        return self.hidden[l]

    def reset(self):
        "Reset the hidden states"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]

#分类任务实现
class ULMFiTForSequenceClassification(ULMFiTPretrainedModel):
    def __init__(self, ulmfit, num_classes=2, dropout=None):
        super(ULMFiTForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ulmfit = ulmfit
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                    self.ulmfit.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ulmfit.config["hidden_size"],
                                    self.num_classes)
        self.apply(self.init_weights)

    #postion_ids,attention_mask不清楚是否起作用，暂不加入
    def forward(self, input_ids):
        _, pooled_output = self.ulmfit(
            input_ids)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


#暂不太清楚这个类是干嘛的
class ULMFiTForPretraining(ULMFiTPretrainedModel):
    None
