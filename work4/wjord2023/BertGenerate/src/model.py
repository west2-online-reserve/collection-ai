from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
from my_torch_tools.transformer import TransformerDecoder

class BertGenerate(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        tokenize = BertTokenizer.from_pretrained('bert-base-chinese')
        config = BertConfig.from_pretrained('bert-base-chinese')
        self.decoder = TransformerDecoder(
            tokenize.vocab_size,
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            [config.hidden_size],
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            0.1,
        )
    
    def forward(self, encoder_input, decoder_input, encoder_valid_lens):
        atten_mask = encoder_input.ne(0)
        encoder_output = self.encoder(encoder_input, attention_mask=atten_mask).last_hidden_state # type: ignore
        decoder_state = self.decoder.init_state(encoder_output, encoder_valid_lens)
        return self.decoder(decoder_input, decoder_state)