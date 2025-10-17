import torch


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))
        
def predict_seq2seq(net, src_sentence, tokenizer, num_steps,
                    device):
    net.eval()
    src_tokens = tokenizer.tokenize('[CLS]' + src_sentence)
    src_tokens = tokenizer.convert_tokens_to_ids(src_tokens)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, tokenizer.convert_tokens_to_ids('[PAD]'))
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    atten_mask = enc_X.ne(0).to(device)
    enc_outputs = net.encoder(enc_X, atten_mask).last_hidden_state
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(
        [tokenizer.convert_tokens_to_ids('[SEP]')], dtype=torch.long, device=device), dim=0
    )
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == tokenizer.convert_tokens_to_ids('[SEP]'):
            break
        output_seq.append(pred)
    return tokenizer.convert_ids_to_tokens(output_seq)