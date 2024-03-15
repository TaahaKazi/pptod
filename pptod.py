import torch
from transformers import T5Tokenizer
from model.pptod.E2E_TOD.modelling.T5Model import T5Gen_Model
from model.pptod.E2E_TOD.ontology import sos_eos_tokens


class PPtod:
    def __init__(self):
        self.model_path = r'./model/pptod/checkpoints/small/'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.special_tokens = sos_eos_tokens
        self.model = T5Gen_Model(self.model_path, self.tokenizer, self.special_tokens, dropout=0.0,
                            add_special_decoder_token=True, is_training=False)
        self.model.eval()

        # prepare some pre-defined tokens and task-specific prompts
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
            self.sos_r_token_id, self.eos_r_token_id, self.sos_ic_token_id, self.eos_ic_token_id = \
            self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>',
                                             '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>', '<eos_r>', '<sos_d>',
                                             '<eos_d>'])
        bs_prefix_text = 'translate dialogue to belief state:'
        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(bs_prefix_text))

        da_prefix_text = 'translate dialogue to dialogue action:'
        self.da_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(da_prefix_text))

        nlg_prefix_text = 'translate dialogue to system response:'
        self.nlg_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(nlg_prefix_text))

        ic_prefix_text = 'translate dialogue to user intent:'
        self.ic_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ic_prefix_text))

    def get_response(self, frame):
        # an example dialogue context
        # dialogue_context = "<sos_u> can i reserve a five star place for thursday night at 3:30 for 2 people <eos_u>"

        # Build dialogue context from frame history
        dialogue_context = ""
        for msg in frame.conv_history:
            if msg['role'] == 'user_agent':
                msg_str = "<sos_u> " + msg['content'] + " <eos_u>"
                dialogue_context = dialogue_context + msg_str

            if msg['role'] == 'tod_system':
                msg_str = "<sos_r> " + msg['content'] + " <eos_r>"
                dialogue_context = dialogue_context + msg_str

        context_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(dialogue_context))

        # predict system response
        input_id = self.nlg_prefix_id + [self.sos_context_token_id] + context_id + [self.eos_context_token_id]
        input_id = torch.LongTensor(input_id).view(1, -1)
        x = self.model.model.generate(input_ids=input_id, decoder_start_token_id=self.sos_r_token_id,
                                 pad_token_id=self.pad_token_id, eos_token_id=self.eos_r_token_id, max_length=128)
        return self.model.tokenized_decode(x[0])[8:-8]