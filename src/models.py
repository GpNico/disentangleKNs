

import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, OPTForCausalLM, PreTrainedTokenizer, XLMWithLMHeadModel, LlamaForCausalLM, BloomForCausalLM


class ModelWrapper(nn.Module):
    
    def __init__(self,
                 model_name: str,
                 device: str,
                 output_hidden_states: bool = False):
        """
            Wrapper class around hugging face models that support
            different model_type.
        """
        super(ModelWrapper, self).__init__()
        
        self.model_name = model_name
        
        # Device
        self.device = device
        
        if 'opt' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f'facebook/{model_name}', 
                                            output_hidden_states = output_hidden_states,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            )
            self.model = OPTForCausalLM.from_pretrained(
                                            f"facebook/{model_name}", 
                                            config=self.config,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            ).to(device)
        if 'gemma' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f'google/{model_name}', 
                                            output_hidden_states = output_hidden_states,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            )
            self.model = AutoModelForCausalLM.from_pretrained(
                                            f"google/{model_name}", 
                                            config=self.config,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            ).to(device)
        elif 'Llama' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f'meta-llama/{model_name}', 
                                            output_hidden_states = output_hidden_states,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            )
            self.model = LlamaForCausalLM.from_pretrained(
                                            f"meta-llama/{model_name}",
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            ).to(device)
        elif 'bert' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            model_name,
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            output_hidden_states = output_hidden_states
                                            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                                            model_name,
                                            torch_dtype = torch.float16,
                                            config=self.config
                                            ).to(device)
        elif 'bloom' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f"bigscience/{model_name}",
                                            torch_dtype = torch.float16, ########## FOR FASTER INFERENCE #############
                                            output_hidden_states = output_hidden_states
                                            )
            self.model = BloomForCausalLM.from_pretrained(
                                            f"bigscience/{model_name}",
                                            torch_dtype = torch.float16,
                                            config=self.config
                                            ).to(device)
        elif 'xlm' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f"FacebookAI/{model_name}",
                                            output_hidden_states = output_hidden_states
                                            )
            self.model = XLMWithLMHeadModel.from_pretrained(
                                            f"FacebookAI/{model_name}",
                                            config=self.config
                                            ).to(device)
        elif 't5' in model_name:
            self.config = AutoConfig.from_pretrained(
                                            f"google/{model_name}",)
                                            #torch_dtype = torch.float16,
                                            #)
            self.model = T5ForConditionalGeneration.from_pretrained(
                                            f"google/{model_name}",
                                            #torch_dtype = torch.float16,
                                            config=self.config
                                            ).to(device)
        self.model.eval()
        
        
    def forward(self,
                input_ids: torch.Tensor = None, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
            Forward the model and returns logits.
            
            Args:
                input_ids (torch.Tensor) shape [BS, L]
                attention_mask (torch.Tensor) shape [BS, L]
            Returns:
                logits (torch.Tensor) shape [BS, L, Voc Size]
        
        """

        logits = self.model(input_ids = input_ids,
                            attention_mask = attention_mask).logits
            
        return logits
    
    def get_prediction_logits(self, 
                              input_ids: torch.Tensor = None, 
                              attention_mask: torch.Tensor = None,
                              tokenizer: PreTrainedTokenizer = None) -> torch.Tensor:
        """
            Wrapper that computes the relevant logits for the prediction:
                - For MLM models (e.g. BERT) the logits under the (last) [MASK]
                  token are returned.
                - For Autoregressive models (e.g. OPT) the logits under the last 
                  non-padding tokens are returned.
        
            In any case the output tensor is [batch_size, voc_size]
        
        """
        
        # forward pass
        
        with torch.no_grad():
            if self.config.model_type != 't5':
                logits = self.forward(
                            input_ids = input_ids.to(self.device),
                            attention_mask = attention_mask.to(self.device)
                            ) # the .logits is in the wrapper class
            else:
                decoder_input_ids = torch.empty((input_ids.shape[0],1), dtype=torch.int32)
                decoder_input_ids.fill_(tokenizer.pad_token_id)
                logits = self.model(
                            input_ids=input_ids.to(self.device), 
                            attention_mask = attention_mask.to(self.device), 
                            decoder_input_ids=decoder_input_ids.to(self.device)
                            ).logits
        
        # Model Specific
        if self.config.model_type in ['bert', 'xlm']:
            mask_pos_i, mask_pos_j = torch.where(input_ids == tokenizer.mask_token_id)
            
            masked_logits = logits[mask_pos_i, mask_pos_j].view(input_ids.shape[0], -1, logits.shape[-1]) # Because in early Autoprompt there's multiple
                                                                                                          # mask tokens so this only focus on the last one
            return masked_logits[:,-1,:]
        
        elif self.config.model_type in ['opt', 'llama', 'bloom', 'gemma2']:
            return logits[torch.arange(input_ids.shape[0]),
                          attention_mask.sum(dim=-1)-1] # that's a neat trick!
        elif self.config.model_type == 't5':
            return logits[:,0,:]