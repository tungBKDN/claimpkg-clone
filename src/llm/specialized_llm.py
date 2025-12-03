from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

class SpecializedLLM:
    MODEL_ID = "meta-llama/Llama-3.2-1B"
    NUM_EPOCHS = 1
    MODEL_DIR = r'..\resources\model\specialized_nov26\epoch_' + str(NUM_EPOCHS)

    def __init__(self):

        # Load the PEFT config
        self.peft_config = PeftConfig.from_pretrained(self.MODEL_DIR)

        # Load the base model (same as used during training)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.peft_config.base_model_name_or_path)

        # Load the adapter weights on top of the base model
        self.model = PeftModel.from_pretrained(self.base_model, self.MODEL_DIR)

        # Load tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(SpecializedLLM.MODEL_ID, use_fast=False)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, input_text: str, max_new_tokens: int = 128):
        prompt = (
            f"Claim: {input_text}\n"
            "Task: Decompose the claim into pseudo-subgraphs.\n"
            "Follow EXACTLY this output format:\n\n"
            "<e>ENTITY_1</e> || RELATION || <e>ENTITY_2</e>\n"
            "<e>ENTITY_3</e> || RELATION || <e>ENTITY_4</e>\n\n"
            "<e>ENTITY_X</e> || RELATION || <e>ENTITY_Y</e>\n\n"
            "Guidelines:\n"
            "- Use multiple pseudo-subgraphs (P1, P2, ...).\n"
            "- Use only KG-style relations (location, partOf, officialLanguage, ...).\n"
            "- Use ~RELATION for reverse edges.\n"
            "- Use unknown_i for implicit entities.\n"
            "- DO NOT repeat the claim.\n"
            "- Your output must match the structure in the labels.\n"
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")

        # Generate output
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,    # limit only the generated output length
                num_beams=4,           # beam search for better results
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id  # avoid warnings about pad_token_id
            )

        # Decode predictions
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs[0]