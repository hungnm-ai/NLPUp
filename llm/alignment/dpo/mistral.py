# from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model
# from transformers import AutoModelForCausalLM
#
# model = AutoModelForCausalLM.from_pretrained('gpt2')
# model = PeftModel.from_pretrained(model, adapter_name="adapter", config=)
# model.load_adapter()
# model.add_adapter()
#
# from trl import DPOTrainer
#
# trainer = DPOTrainer(model)
