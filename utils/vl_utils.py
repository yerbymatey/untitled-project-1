import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images, get_device_and_dtype

def setup_vl_model(model_path: str = "deepseek-ai/deepseek-vl-7b-chat"):
    """Setup the VL model and processor."""
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    device, dtype = get_device_and_dtype()
    vl_gpt = vl_gpt.to(dtype).to(device).eval()
    
    return vl_chat_processor, tokenizer, vl_gpt, device, dtype

def process_vl_conversation(conversation, vl_chat_processor, vl_gpt, device, dtype):
    """Process a VL conversation and return the model's response."""
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(device, dtype=dtype)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=77,  # 77 for CLIP
        do_sample=False,
        use_cache=True
    )

    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer, prepare_inputs['sft_format'][0] 