import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import numpy as np
from transformers import AutoTokenizer
import torch


def load_models(): 

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_ai_hum = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = True, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )

    model_llm = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = True, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )

    model_ai_hum = torch.load('C:/Users/Shoaib/Downloads/sample_localhost/app/bert_model', map_location="cpu", weights_only=False)
    model_llm.load_state_dict(torch.load('C:/Users/Shoaib/Downloads/sample_localhost/app/bert_model_llm_only__2.pth', map_location="cpu"))

    return tokenizer, model_ai_hum, model_llm


def classify_text(text, model_ai_hum, model_llm, tokenizer):

    assert tokenizer is not None, "Error: tokenizer is None!"

    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to("cpu")
    attention_mask = encoded_input["attention_mask"].to("cpu")

    with torch.no_grad():
        outputs = model_ai_hum(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_label = np.argmax(logits.cpu().numpy(), axis=1).item()

    predicted_label_llm = None 

    if predicted_label == 1:
        predicted_label = "Human"
    else:
        predicted_label = "AI"

        with torch.no_grad():
            outputs_llm = model_llm(input_ids, attention_mask=attention_mask)
            logits_llm = outputs_llm.logits
            predicted_label_llm = np.argmax(logits_llm.cpu().numpy(), axis=1).item()

        if predicted_label_llm == 0:
            predicted_label_llm = "LLAMA"
        else:
            predicted_label_llm = "Gemini"

    return {"type": predicted_label, "llm": predicted_label_llm if predicted_label_llm else "N/A"}

    
# Example usage:
# abstract = df["Abstract"].values[39000]
# result = classify_abstract(abstract, model_ai_hum, model_llm, tokenizer)
# print(result)
