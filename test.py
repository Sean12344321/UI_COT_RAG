from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
from peft import get_peft_model
from datasets import load_dataset
from sklearn.metrics import accuracy_score

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
dataset = load_dataset("csv", data_files="./rag_finetune/dataset.csv")
def tokenize_function(examples):
    print(examples.keys())  # 確認 key
    print(examples["模擬輸入內容"][:1])  # 檢查前 3 筆內容是否正確
    return tokenizer(examples["模擬輸入內容"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()