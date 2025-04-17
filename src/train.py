"""
MhRDTech AI Training Module
Created by Mohammad Javad Malekan
Location: Passau, Bayern, Germany
"""

from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import Dataset
import wandb
import logging
from typing import List, Dict
import json

class ModelTrainer:
    def __init__(self, model_name: str = "HooshvareLab/bert-base-parsbert-uncased"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        wandb.init(project="persian-nlp-xai")
        
    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        dataset = Dataset.from_list(data)
        return dataset.map(
            lambda x: self.tokenizer(
                x["prompt"],
                x["completion"],
                truncation=True,
                padding="max_length"
            ),
            batched=True
        )
        
    def train(self, dataset: Dataset, output_dir: str = "./models/parsbert-finetuned"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            save_steps=500,
            logging_steps=100,
            fp16=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        try:
            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    with open("data/training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    dataset = trainer.prepare_dataset(training_data)
    trainer.train(dataset)