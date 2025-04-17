"""
MhRDTech AI Data Processing Module
Created by Mohammad Javad Malekan
Location: Passau, Bayern, Germany
"""

import requests
import json
import sqlite3
import logging
from hazm import Normalizer
import re
from typing import List, Dict, Union
from pathlib import Path

class DataProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.url = "https://api.x.ai/v1/chat/completions"
        self.normalizer = Normalizer()
        
        # Initialize database
        self.db_path = Path("data/training_data.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.setup_database()
        
    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                completion TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        
    def clean_text(self, text: str) -> str:
        """Remove noise and normalize Persian text"""
        text = re.sub(r'[^\u0600-\u06FF\s\d\w]', ' ', text)  # Keep Persian chars, spaces, numbers
        text = self.normalizer.normalize(text)
        return " ".join(text.split())  # Remove extra whitespace
        
    def get_api_response(self, prompt: str) -> Union[str, None]:
        """Get response from xAI API"""
        try:
            payload = {
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(self.url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"API Error: {response.json().get('error', '')}")
                return None
                
        except Exception as e:
            logging.error(f"API request error: {str(e)}")
            return None
            
    def process_prompts(self, prompts: List[str]) -> List[Dict[str, str]]:
        """Process multiple prompts and store results"""
        processed_data = []
        
        for prompt in prompts:
            clean_prompt = self.clean_text(prompt)
            completion = self.get_api_response(clean_prompt)
            
            if completion:
                clean_completion = self.clean_text(completion)
                processed_data.append({
                    "prompt": clean_prompt,
                    "completion": clean_completion
                })
                
                # Store in database
                cursor = self.conn.cursor()
                cursor.execute(
                    "INSERT INTO data (prompt, completion) VALUES (?, ?)",
                    (clean_prompt, clean_completion)
                )
                self.conn.commit()
                
        return processed_data
        
    def export_training_data(self, output_file: str = "data/training_data.json"):
        """Export all data to JSON format for training"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT prompt, completion FROM data")
        rows = cursor.fetchall()
        
        data = [{"prompt": row[0], "completion": row[1]} for row in rows]
        
        Path(output_file).parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    # Example usage
    import os
    processor = DataProcessor(os.getenv("XAI_API_KEY"))
    
    test_prompts = [
        "ترجمه به فارسی: The sky is blue",
        "معنی کلمه 'آسمان' چیست؟",
        "این جمله را بازنویسی کن: هوا امروز خوب است"
    ]
    
    processed_data = processor.process_prompts(test_prompts)
    processor.export_training_data()