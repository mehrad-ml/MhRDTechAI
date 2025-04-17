# MhRDTech AI

A Persian Language Model based on ParsBERT with continuous learning capabilities using xAI API.

## Created by
**Mohammad Javad Malekan**  
Location: Passau, Bayern, Germany

## Project Overview
MhRDTech AI is an advanced Persian language model that leverages the power of ParsBERT and continuous learning through xAI API integration. The model is designed to handle various NLP tasks in Persian, including text generation, translation, and text completion.

## Features
- Persian text processing and normalization
- Integration with xAI API (grok-beta)
- Continuous learning capabilities
- FastAPI-based REST API
- SQLite database for data management
- Weights & Biases integration for monitoring

## Installation
```bash
pip install -r requirements.txt
```

## Environment Setup
Set the following environment variables:
```bash
export XAI_API_KEY="your-api-key"
```

## Usage
1. Start the API server:
```bash
uvicorn src.main:app --reload
```

2. Train the model:
```bash
python src/train.py
```

3. Process new data:
```bash
python src/data_processor.py
```

## Project Structure
```
MhRDTechAI/
├── config/
├── data/
├── logs/
├── models/
└── src/
    ├── main.py
    ├── train.py
    └── data_processor.py
```

## License
All rights reserved. Copyright © 2024 Mohammad Javad Malekan