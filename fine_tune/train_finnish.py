"""
Fine-tune Llama 4 for Finnish language using Fireworks AI

This script fine-tunes the Llama 4 Maverick model on Finnish conversations
to improve its understanding of Finnish language and Ajatuskumppani context.
"""

import os
import json
from fireworks import Dataset, FineTuningJob

# Configuration
BASE_MODEL = "llama4-maverick-instruct-basic"
DATASET_PATH = "../datasets/finnish_conversations.jsonl"
DATASET_NAME = "ajatuskumppani-finnish-v1"
JOB_NAME = "ajatuskumppani-finnish-llama4"

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 4


def load_dataset(path: str) -> list:
    """Load JSONL dataset"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_fireworks_dataset(data: list, name: str) -> Dataset:
    """Create and upload dataset to Fireworks"""
    print(f"📦 Creating dataset: {name}")
    print(f"   Examples: {len(data)}")
    
    dataset = Dataset.from_list(
        data=data,
        name=name
    )
    
    print(f"✅ Dataset created: {dataset.id}")
    return dataset


def start_fine_tuning(dataset: Dataset, base_model: str, job_name: str) -> FineTuningJob:
    """Start fine-tuning job"""
    print(f"\n🚀 Starting fine-tuning job: {job_name}")
    print(f"   Base model: {base_model}")
    print(f"   Dataset: {dataset.name}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    job = FineTuningJob.create(
        base_model=base_model,
        dataset=dataset,
        display_name=job_name,
        hyperparameters={
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
        }
    )
    
    print(f"✅ Job created: {job.id}")
    print(f"   Status: {job.status}")
    print(f"\n📊 Monitor progress at:")
    print(f"   https://app.fireworks.ai/dashboard/fine-tuning/{job.id}")
    
    return job


def main():
    """Main fine-tuning workflow"""
    print("=" * 80)
    print("Ajatuskumppani - Finnish Fine-tuning")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ Error: FIREWORKS_API_KEY not set")
        print("   Set it with: export FIREWORKS_API_KEY=fw_...")
        return
    
    # Load dataset
    print(f"\n📂 Loading dataset: {DATASET_PATH}")
    data = load_dataset(DATASET_PATH)
    print(f"✅ Loaded {len(data)} examples")
    
    # Create Fireworks dataset
    dataset = create_fireworks_dataset(data, DATASET_NAME)
    
    # Start fine-tuning
    job = start_fine_tuning(dataset, BASE_MODEL, JOB_NAME)
    
    # Wait for completion (optional)
    print(f"\n⏳ Waiting for job to complete...")
    print(f"   This may take 30-60 minutes depending on dataset size.")
    print(f"   You can close this script and check status later with:")
    print(f"   python check_job.py {job.id}")
    
    # Save job ID
    with open("job_id.txt", "w") as f:
        f.write(job.id)
    
    print(f"\n✅ Job ID saved to job_id.txt")
    print(f"\n" + "=" * 80)
    print("Fine-tuning started successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

