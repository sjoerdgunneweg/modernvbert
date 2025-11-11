import torch
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from PIL import Image
from huggingface_hub import hf_hub_download

model_id = "ModernVBERT/colmodernvbert"

processor = ColModernVBertProcessor.from_pretrained(model_id)
model = ColModernVBert.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

from PIL import Image
from huggingface_hub import hf_hub_download

# Your inputs
query = "ColModernVBERT matches the performance of models nearly 10x larger on visual document benchmarks."
images = [
    Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space")),
    Image.open(hf_hub_download("ModernVBERT/colmodernvbert", "table.png", repo_type="model"))
]

# Prepare inputs
text_inputs = processor.process_texts([query]).to(model.device)
image_inputs = processor.process_images(images).to(model.device)

# Inference
q_embeddings = model(**text_inputs)
corpus_embeddings = model(**image_inputs)

# Get the similarity scores
scores = processor.score(q_embeddings, corpus_embeddings)[0]

print(f"Query: {query}\n")
print("Similarities:")
for label, score in zip(["Painting Image", "Result Table Image (TARGET)"], scores):
    print(f"  - {label}: {score}")
