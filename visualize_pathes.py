import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from colpali.colpali_engine.models.modernvbert.sparse_mlm.processing_sparsemodernvbert_mlm import SparseModernVBertMLMProcessor
from colpali.colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM

# -----------------------------
# Load model and processor
# -----------------------------
model_path = "/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint"

processor = SparseModernVBertMLMProcessor.from_pretrained(model_path, trust_remote_code=True)
model = SparseModernVBertMLM.from_pretrained(model_path, trust_remote_code=True).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# Load images and query
# -----------------------------
images = [Image.open("photo_28.jpg").convert("RGB"),
          Image.open("photo_27.jpg").convert("RGB")]

query = ["Based on the RadQA context, which cervical spine levels show the most severe degenerative changes?"]

text_inputs = processor.process_texts(query)
image_inputs = processor.process_images(images)

# Move tensors to device
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

# -----------------------------
# Gradient-based patch importance
# -----------------------------
# Enable gradients for pixel_values
image_inputs['pixel_values'].requires_grad_(True)

# Forward pass to get embeddings
query_embeddings = model(**text_inputs)          # shape: (1, embedding_dim)
doc_embeddings = model(**image_inputs)           # shape: (batch, embedding_dim)

# Compute similarity score
score = torch.cosine_similarity(query_embeddings, doc_embeddings, dim=-1).sum()
# Backpropagate
score.backward()

# Gradient w.r.t image pixels
grads = image_inputs['pixel_values'].grad.detach()  # (batch, 3, H, W)
# Aggregate over channels (L2 norm)
grads_norm = grads.norm(dim=1)                      # (batch, H, W)

# Normalize per image
grads_norm = (grads_norm - grads_norm.min(dim=(1,2), keepdim=True)[0]) / \
             (grads_norm.max(dim=(1,2), keepdim=True)[0] - grads_norm.min(dim=(1,2), keepdim=True)[0] + 1e-8)

# -----------------------------
# Visualization
# -----------------------------
def visualize_gradient(img_pil, grad_norm, save_path=None):
    img = np.array(img_pil).astype(np.float32) / 255.0
    heatmap = grad_norm.cpu().numpy()
    heatmap = np.expand_dims(heatmap, axis=-1)      # (H, W, 1)
    heatmap_color = np.concatenate([heatmap, np.zeros_like(heatmap), 1-heatmap], axis=-1)  # blue-red

    overlay = 0.6 * img + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Visualize each image
for i, img in enumerate(images):
    visualize_gradient(img, grads_norm[i], save_path=f"grad_heatmap_image_{i}.png")


# TODO: add matplotlib to requirements.txt
