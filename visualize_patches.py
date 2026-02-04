import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

from colpali.colpali_engine.models.modernvbert.sparse_mlm.processing_sparsemodernvbert_mlm import SparseModernVBertMLMProcessor
from colpali.colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM

model_path = "/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint"

processor = SparseModernVBertMLMProcessor.from_pretrained(model_path, trust_remote_code=True)
model = SparseModernVBertMLM.from_pretrained(model_path, trust_remote_code=True).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

images = [Image.open("photo_28.jpg").convert("RGB"),
          Image.open("photo_27.jpg").convert("RGB")]

query = ["Based on the RadQA context, which cervical spine levels show the most severe degenerative changes?"]

text_inputs = processor.process_texts(query)
image_inputs = processor.process_images(images)

# Move tensors to device
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

image_inputs['pixel_values'].requires_grad_(True)

query_embeddings = model(**text_inputs)          # shape: (1, embedding_dim)
doc_embeddings = model(**image_inputs)           # shape: (batch, embedding_dim)

score = torch.cosine_similarity(query_embeddings, doc_embeddings, dim=-1).sum()
score.backward()

grads = image_inputs['pixel_values'].grad.detach()  # (batch, 3, H, W)
grads_norm = grads.norm(dim=1)                      # (batch, H, W)

min_vals = grads_norm.amin(dim=(1, 2), keepdim=True)
max_vals = grads_norm.amax(dim=(1, 2), keepdim=True)
grads_norm = (grads_norm - min_vals) / (max_vals - min_vals + 1e-8)

def visualize_gradient(img_pil, grad_norm, save_path=None):
    img = np.array(img_pil).astype(np.float32) / 255.0
    H_img, W_img, _ = img.shape

    if grad_norm.dim() == 3:
        # (3, H, W) -> (H, W)
        grad_norm = grad_norm.norm(dim=0)

    grad_norm = grad_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    grad_norm_resized = F.interpolate(
        grad_norm,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False
    )

    grad_norm_resized = grad_norm_resized.squeeze().cpu().numpy()
    grad_norm_resized = (grad_norm_resized - grad_norm_resized.min()) / \
                        (grad_norm_resized.max() - grad_norm_resized.min() + 1e-8)

    heatmap_color = np.stack(
        [grad_norm_resized, np.zeros_like(grad_norm_resized), 1 - grad_norm_resized],
        axis=-1
    )

    overlay = 0.6 * img + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(12, 6))
    plt.imshow(overlay)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()

for i, img in enumerate(images):
    visualize_gradient(img, grads_norm[i], save_path=f"grad_heatmap_image_{i}.png")


# TODO: add matplotlib to requirements.txt
