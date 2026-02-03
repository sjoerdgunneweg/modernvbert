import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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

text_inputs = processor.process_texts(query).to(device)
image_inputs = processor.process_images(images).to(device)

outputs = model(**text_inputs, **image_inputs, output_attentions=True)


if hasattr(outputs, "cross_attentions"):
    cross_attentions = outputs.cross_attentions  # list of tensors
else:
    raise RuntimeError("Model does not return cross-attention. Check the forward pass or enable output_attentions.")


attn_weights = torch.stack(cross_attentions).mean(0).mean(1)  # mean over layers and heads
attn_weights = attn_weights[:, 0, :]  # take first query token as representative

def visualize_attention(img_pil, patch_attn, patch_size=16, save_path=None):
    img = np.array(img_pil)
    H, W, _ = img.shape

    num_patches = patch_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    attn_grid = patch_attn.reshape(grid_size, grid_size).cpu().numpy()

    heatmap = cv2.resize(attn_grid, (W, H))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # normalize

    overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()

for i, img in enumerate(images):
    visualize_attention(img, attn_weights[i], save_path=f"attention_visualization_image_{i}.png")

# TODO: add cv2 (opencv-python), and matplotlib to requirements.txt
