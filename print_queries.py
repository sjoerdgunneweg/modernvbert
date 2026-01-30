from transformers import AutoProcessor, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from PIL import Image

from colpali.colpali_engine.models.modernvbert.sparse_mlm.processing_sparsemodernvbert_mlm import SparseModernVBertMLMProcessor
from colpali.colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM

processor = SparseModernVBertMLMProcessor.from_pretrained("/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint", trust_remote_code=True)
model = SparseModernVBertMLM.from_pretrained("/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint", trust_remote_code=True).eval()

images = [Image.open("photo_28.jpg").convert("RGB"),
          Image.open("photo_27.jpg").convert("RGB")]

query = [
    "Based on the RadQA context, which cervical spine levels show the most severe degenerative changes?", #NOTE: this is query 28
]

text_inputs = processor.process_texts(query).to(model.device)
image_inputs = processor.process_images(images).to(model.device)

with torch.no_grad():
    query_embeddings  = model(**text_inputs)
    doc_embeddings = model(**image_inputs)

print("doc embeddings: ", doc_embeddings)
print("query_embeddings: ", query_embeddings)

