from transformers import AutoProcessor, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from PIL import Image

from colpali.colpali_engine.models.modernvbert.sparse_mlm.processing_sparsemodernvbert_mlm import SparseModernVBertMLMProcessor
from colpali.colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM

processor = SparseModernVBertMLMProcessor.from_pretrained("/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint", trust_remote_code=True)
model = SparseModernVBertMLM.from_pretrained("/home/scur1716/modernvbert/models/sparsemodernvbertmlm_final_checkpoint", trust_remote_code=True).eval()

# images = [Image.open("photo_28.jpg").convert("RGB"),
#           Image.open("photo_27.jpg").convert("RGB")]
images = [Image.open("photo_28.jpg").convert("RGB")]


query = [
    "Based on the RadQA context, which cervical spine levels show the most severe degenerative changes?", #NOTE: this is query 28
]

text_inputs = processor.process_texts(query).to(model.device)
image_inputs = processor.process_images(images).to(model.device)

with torch.no_grad():
    query_embeddings  = model(**text_inputs)
    doc_embeddings = model(**image_inputs)

print("doc embeddings: ", doc_embeddings)
print("query_embeddings: ", query_embeddings, "\n")

print("doc embeddings nonzero indices: ", torch.nonzero(doc_embeddings))
print("query embeddings nonzero indices: ", torch.nonzero(query_embeddings))


query_tokens = processor.tokenizer.convert_ids_to_tokens(torch.nonzero(query_embeddings).squeeze()[:, -1].cpu().numpy())
print("tokens with nonzero values in query embedding: ", query_tokens)

doc_tokens = processor.tokenizer.convert_ids_to_tokens(torch.nonzero(doc_embeddings).squeeze()[:, -1].cpu().numpy())
print("tokens with nonzero values in doc embedding: ", doc_tokens)


# TODO check if this puts the embeddings of the two queries together
# tokens with nonzero values in query embedding:  ['A', 'Ġnew', 'Ġlittle', 'As', 'Ġmain', 'Ġtarget', 'Id', 'ĠThen', 'ips', 'Ġpatient', 'Ġproblems', 'Ġknowledge', 'ĠWar', 'ĠCounty', 'ĠArt', 'Ġoperation', 'Ġactual', 'index', 'Ġfeet', 'Ġfiled', 'Ġmeeting', 'rant', 'çļĦ', 'Ġvisual', 'Ġtransition', 'Ä±', 'Ġfirm', 'ĠChristian', 'Ġreality', 'Ġexcess', 'FS', 'Page', 'ĠMs', 'Ġvoters', 'Ġbasket', 'Ġviewed', 'ĠSteve', 'Ġdestination', 'Ġunexpected', 'Ġmath', 'Ġpoison', 'ĠRick', 'Ġholder', 'Display', 'Ġritual', 'ĠDuncan', 'ĠAsp', 'Ġteens', 'Ġvault', 'Ġbees', 'ĠDrew', 'wang', '510', 'Ġcrawl', 'ĠJiang', 'ĠRaven', 'ĠTales', 'riet', 'Ġtyranny']
# tokens with nonzero values in doc embedding:  ['%', 'A', 'F', 'X', 'ity', 'Ġdata', 'Ġmodel', 'Ġlittle', 'As', 'Ġtarget', 'Id', 'ips', 'Ġpatient', 'Ġproblems', 'Ġknowledge', 'ĠCounty', 'Ġoperation', 'Ġcomment', 'Ġactual', 'index', 'Ġfeet', 'Ġmeeting', 'Ġtask', 'čĊč', 'çļĦ', 'Ġvisual', 'Ġtransition', 'Ä±', 'Ġfirm', 'Ġreality', 'Ġexcess', 'FS', 'Ġvon', 'Ġcrisis', 'Page', 'ĠMs', 'Ġvoters', 'Ġbasket', 'Ġviewed', 'Ġscattering', 'ĠSteve', 'Ġdestination', 'Ġunexpected', 'Ġmath', 'Ġpoison', 'ĠRick', 'Ġholder', 'Display', 'Ġritual', 'named', 'Lib', 'central', 'ĠDuncan', 'Ġteens', 'Ġvault', 'Ġbees', 'ĠDrew', 'wang', '510', 'ĠMask', 'Ġcrawl', 'ĠJiang', 'ĠRaven', 'Ġboo', 'Ġfollower', 'X', 'ity', 'ft', 'Ġdata', 'Ġlittle', 'Ġmain', 'Ġtarget', 'Ġuser', 'Ġland', 'Ġpatient', 'ĠWar', 'Ġoperation', 'Ġactual', 'Ġfeet', 'Ġtask', 'čĊč', 'çļĦ', 'Ġvisual', 'Ġtransition', 'Ä±', 'Ġfirm', 'Ġexcess', 'Ġvon', 'Page', 'Ġbasket', 'Ġviewed', 'Ġscattering', 'ĠSteve', 'Ġdestination', 'Ġunexpected', 'Ġmath', 'Ġpoison', 'ĠRick', 'named', 'central', 'Ġvault', 'Ġbees', 'ĠDrew', 'wang', '510', 'Ġcrawl', 'ĠJiang', 'ĠRaven', 'Ġboo', 'Ġfollower']


