from functools import partial
from typing import Optional

import datasets
import torch
from torch.distributed.nn.functional import all_gather  # PyTorch ≥ 2.1
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

from colpali_engine.data.sampler import SingleDatasetBatchSampler


def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(t), dim=0)  # keeps grad graph
    return t


def concat_datasets(datasets: list[Dataset], batch_size: int) -> Dataset:
    """
    Concatenates a list of datasets into a single dataset.
    This is a utility function to handle the case where multiple datasets are provided.
    """
    # round down each dataset if not divible by global batch size
    for i in range(len(datasets)):
        if len(datasets[i]) % batch_size != 0:
            total_samples = (len(datasets[i]) // batch_size) * batch_size
            datasets[i] = datasets[i].take(total_samples)

    return ConcatDataset(datasets)


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, compute_symetric_loss=False, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], list):
            train_dataset_list = kwargs["train_dataset"]
            kwargs["train_dataset"] = concat_datasets(train_dataset_list, batch_size=kwargs["args"].train_batch_size)
        else:
            train_dataset_list = None

        if isinstance(kwargs["eval_dataset"], list):
            eval_dataset_list = kwargs["eval_dataset"]
            kwargs["eval_dataset"] = concat_datasets(eval_dataset_list)
        else:
            eval_dataset_list = None

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.train_dataset_list = train_dataset_list
        self.eval_dataset_list = eval_dataset_list
        self.compute_symetric_loss = compute_symetric_loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.train_dataset_list is None:
            # If no dataset list, use the default behavior
            return super().get_train_dataloader()

        dataset = self.train_dataset
        description = "Training"
        sampler_fn = self._get_train_sampler
        is_training = True
        dataloader_key = None

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        self.query_prefix = data_collator.query_prefix
        self.pos_prefix = data_collator.pos_doc_prefix
        self.neg_prefix = data_collator.neg_doc_prefix

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                ###### batch_sampler set instead of sampler in trainer code #######
                dataloader_params["batch_sampler"] = sampler_fn()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset_list is None:
            return super()._get_train_sampler()

        # Use SingleDatasetBatchSampler to ensure that each dataset in the list is sampled independently
        # Note: Surely breaks in distributed training
        # TODO: fix this
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.train_dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    def _compute_loss_from_outputs(
        self,
        query_outputs,
        pos_target_outputs,
        neg_target_outputs=None,
    ):
        offset = 0
        batch_size = query_outputs.size(0)
        if self.accelerator.num_processes > 1 and self.accelerator.sync_gradients:
            # gather docs across all processes
            pos_target_outputs = self.accelerator.pad_across_processes(
                pos_target_outputs, dim=1, pad_index=0, pad_first=True
            )
            pos_target_outputs = concat_all_gather(pos_target_outputs)
            rank = self.accelerator.process_index
            offset = rank * batch_size

        if neg_target_outputs is not None:
            loss = self.loss_func(
                query_embeddings=query_outputs,
                doc_embeddings=pos_target_outputs,
                neg_doc_embeddings=neg_target_outputs,
                offset=offset,
            )
        else:
            loss = self.loss_func(query_embeddings=query_outputs, doc_embeddings=pos_target_outputs, offset=offset)

        if isinstance(loss, dict):
            return loss
        else:
            return {"loss": loss}

    def _reshape_neg_doc_inputs(self, inputs):
        """
        Helper function to reshape negative doc inputs to (batch_size * num_neg_docs, ...)
        """
        neg_doc_inputs = {k[len(self.neg_prefix) :]: v for k, v in inputs.items() if k.startswith(self.neg_prefix)}

        for k in neg_doc_inputs:
            # go from (batch_size, num_neg_docs, ...) to (batch_size * num_neg_docs, ...)
            neg_doc_inputs[k] = neg_doc_inputs[k].view(-1, *neg_doc_inputs[k].shape[2:])

        return neg_doc_inputs

    def _reshape_neg_doc_outputs(self, neg_doc_outputs, num_neg_docs):
        """
        Helper function to reshape negative doc outputs to (batch_size, num_neg_docs, ...)
        """
        neg_doc_outputs = neg_doc_outputs.view(-1, num_neg_docs, *neg_doc_outputs.shape[1:])

        return neg_doc_outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_class_name = model.__class__.__name__
        if not ((model_class_name == "SparseModernVBertM2") or (model_class_name == "SparseModernVBertM2SpladeModernBERT")):
            # === Extract inputs ===
            query_inputs = {k[len(self.query_prefix):]: v for k, v in inputs.items() if k.startswith(self.query_prefix)}
            doc_inputs   = {k[len(self.pos_prefix):]:   v for k, v in inputs.items() if k.startswith(self.pos_prefix)}

            query_outputs = model(**query_inputs)
            doc_outputs   = model(**doc_inputs)

            # === Hard negatives ===
            neg_doc_outputs = None
            if "neg_doc_input_ids" in inputs:
                num_negs = inputs["neg_doc_input_ids"].size(1)
                neg_doc_inputs = self._reshape_neg_doc_inputs(inputs)
                neg_doc_outputs = model(**neg_doc_inputs)
                neg_doc_outputs = self._reshape_neg_doc_outputs(neg_doc_outputs, num_negs)
        else:
            dict_outputs = model(**inputs)
            query_outputs = dict_outputs["q_out"]
            doc_outputs = dict_outputs["d_out"]
            neg_doc_outputs = dict_outputs["neg_d_out"]

        # === Main loss: query → doc ===
        loss_out = self._compute_loss_from_outputs(query_outputs, doc_outputs, neg_doc_outputs)

        # Normalize output to dict
        if not isinstance(loss_out, dict):
            loss_out = {"loss": loss_out}

        main_loss = loss_out["loss"]
        logs = {k: v for k, v in loss_out.items() if k != "loss"}

        final_loss = main_loss
        final_logs = logs.copy()
        # === Symmetric loss (doc → query) ===
        if self.compute_symetric_loss:
            assert neg_doc_outputs is None, "Symmetric loss not supported with hard negatives"
            sym_out = self._compute_loss_from_outputs(doc_outputs, query_outputs, None)
            if not isinstance(sym_out, dict):
                sym_out = {"loss": sym_out}
            sym_loss = sym_out["loss"]
            sym_logs = {f"sym_{k}": v for k, v in sym_out.items() if k != "loss"}

            final_loss = (main_loss + sym_loss) / 2
            final_logs.update(sym_logs)

        # Log the additional metrics (detached and converted to scalars for logging)
        logs_to_report = {k: v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in final_logs.items()}
        self.log(logs_to_report)

        # === Final return: scalar loss by default, or tuple if requested ===
        result = {"loss": final_loss, **final_logs}
        return result["loss"] if not return_outputs else (result["loss"], result)


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])

            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    #     if not prediction_loss_only:
    #         raise ValueError("This trainer only supports prediction_loss_only=True")

    #     with torch.no_grad():
    #         query_outputs = model(**{k[len(self.query_prefix):]: v for k, v in inputs.items() if k.startswith(self.query_prefix)})
    #         doc_outputs   = model(**{k[len(self.pos_prefix):]:   v for k, v in inputs.items() if k.startswith(self.pos_prefix)})

    #         if "neg_doc_input_ids" in inputs:
    #             num_negs = inputs["neg_doc_input_ids"].size(1)
    #             neg_inputs = self._reshape_neg_doc_inputs(inputs)
    #             neg_outputs = model(**neg_inputs)
    #             neg_outputs = self._reshape_neg_doc_outputs(neg_outputs, num_negs)
    #             loss_out = self.loss_func(query_outputs, doc_outputs, neg_outputs)
    #         else:
    #             neg_outputs = None
    #             loss_out = self.loss_func(query_outputs, doc_outputs)

    #         loss = loss_out if isinstance(loss_out, dict) else loss_out
    #         return (loss, None, None)