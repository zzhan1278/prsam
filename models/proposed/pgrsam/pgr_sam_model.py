import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pct_prior_encoder import PCTPriorEncoder
from .correction_network import CorrectionNetwork

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class PRSamModel(nn.Module):
    def __init__(self,
                 sam_model_type: str = "vit_b",
                 sam_checkpoint_path: str = None,
                 n_prior_input_atlas_channels: int = 3,
                 n_prior_output_channels: int = 256,
                 n_correct_ecbct_channels: int = 256,
                 n_correct_eprior_channels: int = 256,
                 n_correct_output_delta_e_channels: int = 256,
                 target_feature_spatial_size=(64, 64),
                 device: str = 'cuda'):
        super().__init__()

        self.device = device

        if sam_checkpoint_path is None:
            raise ValueError("SAM checkpoint path must be provided.")
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint_path}")

        # Load and freeze SAM
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        for p in self.sam_model.parameters():
            p.requires_grad = False

        self.sam_img_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        # N_prior
        self.n_prior = PCTPriorEncoder(
            input_atlas_channels=n_prior_input_atlas_channels,
            output_prior_channels=n_prior_output_channels,
            target_spatial_size=target_feature_spatial_size
        ).to(self.device)

        # Correction network (rollback to original fusion)
        self.n_correct = CorrectionNetwork(
            ecbct_channels=n_correct_ecbct_channels,
            eprior_channels=n_correct_eprior_channels,
            output_delta_e_channels=n_correct_output_delta_e_channels
        ).to(self.device)
        # CBCT adapter (PR-SAM+): small bottleneck to preserve CBCT-specific info
        self.cbct_adapter = nn.Sequential(
            nn.Conv2d(n_correct_ecbct_channels, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_correct_ecbct_channels, kernel_size=1, bias=False)
        ).to(self.device)

        # Simple CPU float16 cache for E_cbct to speed up repeated usage
        self.emb_cache = {}
        self.cache_max_items = 20000

        # No refine head and no point prompts in rollback version

    def forward(self,
                cbct_image: torch.Tensor,
                pct_atlas_image: torch.Tensor,
                bbox_prompts_xyxy: torch.Tensor = None,
                original_image_size_hw: torch.Tensor = None,
                multimask_output: bool = False,
                sample_ids: torch.Tensor = None):

        batch_size = cbct_image.shape[0]

        # Ensure inputs are on the same device as SAM
        if cbct_image.device != torch.device(self.device):
            cbct_image = cbct_image.to(self.device)
        if pct_atlas_image.device != torch.device(self.device):
            pct_atlas_image = pct_atlas_image.to(self.device)

        # SAM encoder on CBCT with caching
        with torch.no_grad():
            # Try cache per sample; fallback to compute
            E_cbct_list = []
            use_cache = sample_ids is not None
            if use_cache:
                for i in range(batch_size):
                    sid = int(sample_ids[i].item())
                    if sid in self.emb_cache:
                        # retrieve from CPU fp16 cache and move to device fp32
                        cached = self.emb_cache[sid].to(self.device, dtype=torch.float32)
                        E_cbct_list.append(cached)
                    else:
                        cbct_uint8_rgb_i = (cbct_image[i:i+1].clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1).to(self.device)
                        processed_cbct_i = self.sam_model.preprocess(cbct_uint8_rgb_i)
                        e_i = self.sam_model.image_encoder(processed_cbct_i)
                        E_cbct_list.append(e_i)
                        # save to cache as CPU fp16
                        if len(self.emb_cache) >= self.cache_max_items:
                            # remove arbitrary first item
                            self.emb_cache.pop(next(iter(self.emb_cache)))
                        self.emb_cache[sid] = e_i.detach().cpu().to(torch.float16)
                E_cbct = torch.cat(E_cbct_list, dim=0)
            else:
                cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1).to(self.device)
                processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
                E_cbct = self.sam_model.image_encoder(processed_cbct)

        # N_prior on plan CT
        # pct_atlas_image should be 3-channel prior features derived from CT mask
        E_prior = self.n_prior(pct_atlas_image)

        # Correction fusion: E_fused = E_prior + Delta_E
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E
        E_cbct_adapt = self.cbct_adapter(E_cbct)

        image_pe = self.sam_model.prompt_encoder.get_dense_pe().expand(batch_size, -1, -1, -1)

        sparse_embeddings = None
        dense_embeddings = None
        if bbox_prompts_xyxy is not None and original_image_size_hw is not None:
            transformed = []
            for i in range(batch_size):
                h, w = int(original_image_size_hw[i, 0].item()), int(original_image_size_hw[i, 1].item())
                box = bbox_prompts_xyxy[i].unsqueeze(0)
                box_1024 = self.sam_img_transform.apply_boxes_torch(box, (h, w))
                transformed.append(box_1024)
            input_boxes = torch.cat(transformed, dim=0).to(self.device)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=input_boxes, masks=None)
        else:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=None, masks=None)

        # Use dense embeddings as-is (match batch size)
        expanded_dense = dense_embeddings

        pred_masks_logits_fused, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=E_fused,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        # Second decode on CBCT-only branch
        pred_masks_logits_cbct, _ = self.sam_model.mask_decoder(
            image_embeddings=E_cbct_adapt,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        return pred_masks_logits_fused, iou_predictions, E_cbct, E_prior, delta_E, E_fused, pred_masks_logits_cbct


