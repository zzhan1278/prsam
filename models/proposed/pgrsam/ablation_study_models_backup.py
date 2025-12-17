"""
PR-SAM+ and its Ablation Study Variants
"""

import os
import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from .pct_prior_encoder import PCTPriorEncoder
from .correction_network import CorrectionNetwork


class PRSAM_Ablation_Base(nn.Module):
    """
    Base class for PR-SAM models to share common components.
    """
    def __init__(self, sam_model_type="vit_b", sam_checkpoint_path=None, device='cuda'):
        super().__init__()
        self.device = device
        
        if sam_checkpoint_path is None or not os.path.exists(sam_checkpoint_path):
            # Allow training without a pre-trained model for debugging or from-scratch training
            print(f"Warning: SAM checkpoint not found at {sam_checkpoint_path}. SAM will be initialized with random weights.")
            sam_checkpoint_path = None # Set to None to allow registry to handle random init

        # Load SAM (either pre-trained or random weights)
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        for p in self.sam_model.parameters():
            p.requires_grad = False

        self.sam_img_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def get_sam_image_embeddings(self, cbct_image: torch.Tensor):
        with torch.no_grad():
            cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1)
            processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
            return self.sam_model.image_encoder(processed_cbct)

    def get_prompt_embeddings(self, bbox_prompts_xyxy, original_image_size_hw):
        with torch.no_grad():
            batch_size = bbox_prompts_xyxy.shape[0] if bbox_prompts_xyxy is not None else 1
            
            if bbox_prompts_xyxy is None or original_image_size_hw is None:
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=None, masks=None)
            else:
                transformed_boxes = []
                for i in range(bbox_prompts_xyxy.shape[0]):
                    h, w = int(original_image_size_hw[i, 0].item()), int(original_image_size_hw[i, 1].item())
                    box_1024 = self.sam_img_transform.apply_boxes_torch(bbox_prompts_xyxy[i].unsqueeze(0), (h, w))
                    transformed_boxes.append(box_1024)
                
                input_boxes = torch.cat(transformed_boxes, dim=0)
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=input_boxes, masks=None)
            
            # Ensure correct batch size for embeddings
            if sparse_embeddings.shape[0] != batch_size:
                sparse_embeddings = sparse_embeddings.expand(batch_size, -1, -1)
            if dense_embeddings.shape[0] != batch_size:
                dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
                
            return sparse_embeddings, dense_embeddings

    def decode_mask(self, image_embeddings, sparse_embeddings, dense_embeddings):
        batch_size = image_embeddings.shape[0]
        image_pe = self.sam_model.prompt_encoder.get_dense_pe()
        
        # Ensure dense_embeddings has correct batch size
        if dense_embeddings.shape[0] != batch_size:
            dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
        
        pred_masks_logits, _ = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe.expand(batch_size, -1, -1, -1),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return pred_masks_logits

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        raise NotImplementedError


# --- Ablation Model Implementations ---

class Ablation_CBCT_Only(PRSAM_Ablation_Base):
    """
    Model 1: CBCT-Only Branch with Adapter.
    Removes all prior-guidance. Tests the baseline performance of SAM + Adapter on this task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Only the CBCT adapter is trainable
        self.cbct_adapter = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, bias=False)
        ).to(self.device)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        E_cbct = self.get_sam_image_embeddings(cbct_image)
        E_cbct_adapt = self.cbct_adapter(E_cbct)
        
        sparse_embeds, dense_embeds = self.get_prompt_embeddings(bbox_prompts_xyxy, original_image_size_hw)
        mask_logits = self.decode_mask(E_cbct_adapt, sparse_embeds, dense_embeds)
        
        return mask_logits, None # Single output


class Ablation_Prior_Guided_Only(PRSAM_Ablation_Base):
    """
    Model 2: Prior-Guided Only.
    Removes the CBCT-only branch. Tests the performance of the core guidance mechanism.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_prior = PCTPriorEncoder(input_atlas_channels=3).to(self.device)  # 3-channel prior
        self.n_correct = CorrectionNetwork().to(self.device)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        E_cbct = self.get_sam_image_embeddings(cbct_image)
        E_prior = self.n_prior(pct_atlas_image)
        
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E
        
        sparse_embeds, dense_embeds = self.get_prompt_embeddings(bbox_prompts_xyxy, original_image_size_hw)
        mask_logits = self.decode_mask(E_fused, sparse_embeds, dense_embeds)
        
        return mask_logits, None # Single output


class Ablation_No_Correction_Network(PRSAM_Ablation_Base):
    """
    Model 3: No Correction Network (Simple Fusion).
    Replaces the learned CorrectionNetwork with simple feature addition.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_prior = PCTPriorEncoder(input_atlas_channels=3).to(self.device)  # 3-channel prior
        self.cbct_adapter = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, bias=False)
        ).to(self.device)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        E_cbct = self.get_sam_image_embeddings(cbct_image)
        E_prior = self.n_prior(pct_atlas_image)
        
        # Simple fusion instead of learned correction
        E_fused = E_prior + E_cbct

        E_cbct_adapt = self.cbct_adapter(E_cbct)
        
        sparse_embeds, dense_embeds = self.get_prompt_embeddings(bbox_prompts_xyxy, original_image_size_hw)
        
        mask_logits_fused = self.decode_mask(E_fused, sparse_embeds, dense_embeds)
        mask_logits_cbct = self.decode_mask(E_cbct_adapt, sparse_embeds, dense_embeds)

        return mask_logits_fused, mask_logits_cbct # Dual output


class Ablation_No_CBCT_Adapter(PRSAM_Ablation_Base):
    """
    Model 4: No CBCT Adapter.
    The CBCT-only branch uses raw SAM features. Tests the adapter's utility.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_prior = PCTPriorEncoder(input_atlas_channels=3).to(self.device)  # 3-channel prior
        self.n_correct = CorrectionNetwork().to(self.device)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        E_cbct = self.get_sam_image_embeddings(cbct_image)
        E_prior = self.n_prior(pct_atlas_image)
        
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E
        
        # No adapter on the CBCT branch
        E_cbct_raw = E_cbct
        
        sparse_embeds, dense_embeds = self.get_prompt_embeddings(bbox_prompts_xyxy, original_image_size_hw)
        
        mask_logits_fused = self.decode_mask(E_fused, sparse_embeds, dense_embeds)
        mask_logits_cbct = self.decode_mask(E_cbct_raw, sparse_embeds, dense_embeds)

        return mask_logits_fused, mask_logits_cbct # Dual output


class PRSAM_Plus_Full(PRSAM_Ablation_Base):
    """
    The complete PR-SAM+ model with all components.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_prior = PCTPriorEncoder(input_atlas_channels=3).to(self.device)  # 3-channel prior
        self.n_correct = CorrectionNetwork().to(self.device)
        self.cbct_adapter = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, bias=False)
        ).to(self.device)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        E_cbct = self.get_sam_image_embeddings(cbct_image)
        E_prior = self.n_prior(pct_atlas_image)
        
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E
        
        E_cbct_adapt = self.cbct_adapter(E_cbct)
        
        sparse_embeds, dense_embeds = self.get_prompt_embeddings(bbox_prompts_xyxy, original_image_size_hw)
        
        mask_logits_fused = self.decode_mask(E_fused, sparse_embeds, dense_embeds)
        mask_logits_cbct = self.decode_mask(E_cbct_adapt, sparse_embeds, dense_embeds)

        return mask_logits_fused, mask_logits_cbct # Dual output
