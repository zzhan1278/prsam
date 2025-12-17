"""
PR-SAM+ and its Ablation Study Variants (Fixed for batch processing)
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
    Fixed to handle batch processing correctly.
    """
    def __init__(self, sam_model_type="vit_b", sam_checkpoint_path=None, device='cuda'):
        super().__init__()
        self.device = device
        
        if sam_checkpoint_path is None or not os.path.exists(sam_checkpoint_path):
            print(f"Warning: SAM checkpoint not found at {sam_checkpoint_path}. SAM will be initialized with random weights.")
            sam_checkpoint_path = None

        # Load SAM (either pre-trained or random weights)
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        for p in self.sam_model.parameters():
            p.requires_grad = False

        self.sam_img_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def get_sam_image_embeddings(self, cbct_image: torch.Tensor):
        """Get SAM image embeddings for batch of images"""
        with torch.no_grad():
            cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1)
            processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
            return self.sam_model.image_encoder(processed_cbct)

    def decode_masks_batch(self, image_embeddings, bbox_prompts_xyxy, original_image_size_hw):
        """
        Decode masks for a batch by processing each sample individually.
        This is necessary because SAM's mask decoder doesn't support batch processing with different prompts.
        """
        batch_size = image_embeddings.shape[0]
        masks_list = []
        
        for i in range(batch_size):
            # Get embeddings for this sample
            img_emb_i = image_embeddings[i:i+1]
            
            # Get prompt embeddings for this sample
            if bbox_prompts_xyxy is not None and original_image_size_hw is not None:
                h, w = int(original_image_size_hw[i, 0].item()), int(original_image_size_hw[i, 1].item())
                box = bbox_prompts_xyxy[i:i+1]  # Keep batch dimension [1, 4]
                box_1024 = self.sam_img_transform.apply_boxes_torch(box, (h, w))
                
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None, boxes=box_1024, masks=None
                    )
            else:
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None, boxes=None, masks=None
                    )
            
            # Decode mask for this sample
            image_pe = self.sam_model.prompt_encoder.get_dense_pe()
            
            pred_masks_logits, _ = self.sam_model.mask_decoder(
                image_embeddings=img_emb_i,
                image_pe=image_pe,  # Keep as [1, C, H, W]
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            masks_list.append(pred_masks_logits)
        
        # Stack all masks back into a batch
        return torch.cat(masks_list, dim=0)

    def forward(self, cbct_image, pct_atlas_image, bbox_prompts_xyxy, original_image_size_hw):
        raise NotImplementedError


# --- Ablation Model Implementations ---

class Ablation_Prior_Guided_Only(PRSAM_Ablation_Base):
    """
    Model 1: Prior-Guided Only.
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
        
        mask_logits = self.decode_masks_batch(E_fused, bbox_prompts_xyxy, original_image_size_hw)
        
        return mask_logits, None  # Single output


class Ablation_No_Correction_Network(PRSAM_Ablation_Base):
    """
    Model 2: No Correction Network (Simple Fusion).
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
        
        mask_logits_fused = self.decode_masks_batch(E_fused, bbox_prompts_xyxy, original_image_size_hw)
        mask_logits_cbct = self.decode_masks_batch(E_cbct_adapt, bbox_prompts_xyxy, original_image_size_hw)

        return mask_logits_fused, mask_logits_cbct  # Dual output


class Ablation_No_CBCT_Adapter(PRSAM_Ablation_Base):
    """
    Model 3: No CBCT Adapter.
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
        
        mask_logits_fused = self.decode_masks_batch(E_fused, bbox_prompts_xyxy, original_image_size_hw)
        mask_logits_cbct = self.decode_masks_batch(E_cbct_raw, bbox_prompts_xyxy, original_image_size_hw)

        return mask_logits_fused, mask_logits_cbct  # Dual output


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
        
        mask_logits_fused = self.decode_masks_batch(E_fused, bbox_prompts_xyxy, original_image_size_hw)
        mask_logits_cbct = self.decode_masks_batch(E_cbct_adapt, bbox_prompts_xyxy, original_image_size_hw)

        return mask_logits_fused, mask_logits_cbct  # Dual output
