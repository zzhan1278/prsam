"""
PR-SAM Ablation Study Models
消融实验模型变体实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pct_prior_encoder import PCTPriorEncoder
from .correction_network import CorrectionNetwork

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class PRSamBaseline(nn.Module):
    """
    基础PR-SAM（无CBCT Adapter）
    只包含Prior编码器和Correction Network
    """
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

        # Correction network
        self.n_correct = CorrectionNetwork(
            ecbct_channels=n_correct_ecbct_channels,
            eprior_channels=n_correct_eprior_channels,
            output_delta_e_channels=n_correct_output_delta_e_channels
        ).to(self.device)
        
        # NO CBCT adapter in baseline

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

        # SAM encoder on CBCT
        with torch.no_grad():
            cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1).to(self.device)
            processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
            E_cbct = self.sam_model.image_encoder(processed_cbct)

        # N_prior on plan CT
        E_prior = self.n_prior(pct_atlas_image)

        # Correction fusion: E_fused = E_prior + Delta_E
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E

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

        expanded_dense = dense_embeddings

        # Single decode (no dual branch in baseline)
        pred_masks_logits, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=E_fused,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        
        # Return single mask (not dual)
        return pred_masks_logits, iou_predictions, E_cbct, E_prior, delta_E, E_fused, None


class PRSamNoCorrectionNetwork(nn.Module):
    """
    PR-SAM without Correction Network
    直接使用Prior特征，不进行修正
    """
    def __init__(self,
                 sam_model_type: str = "vit_b",
                 sam_checkpoint_path: str = None,
                 n_prior_input_atlas_channels: int = 3,
                 n_prior_output_channels: int = 256,
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
        
        # NO correction network - use prior directly

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

        # N_prior on plan CT
        E_prior = self.n_prior(pct_atlas_image)
        
        # Directly use E_prior without correction
        E_fused = E_prior

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

        expanded_dense = dense_embeddings

        pred_masks_logits, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=E_fused,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        
        # Return with None for unused components
        return pred_masks_logits, iou_predictions, None, E_prior, None, E_fused, None


class PRSamPriorOnly(nn.Module):
    """
    Prior-Only branch
    只使用Prior分支进行解码，完全不使用CBCT图像信息
    """
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

        # Correction network
        self.n_correct = CorrectionNetwork(
            ecbct_channels=n_correct_ecbct_channels,
            eprior_channels=n_correct_eprior_channels,
            output_delta_e_channels=n_correct_output_delta_e_channels
        ).to(self.device)

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

        # SAM encoder on CBCT
        with torch.no_grad():
            cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1).to(self.device)
            processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
            E_cbct = self.sam_model.image_encoder(processed_cbct)

        # N_prior on plan CT
        E_prior = self.n_prior(pct_atlas_image)

        # Correction fusion: E_fused = E_prior + Delta_E
        delta_E = self.n_correct(E_cbct, E_prior)
        E_fused = E_prior + delta_E

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

        expanded_dense = dense_embeddings

        # Only use Prior-guided branch
        pred_masks_logits, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=E_fused,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        
        return pred_masks_logits, iou_predictions, E_cbct, E_prior, delta_E, E_fused, None


class PRSamCBCTOnly(nn.Module):
    """
    CBCT-Only branch with adapter
    只使用CBCT分支进行解码，不使用Prior信息
    """
    def __init__(self,
                 sam_model_type: str = "vit_b",
                 sam_checkpoint_path: str = None,
                 n_cbct_channels: int = 256,
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

        # CBCT adapter only
        self.cbct_adapter = nn.Sequential(
            nn.Conv2d(n_cbct_channels, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_cbct_channels, kernel_size=1, bias=False)
        ).to(self.device)

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

        # SAM encoder on CBCT
        with torch.no_grad():
            cbct_uint8_rgb = (cbct_image.clamp(0, 1) * 255).byte().repeat(1, 3, 1, 1).to(self.device)
            processed_cbct = self.sam_model.preprocess(cbct_uint8_rgb)
            E_cbct = self.sam_model.image_encoder(processed_cbct)

        # Apply CBCT adapter
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

        expanded_dense = dense_embeddings

        # Only use CBCT branch
        pred_masks_logits, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=E_cbct_adapt,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=expanded_dense,
            multimask_output=multimask_output
        )
        
        return pred_masks_logits, iou_predictions, E_cbct, None, None, E_cbct_adapt, None
