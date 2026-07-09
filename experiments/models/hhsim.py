"""HHSIM Wavelet-based model from temp.py."""
import torch
import torch.nn as nn
import cv2
import pywt
import numpy as np

from ..registry import DefaultExperiment, register_experiment

# --- Algorithm code from temp.py (untouched) ---
def IQA_WS_HV(r_in, d_in):
        M, N = r_in.shape[:2]

        f = min(3, round((max(M, N) / 512) + 5e-16));  # this is added to
        aa = 10 * 2 ** (
            f);  # This is going to regulate the constant in the similarity relation. We see that the constant is determied based on the dimension of the images.

        r = r_in;
        d = d_in;
        T4 = 2 * aa;
        T9 = 2 * aa;
        T8 = aa;
        T5 = aa;
        T6 = 3 * aa;
        T7 = 3 * aa;

        if len(r.shape) > 2 and r.shape[2] > 1:  # this check should be omitted for evaluating the time complexity.
            r = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)

        rcA, (rcH, rcV, rcD) = pywt.dwt2(r, 'haar')
        rcA2, (rcH2, rcV2, rcD2) = pywt.dwt2(rcA, 'haar')
        _, (rcH3, rcV3, _) = pywt.dwt2(rcA2, 'haar')

        dcA, (dcH, dcV, _) = pywt.dwt2(d, 'haar')
        dcA2, (dcH2, dcV2, dcD2) = pywt.dwt2(dcA, 'haar')
        _, (dcH3, dcV3, _) = pywt.dwt2(dcA2, 'haar')

        rHH1 = rcD[:, :]
        rh1 = rcH[:, :]
        rv1 = rcV[:, :]
        dh1 = dcH[:, :]
        dv1 = dcV[:, :]

        rHH2 = cv2.resize(rcD2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rh2 = cv2.resize(rcH2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rv2 = cv2.resize(rcV2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dHH2 = cv2.resize(dcD2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dh2 = cv2.resize(dcH2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dv2 = cv2.resize(dcV2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        dh3 = cv2.resize(dcH3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dv3 = cv2.resize(dcV3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        rh3 = cv2.resize(rcH3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rv3 = cv2.resize(rcV3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dh3 = cv2.resize(dh3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dv3 = cv2.resize(dv3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rh3 = cv2.resize(rh3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rv3 = cv2.resize(rv3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        a, b = rHH1.shape
        rh3 = cv2.resize(rh3, (b, a))
        rv3 = cv2.resize(rv3, (b, a))
        dh3 = cv2.resize(dh3, (b, a))
        dv3 = cv2.resize(dv3, (b, a))
        rh1 = abs(rh1);
        rv1 = abs(rv1);
        dh1 = abs(dh1);
        dv1 = abs(dv1);
        rHH2 = abs(rHH2);
        rh2 = abs(rh2);
        rv2 = abs(rv2);
        dHH2 = abs(dHH2);
        dh2 = abs(dh2);
        dv2 = abs(dv2);
        rh3 = abs(rh3);
        rv3 = abs(rv3);
        dh3 = abs(dh3);
        dv3 = abs(dv3);

        h_map3 = (2 * rh3 * dh3 + T4) / (rh3 * rh3 + dh3 * dh3 + T4);
        h_map2 = (2 * rh2 * dh2 + T5) / (rh2 * rh2 + dh2 * dh2 + T5);
        h_map1 = (2 * rh1 * dh1 + T6) / (rh1 * rh1 + dh1 * dh1 + T6);
        v_map1 = (2 * rv1 * dv1 + T7) / (rv1 * rv1 + dv1 * dv1 + T7);
        v_map2 = (2 * rv2 * dv2 + T8) / (rv2 * rv2 + dv2 * dv2 + T8);
        v_map3 = (2 * rv3 * dv3 + T9) / (rv3 * rv3 + dv3 * dv3 + T9);

        HH3m = np.maximum(rHH2, dHH2)

        min_h = np.min([h_map2.shape[0], h_map3.shape[0], h_map1.shape[0], v_map1.shape[0], v_map2.shape[0], v_map3.shape[0], HH3m.shape[0]])
        min_w = np.min([h_map2.shape[1], h_map3.shape[1], h_map1.shape[1], v_map1.shape[1], v_map2.shape[1], v_map3.shape[1], HH3m.shape[1]])
        
        h_map2= h_map2[:min_h,:min_w]
        h_map3= h_map3[:min_h,:min_w]
        h_map1= h_map1[:min_h,:min_w]
        v_map1= v_map1[:min_h,:min_w]
        v_map2= v_map2[:min_h,:min_w]
        v_map3= v_map3[:min_h,:min_w]
        HH3m= HH3m[:min_h,:min_w]

        SimMatrix = h_map2 * h_map3 * h_map1 * v_map1 * v_map2 * v_map3 * HH3m
        if sum(sum(SimMatrix)) == 0:
            HHSIM = 0
        else:
            HHSIM = sum(sum(SimMatrix)) / sum(sum(HH3m))

        return HHSIM
# -----------------------------------------------

class HHSIMModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ref, dist):
        # The dataloader gives us tensors of shape (B, C, H, W) normalized to [0, 1].
        # We need to convert them to numpy arrays of shape (H, W, C) in [0, 255] for cv2/pywt.
        # Assuming batch size 1 for evaluation.
        
        if ref.dim() == 4:
            ref = ref.squeeze(0)
        if dist.dim() == 4:
            dist = dist.squeeze(0)
            
        # Convert to numpy and transpose to HWC
        r_np = ref.permute(1, 2, 0).cpu().numpy()
        d_np = dist.permute(1, 2, 0).cpu().numpy()
        
        # Scale back to 0-255 range and convert to uint8 or float32. 
        # OpenCV cvtColor works with uint8 or float32, here we use float32.
        r_np = (r_np * 255.0).astype(np.float32)
        d_np = (d_np * 255.0).astype(np.float32)

        # Call the standalone algorithm
        score = IQA_WS_HV(r_np, d_np)
        
        # Return as a tensor so it matches standard output expectations
        return torch.tensor(score, dtype=torch.float32, device=ref.device)


@register_experiment
class HHSIMExperiment(DefaultExperiment):
    name = "hhsim"
    description = "Wavelet-based IQA model (HHSIM)"
    summary_prefix = "hhsim"

    def add_arguments(self, parser):
        # No special arguments needed since there's no backbone to select
        pass

    def build_model(self, device, args):
        return HHSIMModel().to(device)

    def slug_args(self, args):
        # Return a simple dict so the run_slug function creates a simple, clean filename
        return {"backbone": "none", "feature_layer": "none"}
