import tqdm
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
import random
import cv2

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)

@torch.inference_mode()
@torch.cuda.amp.autocast()
def matanyone(processor, frames_np, mask, mask_adjustment=0, n_warmup=10, device="cuda", progress_callback=None):
    """
    Args:
        frames_np: [(H,W,C)]*n, uint8
        mask: (H,W), uint8
        progress_callback: Optional callback function(current_frame, total_frames, stage)
    Outputs:
        com: [(H,W,C)]*n, uint8
        pha: [(H,W,C)]*n, uint8
    """

    # print(f'===== [mask_adjustment] {mask_adjustment} =====')
    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3))
    objects = [1]

    # [optional] adjust mask edges based on single parameter
    if mask_adjustment > 0:
        # Positive values: dilate (expand mask)
        mask = gen_dilate(mask, mask_adjustment, mask_adjustment)
    elif mask_adjustment < 0:
        # Negative values: erode (shrink mask)
        mask = gen_erosion(mask, abs(mask_adjustment), abs(mask_adjustment))

    mask = torch.from_numpy(mask).to(device)

    frames_np = [frames_np[0]]* n_warmup + frames_np

    frames = []
    phas = []
    total_frames = len(frames_np)
    actual_frames = total_frames - n_warmup  # Frames that will be in output
    
    for ti, frame_single in enumerate(frames_np):
        image = to_tensor(frame_single).to(device).float()

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)      # encode given mask
            output_prob = processor.step(image, first_frame_pred=True)      # clear past memory for warmup frames
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)  # clear past memory for warmup frames
            else:
                output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)

        pha = mask.unsqueeze(2).cpu().numpy()
        com_np = frame_single / 255. * pha + bgr * (1 - pha)
        
        # DONOT save the warmup frames
        if ti > (n_warmup-1):
            frames.append((com_np*255).astype(np.uint8))
            phas.append((pha*255).astype(np.uint8))
            
            # Call progress callback for actual processed frames
            if progress_callback:
                current_frame = ti - n_warmup + 1
                progress_callback(current_frame, actual_frames)
    
    return frames, phas
