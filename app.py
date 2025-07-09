import sys
sys.path.append("../")
sys.path.append("../../")

import os
import json
import time
import psutil
import ffmpeg
import imageio
import argparse
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import cv2
import torch
import numpy as np
import gradio as gr
 
from tools.painter import mask_painter
from tools.interact_tools import SamControler
# from tools.misc import get_device
from tools.download_util import load_file_from_url

from matanyone_wrapper import matanyone
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.inference.inference_core import InferenceCore

import warnings
warnings.filterwarnings("ignore")

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default=None, help="GPU ID to use (e.g., 0, 1, 2)")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")  
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()
    
    if not args.device:
        # bypass misc.py and implement device selection directly
        gpu_str = ''
        
        # Validate gpu_id is a valid integer if provided
        if args.gpu_id is not None:
            try:
                gpu_id_int = int(args.gpu_id)
                gpu_str = f':{gpu_id_int}'
            except ValueError:
                print(f"Warning: Invalid GPU ID '{args.gpu_id}'. Expected an integer. Using default GPU.")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = "mps" + gpu_str
        elif torch.cuda.is_available() and torch.backends.cudnn.is_available():
            args.device = "cuda" + gpu_str
        else:
            args.device = "cpu"

    return args

# SAM generator
class MaskGenerator():
    def __init__(self, sam_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)
       
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    
# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video - Updated for File input

def process_video_frames_only(video_file, video_state, progress=gr.Progress()):
    """
    Extract frames from video - focused function that returns only video_state
    """
    if not video_file:
        return video_state
    
    # Check if uploaded file is a valid video
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
        gr.Warning("Please upload a valid video file (.mp4, .avi, .mov, .mkv, .webm)")
        return video_state
    
    # Extract frames using existing logic
    video_state, frames_extracted = extract_frames_from_video(video_file, video_state, progress)
    
    # Set up SAM controller if frames were extracted successfully
    if frames_extracted and video_state.get("origin_images"):
        model.samcontroler.sam_controler.reset_image() 
        model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    
    return video_state

def update_ui_after_video_upload(video_state):
    """
    Update UI components after video processing is complete
    """
    # Check if video processing was successful
    if not video_state.get("origin_images"):
        return [
            gr.update(visible=True),  # video_upload
            gr.update(visible=False),  # unified_image
            "**📊 Video Info:** Upload failed - try another video",
            gr.update(interactive=False),  # remove_video_btn
            gr.update(interactive=False),  # image_selection_slider
            gr.update(interactive=False),  # point_prompt
            gr.update(interactive=False),  # clear_button_click
            gr.update(interactive=False),  # add_mask_button
            gr.update(interactive=False),  # remove_mask_button
            gr.update(interactive=False),  # mask_dropdown
            gr.update(interactive=False),  # matting_button
        ]
    
    # Switch to interactive mode
    first_frame = video_state["origin_images"][0]
    video_info_text = f"**📊 Video Info:** {len(video_state['origin_images'])} frames | {video_state['fps']:.1f} fps | {video_state['video_name']}"
    
    return [
        gr.update(visible=False),  # video_upload - hide it
        gr.update(value=first_frame, visible=True, interactive=True),  # unified_image - show it
        video_info_text,  # video_info
        gr.update(interactive=True),  # remove_video_btn
        gr.update(interactive=True, maximum=len(video_state['origin_images']), value=1),  # image_selection_slider
        gr.update(interactive=True),  # point_prompt
        gr.update(interactive=True),  # clear_button_click
        gr.update(interactive=True),  # add_mask_button
        gr.update(interactive=True),  # remove_mask_button
        gr.update(interactive=True),  # mask_dropdown
        gr.update(interactive=False),  # matting_button (disabled until mask created)
    ]

def process_video_upload_focused(video_file, video_state, progress=gr.Progress()):
    """
    Process video upload with standard progress bars
    """
    # Extract frames (this is the heavy operation that needs progress)
    video_state, frames_extracted = process_video_frames_only(video_file, video_state, progress)
    
    if not frames_extracted:
        return [
            video_state,
            gr.update(visible=True),  # video_upload
            gr.update(visible=False),  # unified_image
            "**📊 Video Info:** Upload failed - try another video",
            gr.update(interactive=False),  # remove_video_btn
            gr.update(interactive=False),  # image_selection_slider
            gr.update(interactive=False),  # point_prompt
            gr.update(interactive=False),  # clear_button_click
            gr.update(interactive=False),  # add_mask_button
            gr.update(interactive=False),  # remove_mask_button
            gr.update(interactive=False),  # mask_dropdown
            gr.update(interactive=False),  # matting_button
        ]
    
    # Switch to interactive mode
    first_frame = video_state["origin_images"][0]
    video_info_text = f"**📊 Video Info:** {len(video_state['origin_images'])} frames | {video_state['fps']:.1f} fps | {video_state['video_name']}"
    
    return [
        video_state,
        gr.update(visible=False),  # video_upload - hide it
        gr.update(value=first_frame, visible=True, interactive=True),  # unified_image - show it
        video_info_text,  # video_info
        gr.update(interactive=True),  # remove_video_btn
        gr.update(interactive=True, maximum=len(video_state['origin_images']), value=1),  # image_selection_slider
        gr.update(interactive=True),  # point_prompt
        gr.update(interactive=True),  # clear_button_click
        gr.update(interactive=True),  # add_mask_button
        gr.update(interactive=True),  # remove_mask_button
        gr.update(interactive=True),  # mask_dropdown
        gr.update(interactive=False),  # matting_button (disabled until mask created)
    ]

def extract_frames_from_video(video_input, video_state, progress=gr.Progress()):
    """
    Extract frames from video and return updated video_state and success flag
    """
    if not video_input:
        gr.Warning("Please upload a video first")
        return video_state, False
    
    video_path = video_input.name if hasattr(video_input, 'name') else video_input
    frames = []
    user_name = time.time()

    progress(0.1, desc="Analyzing video metadata")
    
    # extract Audio
    try:
        audio_path = video_path.replace(".mp4", "_audio.wav")
        probe = ffmpeg.probe(video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        progress(0.2, desc="Extracting audio track")
        if audio_streams:  # Only try to extract if video has audio
            try:
                ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100').run(overwrite_output=True, quiet=True)
            except ffmpeg.Error as e:
                print(f"Audio extraction error: {str(e)}")
                audio_path = ""
        else:
            audio_path = ""
            print("Note: Input video has no audio track")
    except Exception as e:
        print(f"Error checking audio stream: {str(e)}")
        audio_path = ""
    
    # extract frames
    try:
        progress(0.3, desc="Opening video file")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress(0.4, desc="Starting frame extraction")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
                
                # Update progress for each frame
                progress_val = 0.4 + (frame_count / total_frames) * 0.5  # 0.4 to 0.9
                progress(progress_val, desc=f"Extracting frame {frame_count}/{total_frames} (Memory: {current_memory_usage:.0f}%)")
                
                if current_memory_usage > 90:
                    print(f"Memory usage high ({current_memory_usage}%), stopping frame extraction")
                    break
            else:
                break
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    # Use native resolution - no resizing for maximum quality
    progress(0.9, desc="Initializing processing pipeline")
    image_size = (frames[0].shape[0], frames[0].shape[1])

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "audio": audio_path,
        "original_video_path": video_path
    }
    
    
    progress(0.95, desc="Setting up SAM controller")
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    
    progress(1.0, desc="Video processing ready")
    return video_state, True

def reset_to_upload_mode():
    """
    Reset the unified component back to upload mode
    """
    # Reset all states
    video_state, interactive_state, click_state = restart()[:3]
    
    return [
        video_state, interactive_state, click_state,
        gr.update(visible=True, value=None),  # video_upload - show it
        gr.update(visible=False, value=None),  # unified_image - hide it
        "**📊 Video Info:** Upload a video to begin",  # video_info
        gr.update(interactive=False),  # remove_video_btn
        gr.update(interactive=False, value=1),  # image_selection_slider
        gr.update(interactive=False),  # point_prompt
        gr.update(interactive=False),  # clear_button_click
        gr.update(interactive=False),  # add_mask_button
        gr.update(interactive=False),  # remove_mask_button
        gr.update(interactive=False, value=[]),  # mask_dropdown
        gr.update(interactive=False),  # matting_button
        gr.update(value=None),  # foreground_video_output
        gr.update(value=None),  # alpha_video_output
        gr.update(value=None),  # background_output
    ]

# get the select frame from gradio slider
def select_video_template_unified(image_selection_slider, video_state, interactive_state):
    """
    Update the unified image component when frame selection changes
    """
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # Update unified image with selected frame
    selected_frame = video_state["painted_images"][image_selection_slider]
    return gr.update(value=selected_frame), video_state, interactive_state


# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData, progress=gr.Progress()):
    """
    Generate mask with SAM - focused function that returns only the image
    """
    progress(0.1, desc="Processing click")
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    progress(0.3, desc="Preparing SAM")
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    progress(0.7, desc="Generating mask")
    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    progress(0.9, desc="Updating mask")
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    progress(1.0, desc="Mask complete")
    return painted_image

def update_states_after_sam(video_state, interactive_state):
    """
    Update states and enable matting button after SAM processing
    """
    return video_state, interactive_state, gr.update(interactive=True)  # Enable matting button

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    mask = video_state["masks"][video_state["select_frame_number"]]
    
    # Check if mask is empty or covers the entire frame (all zeros or all ones)
    if mask.size == 0 or np.all(mask == 0) or np.all(mask == 1):
        gr.Warning("Please create a mask by clicking on the image before saving")
        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), gr.update(value=video_state["painted_images"][video_state["select_frame_number"]]), [[],[]]
    
    interactive_state["multi_mask"]["masks"].append(mask)
    interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    select_frame = show_mask(video_state, interactive_state, mask_dropdown)

    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), gr.update(value=select_frame), [[],[]]

def clear_click_unified(video_state, click_state):
    """
    Clear click state and reset unified image to original frame
    """
    click_state = [[],[]]
    original_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return gr.update(value=original_frame), click_state

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    return interactive_state, gr.update(choices=[],value=[])

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if video_state["origin_images"]:
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["multi_mask"]["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        
        return select_frame
    return None

def show_mask_for_dropdown(video_state, interactive_state, mask_dropdown):
    """Wrapper for mask dropdown change event that returns gr.update"""
    result = show_mask(video_state, interactive_state, mask_dropdown)
    if result is not None:
        return gr.update(value=result)
    return gr.update()

def generate_output_filename(video_name, output_type="fg"):
    # Remove extension from video name and limit length to 32 chars
    base_name = os.path.splitext(video_name)[0][:32]
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Add output type (fg for foreground or alpha for alpha mask)
    suffix = "_fg" if output_type == "fg" else "_alpha"
    
    return f"{base_name}_{timestamp}{suffix}.mp4"

    
def process_matting_sequence(processor, frames, template_mask, mask_adjustment, device, progress_callback=None):
    """
    Helper function to process a sequence of frames with MatAnyone
    
    Args:
        processor: InferenceCore instance
        frames: List of frames to process
        template_mask: Mask for the template frame
        mask_adjustment: Mask edge adjustment (negative=shrink, positive=expand)
        device: Processing device
        progress_callback: Optional callback for progress updates
    
    Returns:
        foreground: List of foreground frames
        alpha: List of alpha frames
    """
    if len(frames) == 0:
        return [], []
    
    foreground, alpha = matanyone(processor, frames, template_mask*255, 
                                  mask_adjustment=mask_adjustment, device=device,
                                  progress_callback=progress_callback)
    return foreground, alpha

# Video Matting with Bidirectional Support
def video_matting(video_state, interactive_state, mask_dropdown, mask_adjustment, progress=gr.Progress()):
    progress(0.05, desc="Processing mask selection")
    
    # Process mask selection (same as before)
    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    
    fps = video_state["fps"]
    audio_path = video_state["audio"]
    
    progress(0.1, desc="Validating mask data")
    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
    
    # Get frame indices
    select_frame_num = video_state["select_frame_number"]
    all_frames = video_state["origin_images"]
    total_frames = len(all_frames)
    
    # Process all frames to the end (removed end frame selection)
    end_frame_num = total_frames
    
    progress(0.15, desc="Analyzing processing strategy")
    # Check if bidirectional processing is needed
    if select_frame_num > 0:
        print(f"🔄 BIDIRECTIONAL PROCESSING ACTIVATED")
        print(f"📊 Total frames: {total_frames}, Selected frame: {select_frame_num}, End frame: {end_frame_num}")
        
        # Process backward sequence (frames 0 to select_frame_num-1, reversed)
        backward_frames = all_frames[:select_frame_num]
        if len(backward_frames) > 0:
            progress(0.2, desc=f"Processing backward: 0/{len(backward_frames)} frames")
            print(f"⬅️  Processing BACKWARD sequence: frames 0 to {select_frame_num-1} ({len(backward_frames)} frames)")
            backward_frames_reversed = list(reversed(backward_frames))
            matanyone_processor_backward = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
            
            # Create progress callback for backward processing
            def backward_progress(current, total):
                progress_val = 0.2 + (current / total) * 0.3  # 0.2 to 0.5
                progress(progress_val, desc=f"Processing backward: {current}/{total} frames")
            
            backward_fg, backward_alpha = process_matting_sequence(
                matanyone_processor_backward, backward_frames_reversed, template_mask,
                mask_adjustment, args.device, progress_callback=backward_progress)
            
            # Reverse results back to correct temporal order
            backward_fg = list(reversed(backward_fg))
            backward_alpha = list(reversed(backward_alpha))
            print(f"✅ Backward processing complete: {len(backward_fg)} frames generated")
        else:
            backward_fg, backward_alpha = [], []
            progress(0.3, desc="No backward frames to process")
            print(f"⚠️  No backward frames to process")
        
        # Process forward sequence (select_frame_num to end_frame_num)
        forward_frames = all_frames[select_frame_num:end_frame_num]
        if len(forward_frames) > 0:
            progress(0.55, desc=f"Processing forward: 0/{len(forward_frames)} frames")
            print(f"➡️  Processing FORWARD sequence: frames {select_frame_num} to {end_frame_num-1} ({len(forward_frames)} frames)")
            matanyone_processor_forward = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
            
            # Create progress callback for forward processing
            def forward_progress(current, total):
                progress_val = 0.55 + (current / total) * 0.25  # 0.55 to 0.8
                progress(progress_val, desc=f"Processing forward: {current}/{total} frames")
            
            forward_fg, forward_alpha = process_matting_sequence(
                matanyone_processor_forward, forward_frames, template_mask,
                mask_adjustment, args.device, progress_callback=forward_progress)
            print(f"✅ Forward processing complete: {len(forward_fg)} frames generated")
        else:
            forward_fg, forward_alpha = [], []
            progress(0.6, desc="No forward frames to process")
            print(f"⚠️  No forward frames to process")
        
        # Combine results
        foreground = backward_fg + forward_fg
        alpha = backward_alpha + forward_alpha
        print(f"🔗 COMBINED RESULTS: {len(backward_fg)} backward + {len(forward_fg)} forward = {len(foreground)} total frames")
        
    else:
        print(f"➡️  UNIDIRECTIONAL PROCESSING (original behavior)")
        print(f"📊 Processing frames {select_frame_num} to {end_frame_num-1} ({end_frame_num - select_frame_num} frames)")
        
        # Unidirectional processing (original behavior for select_frame_num = 0)
        following_frames = all_frames[select_frame_num:end_frame_num]
        total_frames = len(following_frames)
        progress(0.2, desc=f"Processing: 0/{total_frames} frames")
        matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
        
        # Create progress callback for unidirectional processing
        def uni_progress(current, total):
            progress_val = 0.2 + (current / total) * 0.6  # 0.2 to 0.8
            progress(progress_val, desc=f"Processing: {current}/{total} frames")
        
        foreground, alpha = process_matting_sequence(
            matanyone_processor, following_frames, template_mask,
            mask_adjustment, args.device, progress_callback=uni_progress)
        print(f"✅ Unidirectional processing complete: {len(foreground)} frames generated")

    # Generate output filenames with timestamps
    progress(0.85, desc="Preparing output files")
    fg_output_name = generate_output_filename(video_state["video_name"], output_type="fg")
    alpha_output_name = generate_output_filename(video_state["video_name"], output_type="alpha")

    # Create outputs directory
    output_dir = os.path.join(".", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate videos with consistent naming and high quality
    progress(0.9, desc="Generating foreground video")
    foreground_output = generate_video_from_frames(foreground, 
        output_path=os.path.join(output_dir, fg_output_name),
        fps=fps, audio_path=audio_path, high_quality=True)
    
    progress(0.95, desc="Generating alpha video")
    alpha_output = generate_video_from_frames(alpha, 
        output_path=os.path.join(output_dir, alpha_output_name),
        fps=fps, gray2rgb=True, audio_path=audio_path, high_quality=True)
    
    # Store paths in video_state for background reconstruction
    video_state["foreground_video_path"] = foreground_output
    video_state["alpha_video_path"] = alpha_output
    
    progress(1.0, desc="Video matting complete")
    return foreground_output, alpha_output


def reconstruct_background_from_videos(input_video_path, alpha_video_path, output_dir, dilate_pixels=2, max_samples=200, progress=gr.Progress()):
    """
    Reconstruct background from original video and alpha mask video.
    
    Args:
        input_video_path: Path to original input video
        alpha_video_path: Path to alpha mask video from matting
        output_dir: Directory to save the output
        dilate_pixels: Number of pixels to dilate foreground areas
        max_samples: Maximum number of samples per pixel
    
    Returns:
        Path to the saved background image
    """
    progress(0.05, desc="Opening video files")
    # Open video captures
    cap_input = cv2.VideoCapture(input_video_path)
    cap_alpha = cv2.VideoCapture(alpha_video_path)
    
    if not cap_input.isOpened() or not cap_alpha.isOpened():
        raise ValueError("Cannot open video files")
    
    progress(0.1, desc="Analyzing video properties")
    # Get video properties
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(
        int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_alpha.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    
    print(f"\n📊 Processing {total_frames} frames at {width}x{height} resolution")
    print(f"🎯 Background detection: 100% black pixels only")
    if dilate_pixels > 0:
        print(f"🛡️ Safety: Foreground areas dilated by {dilate_pixels}px to avoid edge artifacts")
    else:
        print(f"⚠️ No dilation applied - edge artifacts may occur")
    
    # Pre-allocate storage
    progress(0.15, desc="Initializing reconstruction arrays")
    pixel_data = np.zeros((height, width, max_samples, 3), dtype=np.uint8)
    pixel_counts = np.zeros((height, width), dtype=np.int32)
    pixels_initialized = np.zeros((height, width), dtype=bool)
    
    try:
        # Process frames with progress bar
        print("\nProcessing frames for background reconstruction...")
        pbar = tqdm(total=total_frames, desc="Processing")
        
        frame_idx = 0
        
        progress(0.2, desc="Starting frame processing")
        while frame_idx < total_frames:
            # Read frames
            ret_input, frame_input = cap_input.read()
            ret_alpha, frame_alpha = cap_alpha.read()
            
            if not ret_input or not ret_alpha:
                break
            
            # Convert input frame to RGB
            frame_input_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            
            # Process alpha mask
            if len(frame_alpha.shape) == 3:
                alpha_mask = cv2.cvtColor(frame_alpha, cv2.COLOR_BGR2GRAY)
            else:
                alpha_mask = frame_alpha
            
            # Resize alpha mask if needed
            if alpha_mask.shape != (height, width):
                alpha_mask = cv2.resize(alpha_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Create background mask (pixels where alpha is exactly black)
            background_mask = alpha_mask == 0
            
            # Apply dilation to foreground areas if specified
            if dilate_pixels > 0:
                kernel_size = dilate_pixels * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                background_mask = cv2.erode(background_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            # Process background pixels
            bg_positions = np.where(background_mask)
            
            if len(bg_positions[0]) > 0:
                bg_y, bg_x = bg_positions
                bg_rgb_values = frame_input_rgb[bg_positions]
                
                # Get current counts
                current_counts = pixel_counts[bg_y, bg_x]
                
                # Only process pixels that haven't reached max samples
                under_max_mask = current_counts < max_samples
                
                if np.any(under_max_mask):
                    under_y = bg_y[under_max_mask]
                    under_x = bg_x[under_max_mask]
                    under_rgb = bg_rgb_values[under_max_mask]
                    under_counts = current_counts[under_max_mask]
                    
                    pixel_data[under_y, under_x, under_counts] = under_rgb
                    pixel_counts[under_y, under_x] += 1
                    pixels_initialized[under_y, under_x] = True
            
            frame_idx += 1
            pbar.update(1)
            
            # Update progress info for each frame
            unique_pixels = np.sum(pixel_counts > 0)
            total_pixels = height * width
            coverage = (unique_pixels / total_pixels) * 100
            pbar.set_postfix_str(f"Coverage: {coverage:.1f}%")
            
            # Update Gradio progress (20% to 80% of total progress)
            progress_val = 0.2 + (frame_idx / total_frames) * 0.6
            progress(progress_val, desc=f"Frame {frame_idx}/{total_frames} analyzed (Coverage: {coverage:.1f}%)")
        
        pbar.close()
    
    finally:
        cap_input.release()
        cap_alpha.release()
    
    # Calculate final background using median
    progress(0.85, desc="Computing final background")
    print("Computing final background using median...")
    
    background_result = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA format
    
    # Find all pixels that have data
    valid_positions = np.where(pixels_initialized)
    total_valid = len(valid_positions[0])
    
    if total_valid > 0:
        valid_y, valid_x = valid_positions
        valid_counts = pixel_counts[valid_y, valid_x]
        
        # Process all pixels
        for y, x, count in zip(valid_y, valid_x, valid_counts):
            if count > 0:
                # Get RGB values and compute median
                pixel_rgb_values = pixel_data[y, x, :count]
                median_rgb = np.median(pixel_rgb_values, axis=0).astype(np.uint8)
                
                # Set RGBA values
                background_result[y, x, :3] = median_rgb
                background_result[y, x, 3] = 255  # Opaque
    
    # Calculate statistics
    unique_pixels = np.sum(pixel_counts > 0)
    total_pixels = height * width
    coverage = (unique_pixels / total_pixels) * 100
    
    print(f"Background reconstruction complete!")
    print(f"Coverage: {coverage:.1f}% ({unique_pixels}/{total_pixels} pixels)")
    
    # Generate output filename
    progress(0.95, desc="Saving background image")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"reconstructed_background_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save result with alpha channel
    background_pil = Image.fromarray(background_result, mode='RGBA')
    background_pil.save(output_path)
    
    progress(1.0, desc="Background reconstruction complete")
    print(f"Background saved to: {output_path}")
    
    return output_path


def create_progress_html(value, desc):
    """Create custom progress HTML"""
    return f'''
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 16px; margin: 8px 0; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-weight: 600; color: white; font-size: 14px;">{desc}</span>
            <span style="font-size: 13px; color: rgba(255,255,255,0.9); background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 6px;">{int(value*100)}%</span>
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 6px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); height: 100%; width: {value*100}%; transition: width 0.3s ease; border-radius: 6px;"></div>
        </div>
    </div>
    '''

def video_matting_focused(video_state, interactive_state, mask_dropdown, mask_adjustment, progress=gr.Progress()):
    """
    Video matting function that returns only the primary foreground video
    """
    # Call the original video_matting function
    foreground_output, alpha_output = video_matting(
        video_state, interactive_state, mask_dropdown, 
        mask_adjustment, progress
    )
    
    # Store alpha output in video_state for later use
    video_state["alpha_video_path"] = alpha_output
    
    # Return only the primary foreground video
    return foreground_output

def update_ui_after_matting(video_state):
    """
    Update UI components after video matting is complete
    """
    alpha_output = video_state.get("alpha_video_path")
    return (
        gr.update(value=alpha_output),       # alpha_video_output
        gr.update(),  # bg_settings_card (no update needed)
        gr.update(interactive=True)  # Enable background reconstruction button
    )

def restart_workflow():
    """
    Restart the entire workflow with new dual-column UI structure.
    """
    # Reset all states
    video_state, interactive_state, click_state = restart()[:3]
    
    return [
        video_state, interactive_state, click_state,
        gr.update(interactive=False, value=None), # template_frame
        gr.update(value="### 👆 Upload a video to begin"),  # placeholder_msg
        gr.update(interactive=False, value=1),  # image_selection_slider
        "**📊 Video Info:** Upload a video to see details here",  # video_info
        gr.update(),  # masking_controls (no update needed)
        gr.update(interactive=False, value=[]),  # mask_dropdown
        gr.update(interactive=False), # matting_button
        gr.update(),  # bg_settings_card (no update needed)
        gr.update(value=None), # foreground_video_output
        gr.update(value=None), # alpha_video_output
        gr.update(value=None), # background_output
    ]


def process_background_reconstruction(video_state, dilate_pixels, max_samples, progress=gr.Progress()):
    """
    Wrapper function for Gradio interface to reconstruct background.
    
    Args:
        video_state: State containing video paths
        dilate_pixels: Dilation parameter
        max_samples: Maximum samples per pixel
        progress: Gradio progress tracker
    
    Returns:
        Path to the reconstructed background image
    """
    if "video_name" not in video_state or not video_state["video_name"]:
        gr.Warning("Please process a video first")
        return None
    
    if "alpha_video_path" not in video_state or not video_state["alpha_video_path"]:
        gr.Warning("Please complete video matting first")
        return None
    
    # Get the original video path
    input_video_path = video_state.get("original_video_path")
    if not input_video_path:
        gr.Warning("Original video path not found")
        return None
    
    alpha_video_path = video_state["alpha_video_path"]
    
    # Determine output directory
    output_dir = os.path.join(".", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        background_path = reconstruct_background_from_videos(
            input_video_path,
            alpha_video_path,
            output_dir,
            dilate_pixels,
            max_samples,
            progress
        )
        return background_path
    except Exception as e:
        gr.Warning(f"Background reconstruction failed: {str(e)}")
        return None


def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)

        _ = (
            ffmpeg
            .output(video_input, audio_input, output_path, 
                   vcodec="copy", acodec="aac", audio_bitrate="320k")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error:\n{e.stderr.decode()}")
        return None


def generate_video_from_frames(frames, output_path, fps=30, gray2rgb=False, audio_path="", high_quality=True):
    """
    Generates a video from a list of frames with optimized quality settings.
    """
    frames = torch.from_numpy(np.asarray(frames))
    _, h, w, _ = frames.shape
    if gray2rgb:
        frames = np.repeat(frames, 3, axis=3)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    # Write to a temporary file first
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    
    if high_quality:
        # High quality settings: lossless/near-lossless encoding
        ffmpeg_params = [
            "-crf", "10",  # Very high quality (0-51, lower = better)
            "-preset", "slow",  # Better compression efficiency
            "-pix_fmt", "yuv420p",  # Better compatibility
            "-movflags", "+faststart",  # Web optimization
            "-vf", f"scale={w}:{h}"
        ]
        imageio.mimwrite(temp_path, frames, fps=fps, codec='libx264', ffmpeg_params=ffmpeg_params)
    else:
        # Fallback to original settings
        imageio.mimwrite(temp_path, frames, fps=fps, quality=7, 
                         codec='libx264', ffmpeg_params=["-vf", f"scale={w}:{h}"])
    
    # Add audio if it exists, otherwise just rename temp to final
    if audio_path and os.path.exists(audio_path):
        output_path = add_audio_to_video(temp_path, audio_path, output_path)    
        os.remove(temp_path)
        return output_path
    else:
        os.rename(temp_path, output_path)
        return output_path

# reset all states for a new input
# UI Helper Functions for simplified interface


def restart():
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30,
            "audio": ""
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            }, [[],[]]

# args, defined in track_anything.py
args = parse_augment()
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_folder = os.path.join('..', 'pretrained_models')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
# initialize sams
model = MaskGenerator(sam_checkpoint, args)

# initialize matanyone
pretrain_model_url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
ckpt_path = load_file_from_url(os.path.join(pretrain_model_url), checkpoint_folder)
matanyone_model = get_matanyone_model(ckpt_path, args.device)
matanyone_model = matanyone_model.to(args.device).eval()
# matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)

# download test samples
media_url = "https://github.com/pq-yang/MatAnyone/releases/download/media/"
test_sample_path = os.path.join('.', "test_sample/")
load_file_from_url(os.path.join(media_url, 'test-sample0-720p.mp4'), test_sample_path)
load_file_from_url(os.path.join(media_url, 'test-sample1-720p.mp4'), test_sample_path)
load_file_from_url(os.path.join(media_url, 'test-sample2-720p.mp4'), test_sample_path)
load_file_from_url(os.path.join(media_url, 'test-sample3-720p.mp4'), test_sample_path)
load_file_from_url(os.path.join(media_url, 'test-sample0.jpg'), test_sample_path)
load_file_from_url(os.path.join(media_url, 'test-sample1.jpg'), test_sample_path)

# download assets
assets_path = os.path.join('.', "assets/")
load_file_from_url(os.path.join(media_url, 'tutorial_single_target.mp4'), assets_path)
load_file_from_url(os.path.join(media_url, 'tutorial_multi_targets.mp4'), assets_path)



with gr.Blocks(title="MatAnyone - Professional Video Matting", fill_height=True) as demo:
    # State management
    click_state = gr.State([[],[]])
    
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30,
        "audio": "",
        }
    )
    
    # Header
    gr.Markdown("# 🎬 MatAnyone - Professional Video Matting")
    gr.Markdown("Extract objects from videos with AI-powered matting technology")
    
    # Optimized dual-column layout with better proportions
    with gr.Row(equal_height=True):

        # LEFT COLUMN: Complete Workflow (Upload → Mask → Process) 
        with gr.Column(scale=3, min_width=450):
            # Video Control Bar
            with gr.Group():
                with gr.Row():
                    video_info = gr.Markdown("**📊 Video Info:** Upload a video to begin")
                    remove_video_btn = gr.Button("🗑️ Remove Video", size="sm", variant="secondary", interactive=False)
            
            # Unified Upload/Interactive Area
            with gr.Group():
                
                # Video upload area (visible when no video loaded)
                video_upload = gr.File(
                    label="Drag & drop a video file here to begin",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    type="filepath",
                    visible=True,
                    elem_classes="unified-area"
                )
                
                # Custom progress indicator for video upload
                upload_progress = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes="custom-progress"
                )
                
                # Interactive masking area (visible when video loaded)
                unified_image = gr.Image(
                    label="Click on the target object to create masks", 
                    sources=[],
                    interactive=False,
                    container=True,
                    show_download_button=False,
                    scale=1,
                    visible=False,
                    elem_classes="unified-area"
                )
                
                # Frame selection directly under masking widget
                image_selection_slider = gr.Slider(
                    minimum=1, maximum=100, step=1, value=1, 
                    label="Select Start Frame", 
                    info="Choose where to begin processing",
                    interactive=False
                )
            
            # Masking controls
            with gr.Group() as masking_controls:
                gr.Markdown("**🎯 Click Mode** - Select how your clicks affect the mask")
                point_prompt = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="",
                    info="✅ Positive: Add to mask | ❌ Negative: Remove from mask",
                    interactive=False,
                    elem_classes="click-mode-radio"
                )
                
                with gr.Row(equal_height=True):
                    clear_button_click = gr.Button("🗑️ Clear", size="sm", scale=1, variant="secondary", interactive=False)
                    add_mask_button = gr.Button("💾 Save Mask", size="sm", scale=2, variant="primary", interactive=False)
                
                gr.Markdown("**🎭 Mask Management** - Manage your saved masks")
                mask_dropdown = gr.Dropdown(
                    multiselect=True, value=[], 
                    label="Saved Masks", 
                    info="💡 Select which saved masks to use for processing",
                    interactive=False
                )
                remove_mask_button = gr.Button("❌ Remove All Masks", size="sm", variant="secondary", interactive=False)
            
            # Processing Settings
            with gr.Group():
                gr.Markdown("### ⚙️ Processing Settings")
                mask_adjustment = gr.Slider(
                    label='Mask Edge Adjustment',
                    minimum=-30,
                    maximum=30,
                    step=1,
                    value=0,
                    info="Negative shrinks, positive expands mask edges"
                )
                
                # Start processing button at bottom
                matting_button = gr.Button(
                    "🎬 Start Video Matting", 
                    variant="primary",
                    size="lg",
                    interactive=False,
                    scale=1,
                    elem_classes="main-process-button"
                )
        
        # RIGHT COLUMN: Results & Outputs
        with gr.Column(scale=4, min_width=500):
            # Results Section
            with gr.Group():
                gr.Markdown("### 📊 Results")
                
                # Custom progress indicator for video matting
                matting_progress = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes="matting-progress"
                )
                
                foreground_video_output = gr.Video(
                    label="🎬 Foreground Video",
                    autoplay=True,
                    show_download_button=True,
                    container=True,
                    elem_classes="result-video"
                )
                alpha_video_output = gr.Video(
                    label="🎭 Alpha Mask Video",
                    autoplay=True,
                    show_download_button=True,
                    container=True,
                    elem_classes="result-video"
                )
            
            # Background Reconstruction
            bg_settings_card = gr.Group()
            with bg_settings_card:
                gr.Markdown("### 🖼️ Background Reconstruction")
                
                with gr.Row(equal_height=True):
                    bg_dilate_pixels = gr.Slider(
                        label="Edge Protection",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=2,
                        info="Avoid edge artifacts",
                        scale=1
                    )
                    
                    bg_max_samples = gr.Slider(
                        label="Sample Density",
                        minimum=50,
                        maximum=500,
                        step=50,
                        value=200,
                        info="Quality vs speed",
                        scale=1
                    )
                
                bg_recon_button = gr.Button("🖼️ Extract Background", variant="secondary", interactive=False, elem_classes="bg-process-button")
                
                background_output = gr.Image(
                    label="Reconstructed Background", 
                    type="filepath",
                    sources=[],
                    interactive=False,
                    show_download_button=True,
                    container=True
                )
            
                
        

    # Handle video upload to file component  
    video_upload.upload(
        fn=process_video_frames_only,
        inputs=[video_upload, video_state],
        outputs=[video_state],
        show_progress="full"
    ).then(
        fn=update_ui_after_video_upload,
        inputs=[video_state],
        outputs=[
            video_upload,
            unified_image,
            video_info,
            remove_video_btn,
            image_selection_slider,
            point_prompt,
            clear_button_click,
            add_mask_button,
            remove_mask_button,
            mask_dropdown,
            matting_button,
        ],
        show_progress=False
    )
    
    # Handle remove video button
    remove_video_btn.click(
        fn=reset_to_upload_mode,
        inputs=[],
        outputs=[
            video_state, interactive_state, click_state,
            video_upload,
            unified_image,
            video_info,
            remove_video_btn,
            image_selection_slider,
            point_prompt,
            clear_button_click,
            add_mask_button,
            remove_mask_button,
            mask_dropdown,
            matting_button,
            foreground_video_output,
            alpha_video_output,
            background_output,
        ]
    )   


    # second step: select images from slider
    image_selection_slider.release(
        fn=select_video_template_unified, 
        inputs=[image_selection_slider, video_state, interactive_state], 
        outputs=[unified_image, video_state, interactive_state], 
        api_name="select_image"
    )
    
    # click select image to get mask using sam
    unified_image.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[unified_image],
        show_progress="full"
    ).then(
        fn=update_states_after_sam,
        inputs=[video_state, interactive_state],
        outputs=[video_state, interactive_state, matting_button],
        show_progress=False
    )

    # add different mask
    add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, unified_image, click_state]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown]
    )

    # video matting
    matting_button.click(
        fn=video_matting_focused,
        inputs=[
            video_state, interactive_state, mask_dropdown, 
            mask_adjustment
        ],
        outputs=[foreground_video_output],
        show_progress="full"
    ).then(
        fn=update_ui_after_matting,
        inputs=[video_state],
        outputs=[
            alpha_video_output,
            bg_settings_card,
            bg_recon_button
        ],
        show_progress=False
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask_for_dropdown,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[unified_image]
    )
    
    # points clear
    clear_button_click.click(
        fn=clear_click_unified,
        inputs=[video_state, click_state],
        outputs=[unified_image, click_state]
    )


    
    # Background reconstruction with new UI
    bg_recon_button.click(
        fn=process_background_reconstruction,
        inputs=[video_state, bg_dilate_pixels, bg_max_samples],
        outputs=[background_output],
        show_progress="full"
    ).then(
        fn=lambda bg_path: gr.update(value=bg_path) if bg_path else gr.update(),
        inputs=[background_output],
        outputs=[background_output]
    )



demo.launch(debug=True)
