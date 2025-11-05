import cv2
import numpy as np

def feather_mask(mask, blur_amount=11, erode_amount=3):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if erode_amount > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_amount, erode_amount))
        mask = cv2.erode(mask, kernel, iterations=1)        
    if blur_amount > 0:
        blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
    return mask

def make_texture_seamless(texture):
    h, w = texture.shape[:2]
    texture = texture.astype(np.float32)
    blend_width = min(w // 8, 30)
    blend_height = min(h // 8, 30)

    left = texture[:, :blend_width]
    right = texture[:, -blend_width:]
    mix_lr = cv2.addWeighted(left, 0.5, cv2.flip(right, 1), 0.5, 0)
    texture[:, :blend_width] = mix_lr
    texture[:, -blend_width:] = cv2.flip(mix_lr, 1)

    top = texture[:blend_height, :]
    bottom = texture[-blend_height:, :]
    mix_tb = cv2.addWeighted(top, 0.5, cv2.flip(bottom, 0), 0.5, 0)
    texture[:blend_height, :] = mix_tb
    texture[-blend_height:, :] = cv2.flip(mix_tb, 0)
    return np.clip(texture, 0, 255).astype(np.uint8)

def tile_texture(texture, width, height, scale=1.0):
    texture = make_texture_seamless(texture.copy())    
    th, tw = texture.shape[:2]
    scale = max(scale, 0.1)
    th_scaled, tw_scaled = int(th / scale), int(tw / scale)
    if th_scaled == 0 or tw_scaled == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)         
    texture_scaled = cv2.resize(texture, (tw_scaled, th_scaled), interpolation=cv2.INTER_LANCZOS4)
    th, tw = texture_scaled.shape[:2]
    rep_x = int(np.ceil(width / tw))
    rep_y = int(np.ceil(height / th))
    tiled = np.tile(texture_scaled, (rep_y, rep_x, 1))
    return tiled[:height, :width]

def apply_live_adjustments(image, brightness, contrast, saturation):
    bright_val = (brightness - 50) * 2.55
    contrast_val = 1 + (contrast - 50) / 50.0
    image = cv2.addWeighted(image, contrast_val, image, 0, bright_val)
    image = np.clip(image, 0, 255)

    if saturation != 50:
        sat_val = saturation / 50.0        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = s.astype(np.float32) * sat_val
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image.astype(np.uint8)

def apply_design(room_image, texture_image, mask_image, 
                 scale, brightness, contrast, saturation):
    h, w = room_image.shape[:2]
    design_tiled = tile_texture(texture_image, w, h, scale)
    design_adjusted = apply_live_adjustments(design_tiled, brightness, contrast, saturation)
    room_lab = cv2.cvtColor(room_image, cv2.COLOR_BGR2LAB)
    tex_lab = cv2.cvtColor(design_adjusted, cv2.COLOR_BGR2LAB)
    room_L, _, _ = cv2.split(room_lab)
    _, tex_A, tex_B = cv2.split(tex_lab)    
    blended_lab = cv2.merge([room_L, tex_A, tex_B])
    lit_design = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    mask_f = feather_mask(mask_image, blur_amount=11, erode_amount=3)
    mask_f = mask_f.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_f, mask_f, mask_f])
    final_image = lit_design * mask_3ch + room_image * (1 - mask_3ch)    
    return final_image.astype(np.uint8)