# load segment anything 
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


checkpoint='<path_to_segmentanything>/segment-anything/checkpoints/sam_vit_l_0b3195.pth'
model_type='vit_l'

print(f'==> use segment anything:{model_type}')
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)