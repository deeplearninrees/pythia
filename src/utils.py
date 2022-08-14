import locale
import pygame
import cv2
from torchvision.transforms.functional import * #Includes PIL ops.
from PIL import ImageOps
locale.setlocale(locale.LC_ALL, '')

def get_nparams(model):
    nparams =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    pretty_nparams = "{:n}".format(nparams)
    return pretty_nparams

def space_is_pressed():
	events = pygame.event.get()
	return pygame.key.get_pressed()[pygame.K_SPACE]
	# print(events)
	# reward = 0
	# for i in events:
	# 	if i.type == pygame.KEYDOWN:
	# 		print(i.key)
	# 		if i.key == pygame.K_SPACE:
	# 			return True
	# return False
def get_shot(camera, size=(200,200)):
	#Grabs camera shot from cv2 and returns a tensor of size (1, 3, size[0], size[1]).
	_, frame = camera.read()
	frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pillowed_frame = Image.fromarray(frame_to_rgb)
	resized_frame = pillowed_frame.resize(size)
	tensored_frame = to_tensor(resized_frame)
	dimensional_frame = torch.unsqueeze(tensored_frame, dim=0)

	return dimensional_frame

def show_tensor(tensor, display, surface):
	#Displays a 2d image tensor with Pygame. (Tensor should have 4 dimensions, first two being 1 and 3 respectively.) Size doesn't matter as the window is adaptive.
	size = surface.get_size()
	dimensionless_tensor = torch.squeeze(tensor, dim=0)
	pillowed_tensor = to_pil_image(dimensionless_tensor)
	pygamed_tensor = pygame.image.fromstring(pillowed_tensor.tobytes(), pillowed_tensor.size, pillowed_tensor.mode)
	scaled_tensor = pygame.transform.scale(pygamed_tensor, size)
	display.blit(scaled_tensor, (0,0))
	pygame.display.update()
	pygame.event.get()

def init_camera_and_window():
	#Inits a camera and a window with cv2 and pygame.
	pygame.init()
	
	camera = cv2.VideoCapture(0)
	display = pygame.display.set_mode((0,0), pygame.RESIZABLE)
	surface = pygame.display.get_surface()
	return camera, display, surface