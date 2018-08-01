import ast
import configparser
import numpy as np
import re, os
from scipy.misc import imsave, imread
from PIL import Image, ImageDraw, ImageFont

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()

def create_directory(filename):
    target_dir = os.path.dirname(filename)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

def load_file(filename='config', section='SYNTH_DATA'):
	# returns dictionary with all params

	# Import configuration
	config = configparser.ConfigParser()
	res = config.read(filename)
	if len(res) == 0:
		print("ERROR: couldn't load 'config' file. To fix, copy 'config.sample' to 'config' (and do not version this file)")
		exit(1)

	params = {}
	options = config.options(section)
	for option in options:
		try:
			params[option] = ast.literal_eval(config.get(section, option))
			if params[option] == -1:
				print("skip: %s" % option)
		except:
			print(" CONFIG PARSING EXCEPTION on %s" % option)
			params[option] = None
			raise

	return params

def image_write(filename, image):
    create_directory(filename)
    imsave(filename, image)

def pngdepth_read(filename):
    return imread(filename).astype(np.float32) / 1000.0

def pngdepth_write(filename, depth):
    # this is specifically for writing depth in 16-bit png
    from cv2 import imwrite as depth_write 
    create_directory(filename)
    # we need to save it as uint16
    depth_write(filename, (depth.copy() * 1e3).astype(np.uint16)) 

def readPFM(file):
    """ read the file in pfm format
    Original code from the Scene Flow Dataset (CVPR 2016), Freiburg
    """

    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    """ write the file in pfm format
    Original code from the Scene Flow Dataset (CVPR 2016), Freiburg
    """

    create_directory(file)

    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def flow_visualize(flow, max_range = 1e3):
    """ Original code from SINTEL toolbox, by Jonas Wulff.
    """
    import matplotlib.colors as colors
    du = flow[:, :, 0]
    dv = flow[:, :, 1]
    [h,w] = du.shape
    max_flow = min(max_range, np.max(np.sqrt(du * du + dv * dv)))
    img = np.ones((h, w, 3), dtype=np.float64)
    # angle layer
    img[:, :, 0] = (np.arctan2(dv, du) / (2 * np.pi) + 1) % 1.0
    # magnitude layer, normalized to 1
    img[:, :, 1] = np.sqrt(du * du + dv * dv) / (max_flow + 1e-8)
    # phase layer
    #img[:, :, 2] = valid
    # convert to rgb
    img = colors.hsv_to_rgb(img)
    # remove invalid point
    img[:, :, 0] = img[:, :, 0]
    img[:, :, 1] = img[:, :, 1]
    img[:, :, 2] = img[:, :, 2]
    return img

def flow_read_from_flo(filename):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    create_directory(filename)

    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()