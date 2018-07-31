
import numpy as np
from scipy import ndimage

def coarse_depth_change_map(depth0, depth1, opticalflow):
    u_f = opticalflow[0]
    v_f = opticalflow[1]
    H, W = np.shape(u_f)
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    # calculate the inverse flow map
    depth0_warped = ndimage.map_coordinates(depth1, [y+u_f, x+u_f])
    depth_change = depth0_warped - depth0

    return depth_change

win = [3, 5, 7]
thres = 0.2
def gen_occlusion_maps(flow, depth, frames):
    """
    :param optical flow
    :param depth 
    :parma the number of frames 
    """

    (h, w) = depth.shape

    # get occlusion from depth
    fwd_pixel = -1 * np.ones([h, w])
    fwd_warped = np.zeros([h, w])

    bwd_pixel = -1 * np.ones([h, w])
    bwd_warped = np.zeros([h, w])

    occlusion = 0.5 * np.ones([h, w])

    for x in range(0, w):
        for y in range(0, h):
            i = (x - 1) * h + (y - 1)
            xf = int(round(x + 0.5 * (frames - 1) * flow[y, x, 0]))
            yf = int(round(y + 0.5 * (frames - 1) * flow[y, x, 1]))

            if xf >= 0 and xf < w and yf >= 0 and yf < h:
                if fwd_pixel[yf, xf] < 0:
                    fwd_pixel[yf, xf] = i
                    fwd_warped[yf, xf] = depth[y, x]
                elif fwd_warped[yf, xf] - depth[y, x] > thres:
                    occ_x = int(math.floor(fwd_pixel[yf, xf] / h))
                    occ_y = int(fwd_pixel[yf, xf] % h)

                    occlusion[occ_y, occ_x] = 1

                    fwd_pixel[yf, xf] = i
                    fwd_warped[yf, xf] = depth[y, x]
                elif fwd_warped[yf, xf] - depth[y, x] < -thres:
                    occlusion[y, x] = 1
            else:
                occlusion[y, x] = 1

            xf = int(round(x - 0.5 * (frames - 1) * flow[y, x, 0]))
            yf = int(round(y - 0.5 * (frames - 1) * flow[y, x, 1]))

            if xf >= 0 and xf < w and yf >= 0 and yf < h:
                if bwd_pixel[yf, xf] < 0:
                    bwd_pixel[yf, xf] = i
                    bwd_warped[yf, xf] = depth[y, x]
                elif bwd_warped[yf, xf] - depth[y, x] > thres:
                    occ_x = int(math.floor(bwd_pixel[yf, xf] / h))
                    occ_y = int(bwd_pixel[yf, xf] % h)

                    occlusion[occ_y, occ_x] = 0

                    bwd_pixel[yf, xf] = i
                    bwd_warped[yf, xf] = depth[y, x]
                elif bwd_warped[yf, xf] - depth[y, x] < -thres:
                    occlusion[y, x] = 0
            else:
                occlusion[y, x] = 0

    # median filter only if 0 and 1 state
    tmp = occlusion
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            patch = tmp[y - 1:y + 2, x - 1:x + 2]
            med = np.median(patch)

            if med == 0 or med == 1:
                occlusion[y, x] = med

    return occlusion
