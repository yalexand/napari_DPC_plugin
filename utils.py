import numpy as np
import cv2 as cv
from scipy import signal, ndimage

def norm_u8(img): 
    # img should be of type np.ndarray
    if type(img) != np.ndarray:
        print('Invalid input type: {}. Excpet type: {}'.format(type(img),np.ndarray))
        return None #null
    else:
        if img.dtype == np.uint8:
            img *= 255/img.max()
            img = np.round(img).astype(np.uint8)
        else:
            img = np.round(255*(img.astype('float32')-img.mean())/(img.max()-img.mean())).astype(np.uint8)
        return img

def get_roi(para): #para = [x,y,w,h]
    x,y,w,h = tuple(para)
    return [y,y+h,x,x+w]

def get_img_roi(img, roi): # roi = [y,y+h,x,x+w]
        return img[roi[0]:roi[1],roi[2]:roi[3]]


# class for each images involved in co-registration 
class ImgToReg:
    # generate sift features for all chosen rois in the image
    def update_sift_features(self):
        for roi_img in self.roi_imgs:
            sift = cv.SIFT_create()
            kp, des = sift.detectAndCompute(roi_img,None)
            self.sift_features.append((kp, des))

    # this is a utility function for gen_affineTransform 
    # given sift features from rois in two images and the start points of rois 
    # compute the best matched point sets in two images 
    def match_sift(self, kp1, des1, start_pts1, kp2, des2, start_pts2, num = 1):
        # BFMatcher with default psarams
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            good.append([m])
        good = sorted(good, key=lambda x: x[0].distance, reverse=False)
        
        # generate real kp pairs (without overlapping points) in original imgs
        # Initialize lists
        list_kp1, list_kp2, good1 = [], [], []
        # For each good match...
        count = 0 
        for mat in good:
            mat = mat[0]
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            if count != 0:
                check = (x1+start_pts1[0] in np.asarray(list_kp1)[:,0]) or  (y1+start_pts1[1] in np.asarray(list_kp1)[:,1]) \
                    or (x2+start_pts2[0] in np.asarray(list_kp2)[:,0]) or  (y2+start_pts2[1] in np.asarray(list_kp2)[:,1])
            else:
                check = False

            if not check:
                # Append to each list
                list_kp1.append([x1+start_pts1[0], y1+start_pts1[1]])
                list_kp2.append([x2+start_pts2[0], y2+start_pts2[1]])
                count += 1
                good1.append([mat])
            if count>=num:
                break
        return list_kp1, list_kp2, good1

    # generate affine transform from this image to ref image given chosen rois
    def gen_affineTransform(self, ref_cls, roi_num = 3, each_num = 1):
        roi_num_ = min([len(ref_cls.roi_imgs), len(self.roi_imgs), roi_num])
        kps = [[], [], []] # kp1, kp2, good
        for i in range(roi_num_):
            kp1, des1 = self.sift_features[i]
            start_pts1 = self.roi_sts[i]
            kp2, des2 = ref_cls.sift_features[i]
            start_pts2 = ref_cls.roi_sts[i]
            list_kp1, list_kp2, good1 = self.match_sift(kp1, des1, start_pts1, kp2, des2, start_pts2, each_num)
            kps[0]+=list_kp1
            kps[1]+=list_kp2
            kps[2]+=good1
        # generate affine transform 
        srcTri = np.asarray(kps[0][:3]).astype('float32')
        dstTri = np.asarray(kps[1][:3]).astype('float32')
        affine1 = cv.getAffineTransform(srcTri, dstTri)

        shape = ref_cls.imgs['original'].shape[::-1]
        return affine1, shape
    
    # apply affine transform on original image
    def regTransform_img(self, affine, shape, use_img_flag = ''):
        if self.imgs.get(use_img_flag, -1) == -1:
            use_img_flag = 'original'

        img = self.imgs[use_img_flag].astype('float32')
        warp_dst =cv.warpAffine(img, affine, shape)
        return warp_dst

    # given start points and width&height of chosen rois
    # crop the actual rois from pre-processed original image 
    # and save them in the class attributes for registration 
    def update_rois(self, sts, roi_szs, use_img_flag = ''):
        if self.imgs.get(use_img_flag, -1) == -1:
            use_img_flag = 'pre_processed'
        self.roi_sts += sts
        self.roi_szs += roi_szs
        for st, roi_sz in zip(sts, roi_szs):
            roi = get_roi(st+roi_sz)
            roi_img = get_img_roi(self.imgs[use_img_flag], roi)
            self.rois.append(roi)
            self.roi_imgs.append(roi_img)

    # preprocess image for better feature extraction and matching 
    # median filter -> background subtraction -> norm to uint8 0-255
    def pre_process(self, img):
        temp = cv.medianBlur(img.astype('float32'), 3)
        temp -= temp.mean()
        temp = temp * (temp>0)
        temp = np.round(255*temp/temp.max()).astype('uint8')
        return temp

    # store original image and preprocessed original image in class attributes
    def update_imgs(self, img):
        self.imgs['original'] = img
        self.imgs['pre_processed'] = self.pre_process(img)

    def __init__(self, img, camera, sts, roi_szs): # sts is list of [x,y]; roi_szs is list of [w,h]
        self.imgs = dict()
        self.camera = ''
        self.roi_sts, self.roi_szs, self.rois, self.roi_imgs, self.sift_features = [], [], [], [], []
        
        self.camera = camera # name of camera which takes this image
        self.update_imgs(img)
        self.update_rois(sts, roi_szs)
        self.update_sift_features()

F     = lambda x: np.fft.fft2(x)
IF    = lambda x: np.fft.ifft2(x)

def binning_img(img, binning):
    # crop image if binning number not a factor of img shape
    img = img[:binning*(img.shape[0]//binning),:binning*(img.shape[1]//binning)]

    temp1 = []
    for j in list(range(binning)):
        for k in list(range(binning)):
            temp1.append(img[j::binning,k::binning])
    temp2 = np.asarray(temp1).mean(axis = 0)
    return temp2

# def a function for extract four quadrants from raw Polcam images
def get_quadrants(img_raw,seq_case=6,binning=1):
    # just in case wrong input
    if seq_case not in list(range(8)):
        seq_case = 6

    # all possible pattern of corr. pixels and quadrants
    q_seq = []
    for ind1 in list(range(4)):
        ind4 = 3-ind1
        temp = list(range(4))
        temp.remove(ind1)
        temp.remove(ind4)
        for ind2 in temp:
            q_seq.append((ind1, ind2, 3-ind2, ind4))

    # generate subpixel images corr to each quandrant in order 1-2-3-4
    img_q = []
    for i in list(range(4)):
        temp = img_raw[q_seq[seq_case][i]//2::2, q_seq[seq_case][i]%2::2]
        img_q.append(binning_img(temp,binning))
    return img_q

# def a function for generate half imgs from four quadrants
def quad_to_half(img_q, Mtx_qh = np.matrix([[0.75, 0.75, -0.25, -0.25],
              [0.75, -0.25, 0.75, -0.25],
              [-0.25, -0.25, 0.75, 0.75],
              [-0.25, 0.75, -0.25, 0.75]])):
    # generate half circle images by matrix -- top, left, bottom, right
    img_h = []
    for i in range(len(img_q)):
        temp = Mtx_qh[i,0]*img_q[0]+Mtx_qh[i,1]*img_q[1]+Mtx_qh[i,2]*img_q[2]+Mtx_qh[i,3]*img_q[3]
        img_h.append(temp)
    return img_h

def gen_transferFunctions(shape, mag, NA_img, NA_illu, NA_inner, lbd, ps_cam, angles=[270,180,90,0]):
    # ps_cam = px_cam*2 # camera sensor pixel size um * 2
    # angles should corr. to relative angle btw sample, source mask, and camera
    # may change permutation to find the correct one

    ''' generate grids '''
    # generate grids
    sx = shape[-1] # img size on col-dim
    sy = shape[-2] # img size on row-dim
    ux = (np.arange(sx,dtype='complex64')-(sx//2))
    uy = (np.arange(sy,dtype='complex64')-(sy//2))

    ps = ps_cam/mag # pixel size um

    dfx = (1/ps)/sx # Fs/N = 1/sx/ps
    dfy = (1/ps)/sy

    x = ps*ux
    y = ps*uy

    fx = np.fft.ifftshift(dfx*ux)
    fy = np.fft.ifftshift(dfy*uy)
    fxv, fyv = np.meshgrid(fx,fy)

    ''' pupil and source '''
    # generate and plot pupil & complete source
    pupil = np.array(fxv**2+fyv**2 <= (NA_img/lbd)**2)
    source_all = np.array((fxv**2+fyv**2 <= (NA_illu/lbd)**2) & (fxv**2+fyv**2 >= (NA_inner/lbd)**2))
    # try customize source patter as four split quadrants
    # source_all[abs(fxv)< 2/(2*lbd*40)]=0 # x/(2f*wavelength)
    # source_all[abs(fyv)< 2/(2*lbd*40)]=0

    # generate sources for each image
    sources = []
    for angle in angles:
        temp = np.zeros(shape)
        temp[np.cos(np.deg2rad(angle))*fyv >= np.sin(np.deg2rad(angle))*fxv] = 1
        temp *= source_all
        sources.append(temp)

    ''' generate transfer functions '''
    # generate absorption and phase transfer function
    Hu = []
    Hp = []
    for source in sources:
        # copy from waller codes -- it's basically implementing Chen's PhD thesis Eq 1.18
        FSP_cFP  = F(source*pupil)*F(pupil).conj()
        I0    = (source*pupil*pupil.conj()).sum()
        Hu.append(2.0*IF(FSP_cFP.real)/I0)
        Hp.append(2.0j*IF(1j*FSP_cFP.imag)/I0)
    return Hu, Hp


class pDPC_reconstruction:
    def updateTransferFunc(self, shape, mag, NA_img, NA_illu, NA_inner, lbd, px_cam=3.45):
        self.Hu, self.Hp = gen_transferFunctions(shape, mag, NA_img, NA_illu, NA_inner, lbd, 2*px_cam)
    
    def reconstructPDPC(self, img, seq_case, binning=1, reg_u = 1e-1, reg_p = 5e-3):
        # get half circle imgs for pDPC
        img_hs = quad_to_half(get_quadrants(img,seq_case,binning))
        # normalize & FFT of imgs to delete dc
        norm_imgF = []
        for i in range(len(img_hs)):
            temp = img_hs[i].astype('float32')
            # temp /= ndimage.uniform_filter(temp, size=temp.shape[0]//2)
            temp = (temp - temp.mean())/temp.mean()
            norm_imgF.append(F(temp))
        norm_imgF =  np.asarray(norm_imgF)

        # get a Hu and Hp corresponding to img shape and binning
        Hu = []
        Hp = []
        for i in range(len(img_hs)):
            try:
                Hu.append(binning_img(self.Hu[i], binning))
                Hp.append(binning_img(self.Hp[i], binning))
            except:
                temp = np.zeros((img.shape[0]//(2*binning),img.shape[1]//(2*binning)))
                Hu.append(temp)
                Hp.append(temp)
                print("No {}-th Hu & Hp generated yet. Will just use zeros.".format(i))
        Hu = np.asarray(Hu)
        Hp = np.asarray(Hp)

        # solve inverse problem using least square (l2/Tik)
        # bascially copied from waller codes       
        # calculate by least square methods
        a1 = (Hp*Hp.conj()).sum(axis=0)+reg_p
        a2 = (Hu*Hu.conj()).sum(axis=0)+reg_u
        a3 = (Hu.conj()*norm_imgF).sum(axis=0)
        a4 = (Hp.conj()*norm_imgF).sum(axis=0)
        a5 = (Hu.conj()*Hp).sum(axis=0)
        a6 = (Hu*Hp.conj()).sum(axis=0)

        mu_f = IF((a1*a3-a5*a4)/(a1*a2-a6*a5)).real
        phi_f = IF((a2*a4-a6*a3)/(a1*a2-a6*a5)).real

        return phi_f

#############
def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

# sigmoid-linear unit
def SILIU(x,x0=0.0,width=0.1,p=6): # x presumed numpy array
    z = np.power((x-x0)/width,p)
    z = 1./(1.+np.exp(-z))
    return x*np.power(z,p)
    