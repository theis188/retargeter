import numpy as np
from scipy.misc import imread, imsave, imresize
import pylab as plt
from argparse import ArgumentParser

class Retargeter(object):
	def __init__(self):
		self.rescale_min = 400
	def _LoadImage(self,path):
		self.downsample_scale = None
		self.d_im = None
		self.downsample = False
		self.o_im = imread(path).astype(np.float32)
	def _DownSizeImg(self):
		self.downsample_scale = min(self.o_im.shape[:2])/float(self.rescale_min)
		self.d_im = imresize(self.o_im, 1/self.downsample_scale).astype(np.float32)
	def _DCDY(self,im):
		ret = np.zeros( im.shape )
		ret[1:,:,:] = np.abs(im[1:,:,:] - im[:-1,:,:])
		ret[0,:,:] = ret[1,:,:]
		return ret
	def _DCDX(self,im):
		ret = np.zeros( im.shape )
		ret[:,1:,:] = np.abs(im[:,1:,:] - im[:,:-1,:])
		ret[:,0,:] = ret[:,1,:]
		return ret
	def _Deriv(self,im=None):
		if im is None:
			im = self.im
		vert = self._DCDY(im)
		horiz = self._DCDX(im)
		return ((vert+horiz) / 2.)
	def _GetSubDim(self,crop_shape):
		im = self.im
		aspect_im = float( im.shape[1] ) / im.shape[0]
		aspect_crop = float( crop_shape[1] ) / crop_shape[0]
		if aspect_crop > aspect_im:
			ret = ( int(im.shape[1]/aspect_crop) , im.shape[1] )
		else:
			ret = ( im.shape[0] , int(im.shape[0]*aspect_crop) )
		self.sub_dim = ret
	def _GetCropCoords(self,crop_shape):
		if self.downsample:
			crop_shape = [int(i/self.downsample_scale ) for i in crop_shape]
		self._GetSubDim(crop_shape)
		im = self.im
		y_windows = im.shape[0] - self.sub_dim[0] + 1
		x_windows = im.shape[1] - self.sub_dim[1] + 1
		self.ders = self._Deriv()
		if y_windows>1:
			return self._GetMax(y_windows,0)
		else:
			return self._GetMax(x_windows,1)
	def _GetMax(self,windows,ind):
		offset = [0,0]
		max_sum = 0
		max_coord = 0
		sums = []
		for i in range(windows):
			ys,xs = offset
			xe = xs + self.sub_dim[1]
			ye = ys + self.sub_dim[0]
			curr_sum = np.sum( self.ders[ ys:ye , xs:xe , : ] )
			sums.append(curr_sum)
			if curr_sum > max_sum:
				max_sum = curr_sum
				box = (xs,xe,ys,ye)
			offset[ind] += 1
		return box
	def _DynCrop(self,im_path,crop_shape,new_path):
		self._LoadImage(im_path)
		save_mag = crop_shape[0]*1
		if min(self.o_im.shape[:2])>self.rescale_min:
			self.downsample = True
			self._DownSizeImg()
			self.im = self.d_im
			print('Downsampled',self.downsample_scale)
		else:
			self.im = self.o_im
		box = self._GetCropCoords(crop_shape)
		if self.downsample:
			box = [int(i*self.downsample_scale) for i in box]
		xs,xe,ys,ye = box
		new_im = self.o_im[ ys:ye , xs:xe , :]
		new_im = imresize(new_im, float(save_mag)/new_im.shape[0] )
		imsave(new_path,new_im)
	def _CenterCrop(self,im_path,crop_shape,new_path):
		self._LoadImage(im_path)
		self.im = self.o_im
		self._GetSubDim(crop_shape)
		sub_dim = self.sub_dim
		im = self.im
		xs = int( (im.shape[1]-sub_dim[1])/2 )
		xe = int( (im.shape[1]-sub_dim[1])/2 ) + sub_dim[1]
		ys = int( (im.shape[0]-sub_dim[0])/2 )
		ye = int( (im.shape[0]-sub_dim[0])/2 ) + sub_dim[0]
		new_im = im[ ys:ye , xs:xe , :]
		new_im = imresize(new_im, float(crop_shape[0])/new_im.shape[0] )
		imsave(new_path,new_im)

if __name__ == '__main__':
	parser = ArgumentParser(description='Process images.')
	parser.add_argument('--source', help="source image path.")
	parser.add_argument('--dyn', help= 'output path for dynamic crop')
	parser.add_argument('--cen', help= 'output path for center crop')
	parser.add_argument('--dim', help= 'height,width in pixel ie: 300,300')
	args = parser.parse_args()
	r = Retargeter()
	dims = tuple( int(i) for i in args.dim.split(',') )
	path = args.source
	if args.dyn:
		r._DynCrop(path,dims,args.dyn)
	if args.cen:
		r._CenterCrop(path,dims,args.cen)