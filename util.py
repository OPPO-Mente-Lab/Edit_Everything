from PIL import Image
import copy
import numpy as np
def trans_to_square(read_file): # 创建一个函数用来产生所需要的正方形图片转化
    image = Image.open(read_file).convert('RGB')   # 导入图片
    w, h = image.size  # 得到图片的大小
    new_image = Image.new('RGB', size=(max(w, h), max(w, h)),color= 'white')  # 新建图片填充白色
    box = (int(abs(w - h))//2, 0) if w < h else (0,int(abs(w - h))//2)#(0, length)  # 放在box中
    new_image.paste(image, box)                #产生新的图片
    # new=new.resize((256,256))      #如果需要可以缩小图片方便训练
    return new_image
def trans_to_square_inputImage(image): # 创建一个函数用来产生所需要的正方形图片转化
    w, h = image.size  # 得到图片的大小
    new_image = Image.new('RGB', size=(max(w, h), max(w, h)),color= 'white')  # 新建图片填充白色
    box = (int(abs(w - h))//2, 0) if w < h else (0,int(abs(w - h))//2)#(0, length)  # 放在box中
    new_image.paste(image, box)                #产生新的图片
    # new=new.resize((256,256))      #如果需要可以缩小图片方便训练
    return new_image

def expand_mask_v2(mask,expand_length):
    
    mask_expand=copy.deepcopy(mask)
    h,w=mask.shape
    edit_point=set()
    for i in range(h):
        for j in range(w):
            if mask[i][j]:
                for edit_i in range(-expand_length,expand_length+1):
                    for edit_j in range(-expand_length,expand_length+1):
                        # print(edit_i,edit_j)
                        edit_point.add((i+edit_i,j+edit_j))
                        # if h>i+edit_i>=0 and w>j+edit_j>=0 and (not mask_expand[i+edit_i][j+edit_j]):
                        #     mask_expand[i+edit_i][j+edit_j]=True
                # break
    for go_i,go_j in edit_point:
        if h>go_i>=0 and w>go_j>=0 and (not mask_expand[go_i][go_j]):
            mask_expand[go_i][go_j]=True
    return mask_expand
def expand_mask(mask,expand_length):
    
    mask_expand=copy.deepcopy(mask)
    h,w=mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j]:
                for edit_i in range(-expand_length,expand_length+1):
                    for edit_j in range(-expand_length,expand_length+1):
                        # print(edit_i,edit_j)
                        if h>i+edit_i>=0 and w>j+edit_j>=0 and (not mask_expand[i+edit_i][j+edit_j]):
                            mask_expand[i+edit_i][j+edit_j]=True
                # break
    return mask_expand

def write_segment_mask(all_segments,image_path_):
    meta_image=Image.open(image_path_).convert('RGB')
    drow_image=np.array(meta_image)
    white_down=np.full_like(drow_image,255, dtype=np.uint8)
    h_,w_,alpha=drow_image.shape[0],drow_image.shape[1],0.5
    for mask_item in all_segments:
        assert mask_item['segmentation'].shape==drow_image.shape[:2]
        color = list(np.random.choice(range(256), size=3))
        for i_ in range(h_):
            for j_ in range(w_):
                if mask_item['segmentation'][i_][j_]:
                    # drow_image[i_][j_]=np.array(color,dtype=np.uint8)
                    white_down[i_][j_]=np.array(color,dtype=np.uint8)

    seg_mean_jpg=Image.fromarray(white_down)
    return seg_mean_jpg
def get_mask(mask_array,image_path_,reverse=False):
    a,b=np.where(mask_array==True)
    get_image=Image.open(image_path_).convert('RGB')
    image_data_=np.array(get_image)
    np_mask=np.full_like(image_data_,0, dtype=np.uint8)
    for i_,j_ in zip(a,b):
        np_mask[i_][j_]=np.array([255,255,255],dtype=np.uint8)
    raw_np_mask=np.full_like(image_data_,255, dtype=np.uint8)
    reverse_election=False 
    if reverse:
        np_mask=np.absolute(255-np_mask)
    mask_img_=Image.fromarray(np_mask)
    return mask_img_