import os
import scipy.io as sio

root = "/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_test"
save_root = "/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only/test"

os.makedirs(save_root, exist_ok=True)

for i in range(1, 17):
    fov = 'fov%d'%i
    fov_root = os.path.join(root, fov)
    
    for dtype in ['gt_amplitude', 'gt_phase', 'holography']:
            
        if 'gt' in dtype:
            save_path = os.path.join(save_root, dtype)
            os.makedirs(save_path, exist_ok=True)
            
            data_root = os.path.join(fov_root, 'test', dtype, '%s1.mat'%dtype)
            data = sio.loadmat(data_root)[dtype]
            
            sio.savemat(os.path.join(save_path, '%s.mat'%fov), {dtype:data})           
            
            
        else:
            for d in range(5, 21):
                save_path = os.path.join(save_root, dtype, '%d'%d)
                os.makedirs(save_path, exist_ok=True)
                
                data_root = os.path.join(fov_root, 'test', dtype, '%d'%(d), '%s%d.mat'%(dtype, d-4))
                data = sio.loadmat(data_root)[dtype]
                
                sio.savemat(os.path.join(save_path, '%s.mat'%fov), {dtype:data})   
                