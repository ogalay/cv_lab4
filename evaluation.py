from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS

def get_metrics(enh_img, gt_img):
    '''
        Функция для измерения метрик. Расчитывает среднее значение по батчу
        
        Arguments:
            :enh_img: тензор размером (BxCxHxW), содержащий
                батч изображений после обработки
            
            :gt_img: тензор размером (BxCxHxW), содержащий
                батч ground-truth изображений
    '''
    # PSNR
    psnr = PSNR()
    psnr_score = psnr(enh_img, gt_img)
    # SSIM
    ms_ssim = SSIM()
    ssim_score = ms_ssim(enh_img, gt_img)
    # LILPS
    lpips = LPIPS('alex')
    lpips_score = lpips(enh_img, gt_img).item()
    return psnr_score, ssim_score, lpips_score
