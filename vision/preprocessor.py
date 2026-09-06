"""
영상 전처리 모듈 (preprocessor.py)
"""
import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, clahe_clip_limit=2.5, clahe_tile_grid_size=(8, 8)):
        """
        영상 전처리기 초기화
        :param clahe_clip_limit: CLAHE 대비 제한 임계값
        :param clahe_tile_grid_size: CLAHE 타일 그리드 크기
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
        
        # 샤프닝 커널 (외곽선 강화)
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def to_gray(self, bgr_image: np.ndarray) -> np.ndarray:
        """컬러 BGR 이미지를 그레이스케일로 변환"""
        if len(bgr_image.shape) == 2:
            return bgr_image
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        - 조명 편차, 그림자, 국소적 어두움/밝음 문제를 해결하여 마커의 대비를 극대화
        """
        if len(gray_image.shape) == 3:
            gray_image = self.to_gray(gray_image)
        return self.clahe.apply(gray_image)

    def apply_sharpen(self, gray_image: np.ndarray) -> np.ndarray:
        """
        샤프닝 필터 적용 (모션 블러 및 카메라 포커스 흐림 보정)
        """
        return cv2.filter2D(gray_image, -1, self.sharpen_kernel)

    def adjust_gamma(self, gray_image: np.ndarray, gamma: float = 1.3) -> np.ndarray:
        """감마 보정 (어두운 영역 디테일 복원)"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(gray_image, table)

    def get_multi_stage_candidates(self, bgr_or_gray: np.ndarray):
        """
        ArUco 검출을 위한 다단계 전처리 후보 이미지들을 순차적으로 생성 (Generator)
        1단계: 기본 그레이스케일 (빠름)
        2단계: CLAHE 적용 이미지 (조명/그림자 극복)
        3단계: CLAHE + 샤프닝 (모션 블러 극복)
        4단계: 감마 보정 + 샤프닝 (극단적 조도 극복)
        """
        gray = self.to_gray(bgr_or_gray)
        
        # 1단계: 원본 Gray
        yield "raw_gray", gray
        
        # 2단계: CLAHE Gray
        clahe_gray = self.apply_clahe(gray)
        yield "clahe_gray", clahe_gray
        
        # 3단계: CLAHE + Sharpen
        sharp_gray = self.apply_sharpen(clahe_gray)
        yield "sharp_clahe", sharp_gray
        
        # 4단계: Gamma + Sharpen
        gamma_gray = self.adjust_gamma(gray, gamma=1.4)
        yield "gamma_sharp", self.apply_sharpen(gamma_gray)
