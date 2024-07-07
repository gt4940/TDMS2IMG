import time
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from nptdms import TdmsFile

start = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

input_files = sorted(glob.glob('data/*.tdms'))
length_data = len(input_files)
part_size = length_data//size

i_start = rank*part_size
i_end = i_start + part_size

input_files = input_files[i_start:i_end]

if (rank == (size-1)):
    i_end = length_data


def tdms_to_img(input_files, start_channel_num, end_channel_num, height, rank):
    b = 0
    for input_file in input_files:
        time = TdmsFile.read_metadata(input_file).properties['GPSTimeStamp']    # load time info
        tdms_file = TdmsFile(input_file)
        trace = tdms_file.groups()[0].channels()
        data = np.array(trace)[start_channel_num:end_channel_num, :]            # (channel , samples) (406, 30000)
        sample_num = data.shape[1]
        output_path = 'image_1000_406/'
        for i in range(int(sample_num/height)):
            a = i * height
            time_delta = np.timedelta64(a, 'ms')
            name = np_to_datetime(time+time_delta) + '.png'                     # 이미지 파일명에 시간정보 기입
            image = data[:, a:a+height].T                                       # X-T domain처럼 바꿈
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.savefig(output_path + name, format='png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        b += 1
        print(f"{rank}:", f"{ b/len(input_files)*100}% complite")
    return


def np_to_datetime(np_datetime):
    np_datetime = np_datetime.astype(datetime.datetime)
    strf_datetime = np_datetime.strftime('%y%m%d_%H%M%S_%f')[:-3]
    return strf_datetime


desired_pixel = 1299 # Figsize 결정 시 좀 더 작게 이미지가 생성되어 조금 더 크게 설정 - 본인은 직접 테스트해서 결정
dpi = 100
plt.figure(figsize=(desired_pixel/dpi, desired_pixel/dpi), dpi=dpi)

height = 1000
start_channel_num = 243                                                         # 노이즈 심한 부분 제거
end_channel_num = 649                                                           # 시추공 끝

tdms_to_img(input_files, start_channel_num, end_channel_num, height, rank)
print("Total elapsed time", time.time()-start)
