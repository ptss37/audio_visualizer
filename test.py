import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

# 오디오 파라미터 설정
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# 오디오 스트림 열기
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 초기화
fig, ax = plt.subplots()
x = np.linspace(0, RATE//2, CHUNK//2)
y = np.zeros(CHUNK//2)
line, = ax.plot(x, y)
ax.set_ylim(0, 0.01)  # 초기 y축 범위 설정

# 스펙트럼 업데이트 함수 정의
def update(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    spectrum = np.abs(np.fft.fft(data)[:CHUNK//2])
    line.set_ydata(spectrum)

    # y축 범위 동적으로 조절
    current_max = max(spectrum)
    ax.set_ylim(0, current_max + 10)  # 여유를 두기 위해 10을 더합니다.

    return line,

# 애니메이션 생성
ani = FuncAnimation(fig, update, blit=True)

# 플로팅 시작
plt.show()

# 프로그램 종료 시 처리
stream.stop_stream()
stream.close()
p.terminate()
