# environment.py
import numpy as np

class WirelessEnv:
    def __init__(self):
        self.K = 4  # số cell trong một sector
        # Thông số công suất của 4 loại cell (P0, Pr, P_idle)
        self.P0 = np.array([117.3, 136.83, 62.8, 62.8])
        self.Pr = np.array([137.0, 146.0, 119.8, 119.8])
        self.Pidle = np.array([33.2, 50.1, 37.3, 37.3])
        self.time = 0  # phút tính từ đầu tuần
        self.max_time = 24*60*7  # giả sử tuần 1 tuần
        self.reset()

    def reset(self):
        self.time = 0
        # Tất cả cells ban đầu ở trạng thái ON
        self.modes = np.ones(self.K, dtype=int)
        # Tạo ngẫu nhiên tải ban đầu [0,1]
        self.loads = np.random.rand(self.K)
        # Lịch sử các tải (5 bước trước, ban đầu trùng hiện tại)
        self.hist = [self.loads.copy() for _ in range(5)]
        return self._get_state()

    def _get_state(self):
        hour = (self.time // 60) % 24
        day = (self.time // (60*24)) % 7
        # one-hot giờ và ngày
        hour_vec = np.eye(24)[hour]
        day_vec = np.eye(7)[day]
        # trạng thái bật/tắt của các ô
        mode_vec = self.modes.copy()
        # tải hiện tại và lịch sử (5 bước trước)
        loads = np.concatenate((self.loads, *self.hist))
        # Nếu giả lập 1 sector duy nhất, không cần mã hóa sector.
        return np.concatenate([hour_vec, day_vec, mode_vec, loads])

    def step(self, action):
        # Cập nhật ngưỡng: mỗi cặp (off, on)
        # Giả sử action trong [0,1]^8 (vì 2K=8)
        th_off = np.minimum(action[0::2], action[1::2])
        th_on  = np.maximum(action[0::2], action[1::2])
        # Điều chỉnh trạng thái của từng ô
        for k in range(self.K):
            if self.modes[k] == 1 and self.loads[k] < th_off[k]:
                self.modes[k] = 0
            elif self.modes[k] == 0 and self.loads[k] > th_on[k]:
                self.modes[k] = 1
        # Tính reward: tính công suất tiêu thụ
        power = 0
        for k in range(self.K):
            if self.modes[k] == 1:
                power += (self.P0[k] + self.Pr[k] * self.loads[k])
            else:
                power += self.Pidle[k]
        # Giả sử đơn giản: mỗi bước có U = tải*K * constant user count
        users = int(np.round(self.loads.sum() * 10))
        handovers = np.count_nonzero(self.modes != (self.modes))
        # Kiểm tra ràng buộc throughput (bỏ qua chi tiết, giả sử luôn thỏa mãn)
        if True:
            reward = -1.0 * power + 0.1 * users - 0.1 * handovers
        else:
            reward = -10.0
        # Cập nhật tải cho bước tiếp (ví dụ dao động theo giờ)
        self.time += 60  # mỗi step = 1 giờ
        # Cập nhật tải mới: ví dụ hàm sin theo giờ + nhiễu
        t = (self.time % (24*60)) / 60.0
        base = 0.5 + 0.4 * np.sin(2*np.pi*(t/24))
        self.loads = np.clip(base + 0.1*np.random.randn(self.K), 0, 1)
        # Cập nhật lịch sử
        self.hist.pop(0)
        self.hist.append(self.loads.copy())
        next_state = self._get_state()
        done = (self.time >= self.max_time)
        return next_state, reward, done, {}
