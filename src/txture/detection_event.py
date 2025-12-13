class EventFilter:
    def __init__(self, min_conf=0.8, stable_frames=3):
        self.min_conf = min_conf
        self.stable_frames = stable_frames

        self.current_label = None
        self.stable_count = 0

        self.last_fired_label = None
        self.released = True

    def update(self, label, conf):
        # 유효하지 않은 입력 → 릴리스 처리
        if label is None or conf < self.min_conf:
            self.current_label = None
            self.stable_count = 0
            self.released = True
            return None

        if (
            self.last_fired_label is not None
            and label != self.last_fired_label
        ):
            self.released = True

        # 같은 제스처 반복
        if label == self.current_label:
            self.stable_count += 1
        else:
            self.current_label = label
            self.stable_count = 1

        # 안정되지 않은 프레임 → 이벤트 없음
        if self.stable_count < self.stable_frames:
            return None

        # 이미 발동했고 release 안 되었으면 무시
        if not self.released and label == self.last_fired_label:
            return None

        # 이벤트 확정
        self.last_fired_label = label
        self.released = False

        self.stable_count = 0

        return label
