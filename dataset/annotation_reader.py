from pathlib import Path
import pandas as pd

class AnnotationLoader:
    STAGE_ORDER = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "tM", "tSB", "tB", "tEB", "tHB"]
    STAGE_TO_LABEL = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    def __init__(self, annotation_root):
        self.annotation_root = Path(annotation_root)
        self.cache = {}
    def _build_cache(self, embryo):
        csv_path = self.annotation_root / f"{embryo}_phases.csv"
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["stage", "start", "end"]
        )
        label_map = {}
        for _, row in df.iterrows():
            stage = row["stage"]
            if stage not in self.STAGE_TO_LABEL:
                continue
            if pd.isna(row["start"]):
                continue
            if pd.isna(row["end"]):
                continue
            start = int(row["start"])
            end = int(row["end"])
            label = self.STAGE_TO_LABEL[stage]
            for run in range(start, end + 1):
                label_map[run] = label
        self.cache[embryo] = label_map

    def get_label(self, embryo, run):
        if embryo not in self.cache:
            self._build_cache(embryo)
        return self.cache[embryo].get(run, -1)

if __name__ == "__main__":
    ...