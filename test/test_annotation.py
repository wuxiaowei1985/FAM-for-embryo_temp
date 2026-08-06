from dataset.annotation_reader import AnnotationLoader
annotation_root = r"I:\datasets\胚胎\南特704\embryo_dataset_annotations"

def test_annotation():
    reader = AnnotationLoader(annotation_root)
    for run in [5, 25, 90, 100, 180]:
        print(run, reader.get_label("AA83-7", run))

if __name__ == "__main__":
    test_annotation()