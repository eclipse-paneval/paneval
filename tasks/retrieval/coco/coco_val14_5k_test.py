config = dict(
    dataset_path="nlphuji/mscoco_2014_5k_test_image_text_retrieval",
    split="test",
    processed_dataset_path="retrieval/coco_val14_5k",
    processor="../flickr/process.py",
)

dataset = dict(type="RetrievalBaseDataset", config=config, name="coco_val14_5k_test")

evaluator = dict(type="RetrievalEvaluator")
