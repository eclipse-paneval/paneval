config = dict(
    dataset_path="nlphuji/flickr_1k_test_image_text_retrieval",
    split="test",
    processed_dataset_path="retrieval/flickr30k_1k",
    processor="process.py",
)

dataset = dict(type="RetrievalBaseDataset", config=config, name="flickr_1k_test")

evaluator = dict(type="RetrievalEvaluator")
