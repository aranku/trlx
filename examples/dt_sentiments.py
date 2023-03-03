import os
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams)

    def prepend_label(sample):
        if sample["label"] == 1:
            sample["text"] = "[POS] " + sample["text"]
        else:
            sample["text"] = "[NEG] " + sample["text"]
        return sample

    imdb = load_dataset("imdb", split="train+test").map(prepend_label)

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        samples = list(map(lambda x: x.split("] ")[-1], samples))
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    trainer = trlx.train(
        samples=imdb["text"],
        eval_prompts=["[POS] I don't know much about Hungarian underground"] * 128,
        metric_fn=metric_fn,
        config=config,
    )
    trainer.save_pretrained("reviews-sft")


if __name__ == "__main__":
    main()
