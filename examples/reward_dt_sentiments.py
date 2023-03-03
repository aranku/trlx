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

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def add_reward(samples):
        samples["reward"] = list(map(get_positive_score, sentiment_fn(samples["text"])))
        return samples

    reward_percentiles = {
        0: 0,
        1: 0.007793527748435736,
        2: 0.0484938271343709,
        3: 0.9415658831596374,
        4: 0.9895373225212097,
        5: 1.0,
    }

    def prepend_reward(sample):
        tmp = "[0] "
        for i, p in reward_percentiles.items():
            if sample["reward"] > p:
                tmp = f"[{i + 1}] "
        sample["text"] = tmp + sample["text"]
        return sample

    imdb = load_dataset("imdb", split="train+test").map(add_reward, batched=True, batch_size=256).map(prepend_reward)

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        samples = list(map(lambda x: x[4:], samples))
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    trainer = trlx.train(
        samples=imdb["text"],
        eval_prompts=["[5] I don't know much about Hungarian underground"] * 128,
        metric_fn=metric_fn,
        config=config,
    )
    trainer.save_pretrained("reviews-sft")


if __name__ == "__main__":
    main()
