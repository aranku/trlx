# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

import os
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trainer: AcceleratePPOTrainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * config.train.batch_size,
        config=config,
    )

    prompt = "At first glance, the movie seemed like a typical cliché-ridden Hollywood blockbuster, with its flashy visuals and predictable storyline. However, upon watching it, I was pleasantly surprised to discover a nuanced and thought-provoking narrative that kept me engaged from beginning to end"
    token_list = trainer.tokenizer(prompt).input_ids
    decoded = [trainer.tokenizer.decode(token) for token in token_list]
    print("Prompt:", decoded)
    sentiment = reward_fn([prompt,])
    all_tokens = trainer.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = all_tokens.not_equal(trainer.tokenizer.pad_token_id).long().to(device)

    logits, *_, values = trainer.model(all_tokens, attention_mask=attention_mask)
    print("Value of the prompt:", values)
    print("Sentiment of the prompt:", sentiment)


if __name__ == "__main__":
    main()
