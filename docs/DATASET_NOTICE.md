# Dataset And Content Notice

This repository does not redistribute third-party datasets, raw images, downloaded archives, LMDB stores, model checkpoints, or large generated JSON/JSONL artifacts.

Users are responsible for obtaining third-party datasets from their original sources and complying with each dataset's license, terms of use, citation requirements, and redistribution restrictions.

## Third-Party Data

The code expects local copies of datasets such as CIRR, FashionIQ, MTCIR, LaSCo, LLaVA-pretrain, COCO-derived resources, and related image/text assets. These files are ignored by git and are not covered by this repository's MIT code license.

## Generated Data

Generated annotations, rewrites, evaluation metadata, and other data created by this project are intended to be released under Creative Commons Attribution 4.0 International (CC BY 4.0), unless a specific file states otherwise.

Because generated data may be derived from, aligned to, or paired with third-party datasets, users must also respect the original dataset terms. The CC BY 4.0 notice applies only to rights controlled by this project author.

## Content Warning

Training images used with this project come from the LLaVA-pretrain dataset available on Hugging Face. LLaVA-pretrain is a subset of Google's Conceptual Captions 3M (CC3M).

These images may contain harmful, unsafe, offensive, explicit, biased, or otherwise sensitive content. Users should treat them as uncurated web-scale data and apply appropriate safety review, filtering, access control, and ethical handling before using or sharing derived artifacts.

Do not assume that local datasets are safe for all audiences or use cases.
