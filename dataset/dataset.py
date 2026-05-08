
import pandas as pd
import datasets
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

_VERSION = datasets.Version("0.0.1")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "target": datasets.Image(),
        "source": datasets.Image(),
        "prompt": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class MyData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        metadata_path = os.path.join(_HERE, "prompt.jsonl")
        images_dir = _HERE
        conditioning_images_dir = _HERE
        dl_manager.download_and_extract
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["prompt"]

            image_path = row["target"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["source"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["source"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()


            yield row["target"], {
                "prompt": text,
                "target": {
                    "path": image_path,
                    "bytes": image,
                },
                "source": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }