import os
import io
import json
import numpy as np
import imageio
import albumentations as A
from m2unet import m2unet
import torch
import torch.nn as nn
import torch.optim as optim
from bioimageio.core.build_spec import build_model


class M2UnetInteractiveModel:
    def __init__(
        self,
        model_dir=None,
        type="m2unet",
        model_config=None,
        resume=True,
        pretrained_model=None,
        save_freq=None,
        learning_rate=0.001,
        default_save_path=None,
        model_id=None,
        data_root=None,
        **kwargs,
    ):
        assert type == "m2unet"
        assert model_dir is not None
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.default_save_path = default_save_path

        assert self.default_save_path is not None
        self.data_root = data_root or os.path.join(
            os.path.dirname(self.default_save_path), "data"
        )
        self._config = {
            "default_save_path": self.default_save_path,
            "model_config": model_config,
        }
        self._history = []
        gpu = torch.cuda.is_available()
        if gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if save_freq is None:
            if gpu:
                self.save_freq = 2000
            else:
                self.save_freq = 300
        else:
            self.save_freq = save_freq

        if resume:
            if self.default_save_path and os.path.exists(self.default_save_path):
                pretrained_model = self.default_save_path

        if pretrained_model:
            assert os.path.exists(pretrained_model), "Pretrained model path not found."
            self._config["parent"] = os.path.basename(pretrained_model)
            self.load(pretrained_model)
            if resume and model_config:
                # make sure model_config has not changed
                for k in model_config:
                    if k in ["loss", "optimizer", "augmentation"]:
                        continue
                    if k in self.model_config:
                        assert (
                            self.model_config[k] == model_config[k]
                        ), "Model config has changed, please make sure you used the same model_config as before or set `resume=False`."
        else:
            self.init_model(model_config, model_id)

        os.makedirs(self.model_dir, exist_ok=True)

        self._iterations = 0

        # make sure we override the model_id
        self.set_config("model_id", model_id)

    def init_model(self, model_config, model_id):
        """initialize the model
        Parameters
        --------------
        None

        Returns
        ------------------
        None
        """
        assert "type" in model_config
        self.type = model_config["type"]
        assert self.type == "m2unet"
        assert model_config is not None

        model_kwargs = {}
        model_kwargs.update(model_config)
        model_kwargs["name"] = f"{self.type}-{model_id}"
        augmentation_config = model_kwargs.get("augmentation")
        del model_kwargs["loss"]
        del model_kwargs["optimizer"]
        del model_kwargs["augmentation"]
        del model_kwargs["type"]
        self.model = m2unet(**model_kwargs).to(self.device)
        self.model_config = model_config
        self.transform = augmentation_config and A.from_dict(augmentation_config)
        loss_class = getattr(nn, self.model_config["loss"]["name"])
        optimizer_class = getattr(
            optim, self.model_config["optimizer"]["name"]
        )
        loss_instance = (
            loss_class(**self.model_config["loss"]["kwargs"])
            if "kwargs" in self.model_config["loss"]
            else loss_class
        )
        self.optimizer = optimizer_class(self.model.parameters(), **self.model_config["optimizer"]["kwargs"])
        self.criterion = loss_instance

    def get_model_id(self):
        """get the model id"""
        return self._config.get("model_id")

    def set_config(self, key, value=None):
        """set the config"""
        if value is None and isinstance(key, dict):
            # assume key is a dictionary
            self._config.update(key)
        else:
            self._config[key] = value

    def get_config(self):
        """augment the images and labels
        Parameters
        --------------
        None

        Returns
        ------------------
        config: dict
            a dictionary contains the following keys:
            1) `batch_size` the batch size for training
        """
        return self._config

    def transform_labels(self, label_image):
        """transform the labels which will be used as training target
        Parameters
        --------------
        label_image: array [width, height, channel]
            a label image

        Returns
        ------------------
        array [width, height, channel]
            the transformed label image
        """
        return label_image

    def augment(self, images, targets):
        """augment the images and labels
        Parameters
        --------------
        images: array [batch_size, width, height, channel]
            a batch of input images

        labels: array [batch_size, width, height, channel]
            a batch of labels

        Returns
        ------------------
        (images, labels) both are: array [batch_size, width, height, channel]
            augmented images and labels
        """
        transformed_images = []
        transformed_targets = []
        for i in range(len(images)):
            if self.transform:
                transformed = self.transform(image=images[i], mask=targets[i])
            else:
                transformed = {"image": images[i], "mask": targets[i]}
            transformed_images.append(transformed["image"])
            transformed_targets.append(transformed["mask"])

        return np.stack(transformed_images, axis=0), np.stack(
            transformed_targets, axis=0
        )

    def train(
        self,
        images,
        targets,
    ):
        imgi, tgs = self.augment(images, targets)
        imgi = torch.from_numpy(imgi.transpose(0, 3, 1, 2))
        tgs = torch.from_numpy(tgs.transpose(0, 3, 1, 2).astype(np.float32))
        imgi = imgi.to(device=self.device, dtype=torch.float32)
        tgs = tgs.to(device=self.device, dtype=torch.float32)
        masks_pred = self.model(imgi)
        train_loss = self.criterion(masks_pred, tgs)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        self._iterations += len(images)
        if self._iterations % self.save_freq == 0:
            self.save()
        return train_loss.item()

    def train_on_batch(self, X, y):
        """train the model for one iteration
        Parameters
        --------------
        X: array [batch_size, width, height, channel]
            the input image with 2 channels

        y: array [batch_size, width, height, channel]
            the mask (a.k.a label) image with one unique pixel value for one object
            if the shape is [1, width, height], then y is the label image
            otherwise, it should have channel=4 where the 1st channel is the label image
            and the other 3 channels are the precomputed flow image

        Returns
        ------------------
        loss value
        """
        assert X.shape[0] == y.shape[0] and X.ndim == 4

        return self.train(X, y)

    def predict(
        self,
        X,
        **kwargs,
    ):
        """predict the model for one input image
        Parameters
        --------------
        X: array [batch_size, width, height, channel]
            the input image with 2 channels

        Returns
        ------------------
        array [batch_size, width, height, channel]
            the predicted label image
        """
        assert X.ndim == 4
        X = torch.from_numpy(X.transpose(0, 3, 1, 2)).to(device=self.device, dtype=torch.float32)
        outputs = self.model(X, **kwargs)
        return outputs.detach().cpu().numpy().transpose(0, 2, 3, 1)

    def can_overwrite(self, token):
        """check if the model can be overwritten"""
        model_token = self._config.get("model_token")
        if not model_token or model_token == token:
            return True
        return False

    def get_weights(self):
        """Get model weights in bytes."""
        buf = io.BytesIO()
        self.model.save(buf)
        buf.seek(0)
        return buf.read()

    def save(self, file_path=None):
        """save the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        if file_path is None and self.default_save_path is not None:
            file_path = self.default_save_path
        assert isinstance(file_path, str)
        torch.save(self.model.state_dict(), file_path)
        with open(os.path.join(os.path.dirname(file_path), "config.json"), "w") as fil:
            fil.write(json.dumps(self.get_config()))

        # append the history and clear it
        with open(os.path.join(os.path.dirname(file_path), "history.csv"), "a") as fil:
            # fil.write("image,labels,steps,seed\n")
            for history in self._history:
                fil.write(",".join(map(str, history)) + "\n")
        self._history = []

    def _save_data_array(self, data):
        buf = io.BytesIO()
        np.save(buf, data)
        hash_code = str(hash(buf))
        file_name = os.path.join(self.data_root, hash_code + ".npy")
        if not os.path.exists(file_name):
            with open(file_name, "wb") as fil:
                fil.write(buf.getbuffer())
        return hash_code

    def record_training_history(self, image, labels, steps, seed):
        image_hash = self._save_data_array(image)
        labels_hash = self._save_data_array(labels)
        self._history.append((image_hash, labels_hash, steps, seed))
        return {
            "image": image_hash,
            "labels": labels_hash,
            "steps": steps,
            "seed": seed,
        }

    def load(self, file_path):
        """load the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        with open(os.path.join(os.path.dirname(file_path), "config.json"), "r") as fil:
            self.set_config(json.loads(fil.read()))
        self.type = self._config["model_config"]["type"]
        self.model_config = self._config["model_config"]
        self.init_model(self.model_config, self._config.get("model_id"))
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))

    def export(
        self,
        format,
        output_path,
        test_image,
        test_mask,
        license="CC-BY-4.0",
        documentation=None,
        description=None,
        author_info=None,
    ):
        """export the model into different format
        Parameters
        --------------
        format: string
            the model format to be exported
        output_path: string
            the file path to the exported model

        Returns
        ------------------
        None
        """
        assert format == "bioimageio"
        root = os.path.dirname(output_path)
        name = f"{self.type}-{self._config.get('model_id')}"
        assert test_image.ndim == 4 and test_mask.ndim == 4
        np.save(f"{root}/test_image.npy", test_image)
        np.save(f"{root}/test_mask.npy", test_mask)
        with open(f"{root}/README.md", "w") as fil:
            if documentation:
                fil.write(f"{documentation}\n")
            else:
                fil.write(f"# {name}\n")
                if description:
                    fil.write(f"{description}\n")
        test_image = test_image[0].astype("uint8")
        test_mask = test_mask[0].astype("uint8")
        if test_image.shape[2] > test_mask.shape[2]:
            test_image = test_image[:, :, : test_mask.shape[2]]
        elif test_image.shape[2] < test_mask.shape[2]:
            test_mask = test_mask[:, :, : test_image.shape[2]]
        print(test_image.shape, test_mask.shape)
        combined = np.concatenate([test_image, test_mask], axis=1)
        if combined.shape[2] == 1:
            combined = np.concatenate([combined, combined, combined], axis=2)
        elif combined.shape[2] == 2:
            combined = np.concatenate(
                [
                    combined,
                    np.zeros(
                        [combined.shape[0], combined.shape[1], 1], dtype=combined.dtype
                    ),
                ],
                axis=2,
            )
        elif combined.shape[2] > 3:
            combined = combined[:, :, :3]
        imageio.imwrite(f"{root}/cover.png", combined)
        if self._config.get("parent"):
            parent = (self._config.get("parent"), "")
        else:
            parent = None
        build_model(
            self.default_save_path,
            source=__file__ + ":M2UnetInteractiveModel",
            weight_type="keras_hdf5",
            test_inputs=[f"{root}/test_image.npy"],
            test_outputs=[f"{root}/test_mask.npy"],
            output_path=output_path,
            name=name,
            description=description
            or f"A {self.type} model trained with the BioEngine.",
            authors=[author_info] if author_info else [],
            license=license or "CC-BY-4.0",
            documentation=f"{root}/README.md",
            covers=[f"{root}/cover.png"],
            tags=["unet"],
            cite={},
            parent=parent,
            root=root,
            model_kwargs=None,
            preprocessing=None,
            postprocessing=None,
        )
        return open(output_path, "rb").read()

    def finalize(self):
        self.model = None
