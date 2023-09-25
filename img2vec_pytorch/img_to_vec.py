import torch
import torchvision.models as models
import torchvision.transforms as transforms

class Img2Vec():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    EFFICIENTNET_OUTPUT_SIZES = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560
    }

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512, gpu=0):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer, self.preprocess = self._get_model_and_layer(model, layer)

        if self.device.type == 'cuda':
            self.model.half()
        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if isinstance(img, list):
            a = [self.preprocess(img) for im in img]
            images = (torch.stack(a).to(self.device).half() if self.device.type == 'cuda'
                      else torch.stack(a).to(self.device))
            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name in ['densenet', 'shufflenet', 'efficientnet_b0']:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            elif self.model_name == 'efficientnet_b1':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 8, 8)
            elif self.model_name == 'efficientnet_b2':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 9, 9)
            elif self.model_name == 'efficientnet_b3':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 10, 10)
            elif self.model_name == 'efficientnet_b4':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 12, 12)
            elif self.model_name == 'efficientnet_b5':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 15, 15)
            elif self.model_name == 'efficientnet_b6':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 17, 17)
            elif self.model_name == 'efficientnet_b7':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 19, 19)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[:, :]
                elif (self.model_name == 'densenet'
                      or self.model_name == 'shufflenet'
                      or 'efficientnet' in self.model_name):
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            if self.device.type == 'cuda':
                image = image.half()

            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name in ['densenet', 'shufflenet', 'efficientnet_b0']:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            elif self.model_name == 'efficientnet_b1':
                my_embedding = torch.zeros(1, self.layer_output_size, 8, 8)
            elif self.model_name == 'efficientnet_b2':
                my_embedding = torch.zeros(1, self.layer_output_size, 9, 9)
            elif self.model_name == 'efficientnet_b3':
                my_embedding = torch.zeros(1, self.layer_output_size, 10, 10)
            elif self.model_name == 'efficientnet_b4':
                my_embedding = torch.zeros(1, self.layer_output_size, 12, 12)
            elif self.model_name == 'efficientnet_b5':
                my_embedding = torch.zeros(1, self.layer_output_size, 15, 15)
            elif self.model_name == 'efficientnet_b6':
                my_embedding = torch.zeros(1, self.layer_output_size, 17, 17)
            elif self.model_name == 'efficientnet_b7':
                my_embedding = torch.zeros(1, self.layer_output_size, 19, 19)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[0, :]
                elif (self.model_name == 'densenet'
                      or self.model_name == 'shufflenet'
                      or 'efficientnet' in self.model_name):
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            model = getattr(models, model_name)(weights='DEFAULT')
            preprocess = models.ResNet18_Weights.DEFAULT.transforms()  # same preprocessing transforms for IMAGENET1K_V1
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer, preprocess
        elif model_name == 'resnet-18':
            model = models.resnet18(weights='DEFAULT')
            preprocess = models.ResNet18_Weights.DEFAULT.transforms()
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer, preprocess

        elif model_name == 'alexnet':
            model = models.alexnet(weights='DEFAULT')
            preprocess = models.AlexNet_Weights.DEFAULT.transforms()
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer, preprocess

        elif model_name == 'vgg':
            # VGG-11
            model = models.vgg11_bn(weights='DEFAULT')
            preprocess = models.VGG11_BN_Weights.DEFAULT.transforms()
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer, preprocess

        elif model_name == 'densenet':
            # Densenet-121
            model = models.densenet121(weights='DEFAULT')
            preprocess = models.DenseNet121_Weights.DEFAULT.transforms()
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer, preprocess

        elif "efficientnet" in model_name:
            # efficientnet-b0 ~ efficientnet-b7
            if model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights='DEFAULT')
                preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b1":
                model = models.efficientnet_b1(weights='DEFAULT')
                preprocess = models.EfficientNet_B1_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b2":
                model = models.efficientnet_b2(weights='DEFAULT')
                preprocess = models.EfficientNet_B2_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b3":
                model = models.efficientnet_b3(weights='DEFAULT')
                preprocess = models.EfficientNet_B3_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b4":
                model = models.efficientnet_b4(weights='DEFAULT')
                preprocess = models.EfficientNet_B4_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b5":
                model = models.efficientnet_b5(weights='DEFAULT')
                preprocess = models.EfficientNet_B5_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b6":
                model = models.efficientnet_b6(weights='DEFAULT')
                preprocess = models.EfficientNet_B6_Weights.DEFAULT.transforms()
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(weights='DEFAULT')
                preprocess = models.EfficientNet_B7_Weights.DEFAULT.transforms()
            else:
                raise KeyError('Un support %s.' % model_name)

            if layer == 'default':
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer, preprocess

        elif model_name == 'shufflenet':
            model = models.shufflenet_v2_x0_5(weights='DEFAULT')
            preprocess = models.ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
            if layer == 'default':
                layer = model.conv5[-1]
                self.layer_output_size = model.fc.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer, preprocess

        else:
            raise KeyError('Model %s was not found' % model_name)
