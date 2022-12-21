import torch
from learners.fine_tune.src.mnasnet import film_mnasnet1_0, mnasnet1_0
from learners.fine_tune.src.resnet import film_resnet18, resnet18
from learners.fine_tune.src.convnet import ConvnetFeatureExtractor


def create_feature_extractor(feature_extractor_family, feature_adaptation, pretrained_path):
    if feature_adaptation == "film":
        if feature_extractor_family == "mnasnet":
            feature_extractor = film_mnasnet1_0(
                pretrained=True,
                progress=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )

        else:
            feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )

    else:  # no adaptation
        if feature_extractor_family == "mnasnet":
            feature_extractor = mnasnet1_0(
                pretrained=True,
                progress=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )

        elif feature_extractor_family == "resnet":
            feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )

        elif feature_extractor_family == "resnet18":
            from extras.resnet import resnet18_alt
            feature_extractor = resnet18_alt(
                pretrained=True,
                pretrained_model_path=pretrained_path
            )

        elif feature_extractor_family == "resnet34":
            from extras.resnet import resnet34
            feature_extractor = resnet34(
                pretrained=True,
                pretrained_model_path=pretrained_path
            )

        elif feature_extractor_family == "vgg11":
            from extras.vgg import vgg11_bn
            feature_extractor = vgg11_bn(
                pretrained=True,
                pretrained_model_path=pretrained_path
            )

        elif feature_extractor_family == "maml_convnet":
            feature_extractor = ConvnetFeatureExtractor(3, 32)
            saved_model_dict = torch.load(pretrained_path)
            feature_extractor.load_state_dict(saved_model_dict)

        elif feature_extractor_family == "protonets_convnet":
            feature_extractor = ConvnetFeatureExtractor(3, 64)
            saved_model_dict = torch.load(pretrained_path)
            feature_extractor.load_state_dict(saved_model_dict)

        else:
            feature_extractor = None

    # Freeze the parameters of the feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    feature_extractor.eval()

    return feature_extractor
