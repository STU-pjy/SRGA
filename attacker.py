import numpy as np
import torch
import torch.nn as nn
import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import cv2
import torchvision
import random
import sys
from ultralytics import YOLO


class SRGAttacker():
    def __init__(self, model, img_attacker, txt_attacker, max_regions=1):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.max_regions = max_regions

        self.yolo_model = YOLO('your_path')
        self.yolo_model.conf = 0.7
        self.yolo_model.iou = 0.7

        self.yolo_classes = {
            #Modify the dictionary mapping of YOLO categories based on the downloaded YOLO model weights
            '...': '...',
        }

        self.save_counter = 0

    def extract_object_from_text(self, text):
        words = text.lower().split()

        for word in words:
            if word in self.yolo_classes:
                return self.yolo_classes[word]
        return None

    def get_multiple_key_regions_with_yolo(self, img, text, max_regions=None):
        if max_regions is None:
            max_regions = self.max_regions

        object_class = self.extract_object_from_text(text)
        if object_class is None:
            return [torch.tensor([0, 0, img.shape[2], img.shape[1]])]

        img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        results = self.yolo_model(img_np)

        if len(results) == 0:
            return [torch.tensor([0, 0, img.shape[2], img.shape[1]])]

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return [torch.tensor([0, 0, img.shape[2], img.shape[1]])]

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        class_names = result.names

        key_regions = []
        region_info = []

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            class_name = class_names[cls_id]

            if class_name == object_class or self.is_semantically_related(class_name, object_class):
                if conf >= 0.7:
                    key_regions.append(torch.tensor([box[0], box[1], box[2], box[3]]))
                    region_info.append({
                        'box': box,
                        'class_name': class_name,
                        'confidence': conf
                    })
                    if len(key_regions) >= max_regions:
                        break

        if not key_regions:
            return [torch.tensor([0, 0, img.shape[2], img.shape[1]])]

        return key_regions

    def is_semantically_related(self, class1, class2):
        semantic_groups = {
            '...':['...']
        }

        for group in semantic_groups.values():
            if class1 in group and class2 in group:
                return True
        return False

    def filter_texts_by_key_regions(self, texts, key_region_classes):
        return texts
        # filtered_texts = []
        # for text in texts:
        #     keep_text = False
        #     for key_class in key_region_classes:
        #         if key_class and (key_class in text.lower() or
        #                           any(self.is_semantically_related(key_class, word)
        #                               for word in text.lower().split())):
        #             keep_text = True
        #             break
        #     if keep_text:
        #         filtered_texts.append(text)
        # return filtered_texts if filtered_texts else texts

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        self.save_counter = 0

        key_regions_list = []
        key_region_classes_list = []

        for i, img in enumerate(imgs):
            img_texts = [txts[j] for j in range(len(txt2img)) if txt2img[j] == i]
            if img_texts:
                main_text = self.select_most_relevant_text(img_texts)
                key_regions = self.get_multiple_key_regions_with_yolo(img.unsqueeze(0), main_text)
                key_regions_list.append(key_regions)

                key_class = self.extract_object_from_text(main_text)
                key_region_classes_list.append([key_class] * len(key_regions))
            else:
                key_regions_list.append([torch.tensor([0, 0, img.shape[1], img.shape[2]])])
                key_region_classes_list.append([None])

        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]

        filtered_txts = []
        for i, txt in enumerate(txts):
            # img_idx = txt2img[i]
            # key_classes = key_region_classes_list[img_idx]
            # keep_text = False
            # for key_class in key_classes:
            #     if key_class and (key_class in txt.lower() or
            #                       any(self.is_semantically_related(key_class, word)
            #                           for word in txt.lower().split())):
            #         keep_text = True
            #         break
            filtered_txts.append(txt)


        adv_imgs, adv_txts = self.iterative_attack(
            imgs, filtered_txts, txt2img, img_supervisions,
            key_regions_list, key_region_classes_list, device, max_length, scales
        )

        return adv_imgs, adv_txts

    def select_most_relevant_text(self, texts):
        best_text = texts[0]
        max_objects = 0

        for text in texts:
            object_count = sum(1 for word in text.lower().split()
                               if self.extract_object_from_text(word) is not None)
            if object_count > max_objects:
                max_objects = object_count
                best_text = text

        return best_text

    def iterative_attack(self, imgs, txts, txt2img, img_supervisions,
                         key_regions_list, key_region_classes_list, device, max_length, scales, iterations=1):
        adv_imgs = imgs.clone()
        adv_txts = txts.copy()

        modified_word_positions = [None] * len(txts)

        for iter in range(iterations):

            adv_txts, modified_word_positions = self.txt_attacker.img_guided_attack(
                self.model, adv_txts, img_embeds=img_supervisions,
                modified_word_positions=modified_word_positions, iteration=iter
            )

            with torch.no_grad():
                txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                         max_length=max_length, return_tensors="pt").to(device)
                txts_output = self.model.inference_text(txts_input)
                txt_supervisions = txts_output['text_feat']

            adv_imgs = self.img_attacker.txt_guided_attack(
                self.model, adv_imgs, txt2img, device,
                scales=scales, txt_embeds=txt_supervisions,
                key_regions_list=key_regions_list, iteration=iter
            )

            with torch.no_grad():
                adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
                img_supervisions = adv_imgs_outputs['image_feat'][txt2img]

        return adv_imgs, adv_txts


class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        loss_diversity = -torch.log(torch.std(adv_imgs_embeds, dim=0).mean())

        loss = loss_IaTcpos + 0.1 * loss_diversity

        return loss

    def apply_multiple_key_regions_weight(self, sign_grad, key_regions):
        weight = torch.ones_like(sign_grad)

        if key_regions is not None and len(key_regions) > 0:

            for key_region in key_regions:
                if key_region is not None:
                    x1, y1, x2, y2 = key_region.int()
                    x1 = max(0, min(x1, sign_grad.shape[3] - 1))
                    y1 = max(0, min(y1, sign_grad.shape[2] - 1))
                    x2 = max(x1 + 1, min(x2, sign_grad.shape[3]))
                    y2 = max(y1 + 1, min(y2, sign_grad.shape[2]))

                    if x2 > x1 and y2 > y1:
                        h, w = y2 - y1, x2 - x1
                        y_center, x_center = h // 2, w // 2
                        y_coords = torch.arange(h).float() - y_center
                        x_coords = torch.arange(w).float() - x_center
                        y_mesh, x_mesh = torch.meshgrid(y_coords, x_coords)
                        gaussian_mask = torch.exp(-(x_mesh ** 2 + y_mesh ** 2) / (2 * (min(h, w) / 4) ** 2))
                        gaussian_mask = gaussian_mask.to(sign_grad.device)

                        weight[:, :, y1:y2, x1:x2] *= (1 + (2 - 1) * gaussian_mask.unsqueeze(0).unsqueeze(0))

        sign_grad = sign_grad * weight
        return sign_grad

    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None,
                          txt_embeds=None, key_regions_list=None, iteration=0):
        model.eval()
        b, c, h, w = imgs.shape

        if scales is None:
            scales_num = 1
            scales = []
        else:
            scales_num = len(scales) + 1

        scales_num = 1
        scales = []

        adaptive_step_size = self.step_size * (1 + 0.1 * iteration)

        adv_imgs = imgs.detach() + torch.from_numpy(
            np.random.uniform(-self.eps, self.eps, imgs.shape)
        ).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        for step in range(self.steps):
            adv_imgs.requires_grad_()

            scaled_imgs = self.get_enhanced_scaled_imgs(adv_imgs, scales, device, key_regions_list)
            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.inference_image(scaled_imgs)
            adv_imgs_embeds = adv_imgs_output['image_feat']

            model.zero_grad()
            with torch.enable_grad():
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                loss_list = []
                for i in range(scales_num):
                    scale_feat = adv_imgs_embeds[i * b: (i + 1) * b]
                    loss_item = self.loss_func(scale_feat, txt_embeds, txt2img)
                    loss_list.append(loss_item.item())
                    loss += loss_item
            loss.backward()
            grad = adv_imgs.grad.data

            grad_norm = torch.norm(grad.view(b, -1), p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(b, 1, 1, 1) + 1e-8)

            sign_grad = grad_normalized.sign()

            if key_regions_list is not None:
                for img_idx in range(b):
                    key_regions = key_regions_list[img_idx] if img_idx < len(key_regions_list) else None
                    sign_grad[img_idx:img_idx + 1] = self.apply_multiple_key_regions_weight(
                        sign_grad[img_idx:img_idx + 1], key_regions)


            perturbation = adaptive_step_size * sign_grad.sign()

            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        return adv_imgs

    def get_enhanced_scaled_imgs(self, imgs, scales=None, device='cuda', key_regions_list=None):

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        result = [imgs]

        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            reverse_transform = transforms.Resize(ori_shape,
                                                  interpolation=transforms.InterpolationMode.BICUBIC)

            scaled_batch = []
            for img_idx, img in enumerate(imgs):
                if key_regions_list is not None and img_idx < len(key_regions_list):
                    key_regions = key_regions_list[img_idx]
                    local_enhanced = self.enhance_multiple_key_regions_with_augmentation(
                        img, key_regions, scale_transform, reverse_transform, device
                    )
                    scaled_batch.append(local_enhanced)
                else:
                    enhanced_img = self.apply_random_augmentation(img.unsqueeze(0))
                    enhanced_img = scale_transform(enhanced_img)
                    enhanced_img = reverse_transform(enhanced_img)
                    scaled_batch.append(enhanced_img.squeeze(0))

            scaled_batch = torch.stack(scaled_batch)

            noise_std = 0.02 + 0.01 * random.random()
            scaled_batch = scaled_batch + torch.randn_like(scaled_batch) * noise_std
            scaled_batch = torch.clamp(scaled_batch, 0.0, 1.0)
            result.append(scaled_batch)

        return torch.cat(result, 0)

    def apply_random_augmentation(self, img):

        aug_type = random.choice(['brightness', 'contrast', 'rotation', 'none'])

        if aug_type == 'brightness':
            factor = 0.8 + 0.4 * random.random()
            return torch.clamp(img * factor, 0.0, 1.0)
        elif aug_type == 'contrast':
            factor = 0.8 + 0.4 * random.random()
            mean = img.mean()
            return torch.clamp((img - mean) * factor + mean, 0.0, 1.0)
        elif aug_type == 'rotation':
            angle = random.uniform(-5, 5)
            return transforms.functional.rotate(img, angle)
        else:
            return img

    def enhance_multiple_key_regions_with_augmentation(self, img, key_regions, scale_transform, reverse_transform,
                                                       device):
        enhanced_img = img.clone()

        for key_region in key_regions:
            if key_region is not None:
                x1, y1, x2, y2 = key_region.int()

                x1 = max(0, min(x1, img.shape[2] - 1))
                y1 = max(0, min(y1, img.shape[1] - 1))
                x2 = max(x1 + 1, min(x2, img.shape[2]))
                y2 = max(y1 + 1, min(y2, img.shape[1]))

                if x2 > x1 and y2 > y1:
                    key_region_patch = img[:, y1:y2, x1:x2].unsqueeze(0)
                    enhanced_patch = self.apply_random_augmentation(key_region_patch)

                    scaled_region = scale_transform(enhanced_patch)
                    noise = torch.normal(
                        mean=0.0,
                        std=0.05,
                        size=scaled_region.shape,
                        device=scaled_region.device,
                        dtype=scaled_region.dtype
                    )

                    scaled_region = scaled_region + noise

                    restored_region = reverse_transform(scaled_region).squeeze(0)

                    restored_h, restored_w = restored_region.shape[1], restored_region.shape[2]
                    if restored_h != (y2 - y1) or restored_w != (x2 - x1):
                        region_resize = transforms.Resize((y2 - y1, x2 - x1),
                                                          interpolation=transforms.InterpolationMode.BICUBIC)
                        restored_region = region_resize(restored_region.unsqueeze(0)).squeeze(0)


                    enhanced_img[:, y1:y2, x1:x2] = restored_region

        return enhanced_img


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls

    def img_guided_attack(self, net, texts, img_embeds=None, modified_word_positions=None, iteration=0):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        new_modified_positions = []

        for text_idx, text in enumerate(texts):
            already_modified_position = modified_word_positions[text_idx] if modified_word_positions else None

            if iteration > 0 and already_modified_position is not None:
                final_adverse.append(text)
                new_modified_positions.append(already_modified_position)
                continue

            sensitive_scores = self.get_sensitive_scores(text, net, img_embeds, text_idx)

            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[text_idx], self.batch_size,
                                                         self.max_length)
            combined_scores = []
            for i in range(len(important_scores)):
                combined_score = 0.5 * important_scores[i].item() + 0.5 * sensitive_scores[i]
                combined_scores.append(combined_score)

            list_of_index = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0
            modified_position = None

            for top_index in list_of_index:
                if change >= 1:
                    break

                if already_modified_position is not None and top_index[0] != already_modified_position:
                    continue

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[text_idx, keys[top_index[0]][0]:keys[top_index[0]][1]]
                word_pred_scores = word_pred_scores_all[text_idx, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                substitutes = substitutes[:15]

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]

                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue
                    if '##' in substitute:
                        continue

                    if substitute in filter_words:
                        continue

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))

                best_substitute_idx = self.select_best_substitute(net, replace_texts, img_embeds, text_idx,
                                                                  available_substitutes)

                final_words[top_index[0]] = available_substitutes[best_substitute_idx]

                if available_substitutes[best_substitute_idx] != tgt_word:
                    change += 1
                    modified_position = top_index[0]
                    print(f" {text_idx + 1}: '{tgt_word}' --> '{available_substitutes[best_substitute_idx]}'")

            final_adverse.append(' '.join(final_words))
            new_modified_positions.append(modified_position)

        return final_adverse, new_modified_positions

    def select_best_substitute(self, net, replace_texts, img_embeds, text_idx, available_substitutes):

        device = next(net.parameters()).device

        replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                            max_length=self.max_length, return_tensors='pt').to(device)
        replace_output = net.inference_text(replace_text_input)

        replace_feat = replace_output['text_feat']
        if replace_feat.dim() == 3:
            replace_embeds = replace_feat[:, 0, :]
        else:
            replace_embeds = replace_feat

        match_scores = F.cosine_similarity(replace_embeds, img_embeds[text_idx].unsqueeze(0), dim=-1)

        best_idx = match_scores.argmin()

        return best_idx

    def get_sensitive_scores(self, text, net, img_embeds, text_idx):

        words = text.split()
        if len(words) == 0:
            return []

        device = next(net.parameters()).device

        original_input = self.tokenizer([text], padding='max_length', truncation=True,
                                        max_length=self.max_length, return_tensors='pt').to(device)
        original_output = net.inference_text(original_input)


        original_feat = original_output['text_feat']
        if original_feat.dim() == 3:
            original_embed = original_feat[:, 0, :].detach()
        else:
            original_embed = original_feat.detach()

        original_match_score = F.cosine_similarity(original_embed, img_embeds[text_idx], dim=-1).item()

        sensitive_scores = []

        for i in range(len(words)):
            temp_words = words.copy()
            temp_words[i] = "[UNK]"
            temp_text = " ".join(temp_words)
            temp_input = self.tokenizer([temp_text], padding='max_length', truncation=True,
                                        max_length=self.max_length, return_tensors='pt').to(device)
            temp_output = net.inference_text(temp_input)

            temp_feat = temp_output['text_feat']
            if temp_feat.dim() == 3:
                temp_embed = temp_feat[:, 0, :].detach()
            else:
                temp_embed = temp_feat.detach()

            match_score = F.cosine_similarity(temp_embed, img_embeds[text_idx], dim=-1).item()

            sensitive_score = original_match_score - match_score
            sensitive_scores.append(max(sensitive_score, 0))  # 确保非负

        return sensitive_scores

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue
                    if '##' in substitute:
                        continue

                    if substitute in filter_words:
                        continue

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))

                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))
                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device
        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')
        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()

    if sub_len == 0:
        return words
    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words