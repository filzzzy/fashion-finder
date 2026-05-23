from __future__ import annotations

import torch
import torchvision
from PIL import Image, ImageDraw


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def make_retrieval_grid(
    source_imgs: torch.Tensor,
    texts: list[str],
    target_imgs: torch.Tensor,
    retrieved_imgs: torch.Tensor,
    n_top: int = 5,
) -> torch.Tensor:
    source_imgs = _denormalize(source_imgs.cpu())
    target_imgs = _denormalize(target_imgs.cpu())
    retrieved_imgs = _denormalize(retrieved_imgs.cpu())

    rows: list[torch.Tensor] = []
    for idx in range(source_imgs.shape[0]):
        source = torchvision.transforms.ToPILImage()(source_imgs[idx])
        target = torchvision.transforms.ToPILImage()(target_imgs[idx])
        tops = [torchvision.transforms.ToPILImage()(retrieved_imgs[idx, k]) for k in range(n_top)]
        width, height = source.size
        gap = 15
        canvas_w = width * (3 + n_top) + gap * (2 + n_top)
        canvas_h = height + 20

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(source, (0, 0))
        draw.text((0, height + 2), "QUERY", fill=(0, 0, 0))

        text_x = width + gap
        line = ""
        y_text = 10
        for word in texts[idx].split():
            if len(line) + len(word) > 20:
                draw.text((text_x, y_text), line, fill=(0, 0, 255))
                y_text += 15
                line = word + " "
            else:
                line += word + " "
        draw.text((text_x, y_text), line, fill=(0, 0, 255))
        draw.text((text_x, height + 2), "MODIFIER", fill=(0, 0, 255))

        target_x = (width + gap) * 2
        canvas.paste(target, (target_x, 0))
        draw.rectangle([target_x, 0, target_x + width, height], outline=(0, 200, 0), width=3)
        draw.text((target_x, height + 2), "TARGET (GT)", fill=(0, 150, 0))

        start_tops_x = (width + gap) * 3
        for rank, top_image in enumerate(tops):
            curr_x = start_tops_x + rank * (width + gap)
            canvas.paste(top_image, (curr_x, 0))
            draw.text((curr_x, height + 2), f"RANK {rank + 1}", fill=(200, 0, 0))

        rows.append(torchvision.transforms.ToTensor()(canvas))
    return torch.stack(rows)
