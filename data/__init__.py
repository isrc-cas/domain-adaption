from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    # img, img_meta, target
    batch = list(zip(*batch))
    imgs = default_collate(batch[0])
    return imgs, batch[1], batch[2]