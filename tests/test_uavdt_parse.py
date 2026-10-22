# SPDX-License-Identifier: Apache-2.0
"""UAVDT parsing on a tiny fake release (no dataset needed)."""
from uav_shiftlab.datasets import uavdt


def _fake(tmp_path):
    img, gt, attr = tmp_path / "UAV-benchmark-M", tmp_path / "GT", tmp_path / "M_attr"
    for split, seq, vals in (("train", "M0101", "1,0,0,1,0,0,1,0,0,0"), ("test", "M0203", "0,1,0,0,1,0,0,1,0,0")):
        (attr / split).mkdir(parents=True, exist_ok=True)
        (attr / split / f"{seq}_attr.txt").write_text(vals)
        (img / seq).mkdir(parents=True, exist_ok=True)
        for fr in (1, 2, 3):
            (img / seq / f"img{fr:06d}.jpg").write_bytes(b"")
    gt.mkdir()
    (gt / "M0203_gt_whole.txt").write_text("1,7,10,20,30,40,1,1,1\n3,8,5,5,8,8,1,1,3\n")
    (gt / "M0203_gt_ignore.txt").write_text("2,1,100,100,50,50,1,-1,-1\n")
    return str(img), str(gt), str(attr)


def test_sequence_to_coco(tmp_path):
    img, gt, attr = _fake(tmp_path)
    attrs = uavdt.load_attributes(attr)
    assert attrs["M0203"] == {"split": "test", "attrs": dict(zip(uavdt.ATTR_NAMES, [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]))}
    splits = uavdt.build_splits(dict(attrs, **{s: {"split": "train", "attrs": attrs["M0101"]["attrs"]}
                                               for s in uavdt.SOURCE_VAL}))
    assert splits["test_night"] == ["M0203"] and splits["source_train"] == ["M0101"]
    images, anns, size = uavdt.build_sequence("M0203", img, gt, lambda p: (1024, 540))
    assert size == (1024, 540) and [im["frame_id"] for im in images] == [1, 2, 3]
    assert images[0] == {"id": 20300001, "file_name": "M0203/img000001.jpg", "width": 1024, "height": 540,
                         "video": "M0203", "frame_id": 1, "ignore_regions": []}
    real = [a for a in anns if not a["iscrowd"]]
    assert [(a["image_id"], a["category_id"], a["bbox"]) for a in real] == [
        (20300001, 1, [10.0, 20.0, 30.0, 40.0]), (20300003, 3, [5.0, 5.0, 8.0, 8.0])]
    crowd = [a for a in anns if a["iscrowd"]]
    assert len(crowd) == 3 and {a["category_id"] for a in crowd} == {1, 2, 3}
    assert images[1]["ignore_regions"] == [[100.0, 100.0, 50.0, 50.0]]
